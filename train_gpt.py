import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def mirror_map(w, eps):
    """
    Gradient of psi(w) = sum_j |w_j|^{1+eps}.
    Coordinate-wise: grad_psi_j(w) = (1+eps)*|w_j|^eps * sign(w_j).
    """
    return (1 + eps) * w.abs().pow(eps) * w.sign()


def mirror_map_inv(theta, eps):
    """
    Inverse map of mirror_map:
      w_j = sign(theta_j) * (|theta_j| / (1+eps))^(1/eps).
    If theta_j is zero, the corresponding w_j is zero.
    """
    # Avoid division by zero if (1+eps) is small, though typically eps>0. 
    # In practice just do (theta.abs()/(1+eps)).pow(1/eps).
    denom = (1 + eps)
    # If denom is truly 0, eps would be -1, which isn't valid. We assume eps>0.
    w = (theta.abs() / denom).pow(1.0 / eps) * theta.sign()
    return w

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.mirror_eps = 0.1  # Fixed for now
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
                    # p_world_mirror = mirror_map(p_world, self.mirror_eps)
                    # p_world_mirror.add_(g_world.view_as(p_world),
                    #              alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
                    # p_world = mirror_map_inv(p_world_mirror, self.mirror_eps)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

class SMDAdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), adamw_eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, mirror_eps=0.1, shrink_factor=0.0, zero_threshold=0.0):
        print(
            "Initializing CustomAdamWOptimizer with lr: ", lr, 
            "betas: ", betas, 
            "adamw_eps: ", adamw_eps, 
            "weight_decay: ", weight_decay, 
            "amsgrad: ", amsgrad,
            "mirror_eps: ", mirror_eps,
            "shrink_factor: ", shrink_factor,
            "zero_threshold: ", zero_threshold
        )
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if adamw_eps < 0.0:
            raise ValueError(f"Invalid epsilon: {adamw_eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, adamw_eps=adamw_eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
            mirror_eps=mirror_eps, 
            shrink_factor=shrink_factor, 
            zero_threshold=zero_threshold,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            adamw_eps = group['adamw_eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            mirror_eps = group['mirror_eps']
            shrink_factor = group['shrink_factor']
            zero_threshold = group['zero_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1
                step = state['step']

                # Apply AdamW weight decay directly to the parameters
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute max of second moment if using AMSGrad
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(adamw_eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(adamw_eps)

                # Bias corrections
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                 # 1) Dual = grad_psi(w)
                theta = mirror_map(p.data, mirror_eps)
                
                # 2) Dual update: theta <- theta - lr * grad_L(w)
                # theta = theta - lr * p.grad
                theta.addcdiv_(exp_avg, denom, value=-step_size)

                # # 2.5) Dual update: shrink(theta)
                # # shrink = lr*shrink_factor/(p.data.numel()**0.5) # factor based on earlier experiments
                shrink = lr*shrink_factor
                theta = torch.sign(theta) * torch.maximum(
                    theta.abs() - shrink, torch.tensor(0.0)
                )
                # 3) Primal = grad_psi^*(theta)
                p_update = mirror_map_inv(theta, mirror_eps)

                # # Apply update - Normal AdamW
                # p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if zero_threshold > 0 and step > 200:
                    # Only update if the value is above the threshold, otherwise perma-zero
                    p.data = torch.where(p.data.abs() >= zero_threshold, p_update, torch.tensor(0.0))
                else:
                    p.data = p_update

        return loss

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=15/self.head_dim).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean')
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main
@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48*1024 # FlexAttention sequence length
    val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    # optimization
    num_iterations = 1770 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
    adam_head_lr = 0.22
    adam_embed_lr = 0.6
    adam_scalar_lr = 0.04
    muon_lr = 0.05
    muon_momentum = 0.95

def run_experiment(
    adam_head_lr = 0.22,
    adam_embed_lr = 0.6,
    adam_scalar_lr = 0.04,
    muon_lr = 0.05,
    muon_momentum = 0.95,
    cooldown_frac = 0.4,
    iter_limit = 1770,
):

    args = Hyperparameters()
    args.cooldown_frac = cooldown_frac
    args.adam_head_lr = adam_head_lr
    args.adam_embed_lr = adam_embed_lr
    args.adam_scalar_lr = adam_scalar_lr
    args.muon_lr = muon_lr
    args.muon_momentum = muon_momentum
    # args.num_iterations = args.num_iterations
    args.num_iterations = iter_limit

    print("HYPERPARAMETERS:")
    print("adam_head_lr: ", args.adam_head_lr)
    print("adam_embed_lr: ", args.adam_embed_lr)
    print("adam_scalar_lr: ", args.adam_scalar_lr)
    print("muon_lr: ", args.muon_lr)
    print("muon_momentum: ", args.muon_momentum)
    print("cooldown_frac: ", args.cooldown_frac)
    print("num_iterations: ", args.num_iterations)
    

    # torchrun sets these env variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # changes for running on 2x4090
    desired_world_size = 8
    world_size_factor = desired_world_size // world_size
    original_seq_len = args.train_seq_len
    args.train_seq_len = 16 * 1024
    args.val_seq_len = 16 * 1024 
    gradient_accumulation_steps = (original_seq_len * world_size_factor) // args.train_seq_len
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    assert world_size * world_size_factor == desired_world_size, f"This code is designed for 8xH100. {world_size * world_size_factor=} != {desired_world_size=}"
    assert gradient_accumulation_steps * args.train_seq_len == desired_world_size * original_seq_len, f"{gradient_accumulation_steps * args.train_seq_len=} != {desired_world_size * original_seq_len=}"
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master_process = (rank == 0) # this process will do logging, checkpointing etc.

    # begin logging
    logfile = None
    if master_process:
        run_id = uuid.uuid4()
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)
    def print0(s, console=False):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)

    # begin by printing this file (the Python code)
    print0(code)
    print0("="*100)
    # log information about the hardware/software environment this is running on
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    def nvidia_smi():
        import subprocess  # avoid top level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi())
    print0("="*100)

    ########################################
    #    Construct model and optimizer     #
    ########################################

    model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768,
                        max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    # collect the parameters to optimize
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    # Break hidden matrices into 2 parts:
    # hidden_matrix_params     = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    attn_matrix_params = [p for n, p in model.blocks.named_parameters() if "attn" in n and p.ndim >= 2 and "embed" not in n]
    mlp_matrix_params = [p for n, p in model.blocks.named_parameters() if "mlp" in n and p.ndim >= 2 and "embed" not in n]


    # init the optimizer(s)
    adam_params = [dict(params=head_params, lr=adam_head_lr), dict(params=embed_params, lr=adam_embed_lr), dict(params=scalar_params, lr=adam_scalar_lr)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(attn_matrix_params, lr=muon_lr, momentum=muon_momentum, rank=rank, world_size=world_size)
    # optimizer3 = Muon(mlp_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
    # optimizer3 = SMDAdamWOptimizer(
    #     mlp_matrix_params,
    #     lr=3e-3,
    #     betas=(0.9, 0.999),
    #     adamw_eps=1e-8,
    #     weight_decay=0,
    #     amsgrad=False,
    #     mirror_eps=0.1,
    #     shrink_factor=1e-5,
    #     zero_threshold=1e-6,
    # )
    optimizer3 = Muon(mlp_matrix_params, lr=muon_lr, momentum=muon_momentum, rank=rank, world_size=world_size)
    # optimizer3 = torch.optim.Adam(mlp_matrix_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizers = [optimizer1, optimizer2, optimizer3]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def get_lr(step: int):
        x = step / args.num_iterations # progress in training
        # assert 0 <= x <= 1
        if x < 1 - args.cooldown_frac:
            return 1.0
        else:
            w = (1 - x) / args.cooldown_frac
            return w * 1.0 + (1 - w) * 0.1

    # attention window size schedule: linearly increase
    @lru_cache(1)
    def get_window_size_blocks_helper(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    def get_window_size_blocks(step: int):
        x = step / 1770 # args.num_iterations # progress in training
        assert 0 <= x <= 1
        # Linearly increase the block-wise sliding window size over training 128 -> 1792
        # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
        window_size = next_multiple_of_n(1728 * x, n=128)
        return get_window_size_blocks_helper(window_size)

    model: nn.Module = torch.compile(model, dynamic=False)

    ########################################
    #            Warmup kernels            #
    ########################################

    # Warmup the training kernels, then re-initialize the state so we aren't cheating
    warmup_steps = 10
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
    for _ in range(warmup_steps):
        inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
        model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state

    # Print model modules and parameter counts in a nice format
    def print_model_summary(model: nn.Module):
        from collections import defaultdict
        total_params = 0
        print0("="*100, console=True)
        print0(f"{'Module':<40} {'Parameters':<20}", console=True)
        print0("="*100, console=True)
        summary = defaultdict(int)
        for name, module in model.named_modules():
            module_params = sum(p.numel() for p in module.parameters())
            total_params += module_params
            print0(f"{name:<40} {module_params:<20}", console=True)
            if "mlp" in name:
                summary["mlp"] += module_params
            elif "attn" in name:
                summary["attn"] += module_params
            elif "embed" in name:
                summary["embed"] += module_params
            elif "lm_head" in name:
                summary["lm_head"] += module_params
            else:
                summary["other"] += module_params
        print0(f"{'Total':<40} {total_params:<20}", console=True)
        for k, v in summary.items():
            print0(f"{k}: {v}", console=True)
        print0("="*100, console=True)
        for k, v in summary.items():
            print0(f"{k}: {v}", console=True)

    print_model_summary(model)


    ########################################
    #        Training and validation       #
    ########################################

    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    train_steps = args.num_iterations
    val_loss = 0
    for step in range(train_steps + 1):
        last_step = (step == train_steps)

        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every >= 0 and step % args.val_loss_every == 0) or (step == 300):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = args.val_tokens // val_batch_size
            val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets, get_window_size_blocks(step))
            val_loss /= val_steps
            del val_loader
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms, lr:{get_lr(step):.4f}", console=True)
            all_weights = torch.cat([param.view(-1) for param in model.parameters()])
            if all_weights.numel() > 1_000_000:
                subsample_indices = torch.randperm(all_weights.numel(), device=all_weights.device)[:1_000_000]
                all_weights = all_weights[subsample_indices]
                all_weights = torch.abs(all_weights)
            quantiles = torch.tensor([0.10, 0.5, 0.90], device=all_weights.device)
            quantile_values = torch.quantile(all_weights, quantiles)
            print0(f"Quantiles of all_weights: 10%: {quantile_values[0]}, 50%: {quantile_values[1]}, 90%: {quantile_values[2]}", console=True)        
            mlp_weights = [param.view(-1) for name, param in model.named_parameters() if "mlp" in name]
            if mlp_weights:
                concatenated_mlp_weights = torch.cat(mlp_weights)
                if concatenated_mlp_weights.numel() > 1_000_000:
                    subsample_indices = torch.randperm(concatenated_mlp_weights.numel(), device=concatenated_mlp_weights.device)[:1_000_000]
                    concatenated_mlp_weights = concatenated_mlp_weights[subsample_indices]
                    concatenated_mlp_weights = torch.abs(concatenated_mlp_weights)
                mlp_quantile_values = torch.quantile(concatenated_mlp_weights, quantiles)
                print0(f"Quantiles of mlp_weights: 10%: {mlp_quantile_values[0]}, 50%: {mlp_quantile_values[1]}, 90%: {mlp_quantile_values[2]}", console=True)
            if mlp_weights:
                non_zero_mlp_weights = concatenated_mlp_weights[concatenated_mlp_weights > 1e-6]
                percent_non_zero_mlp_weights = (non_zero_mlp_weights.numel() / concatenated_mlp_weights.numel()) * 100
                print0(f"Percent non-zero mlp_weights: {percent_non_zero_mlp_weights:.2f}%", console=True)
            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        for _ in range(gradient_accumulation_steps):
            inputs, targets = next(train_loader)
            model(inputs, targets, get_window_size_blocks(step)).backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        # set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1) # momentum warmup for muon
            group["momentum"] = (1 - frac) * (args.muon_momentum - 0.1) + frac * args.muon_momentum
        # step the optimizers
        for opt in optimizers:
            opt.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms, lr:{get_lr(step):.4f}", console=True)

    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    dist.destroy_process_group()
    return val_loss.item()

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train GPT experiment with modifiable hyperparameters."
    )
    parser.add_argument("--cooldown_frac", type=float, help="Cooldown fraction")
    parser.add_argument("--adam_head_lr", type=float, help="Adam head learning rate")
    parser.add_argument("--adam_embed_lr", type=float, help="Adam embed learning rate")
    parser.add_argument("--adam_scalar_lr", type=float, help="Adam scalar learning rate")
    parser.add_argument("--muon_lr", type=float, help="Muon learning rate")
    parser.add_argument("--muon_momentum", type=float, help="Muon momentum")
    parser.add_argument("--iter_limit", type=int, help="Number of iterations to run (if stopping early)")

    args = parser.parse_args()

    # Construct the params dict only for passed arguments.
    params = {}
    if args.cooldown_frac is not None:
        params["cooldown_frac"] = args.cooldown_frac
    if args.adam_head_lr is not None:
        params["adam_head_lr"] = args.adam_head_lr
    if args.adam_embed_lr is not None:
        params["adam_embed_lr"] = args.adam_embed_lr
    if args.adam_scalar_lr is not None:
        params["adam_scalar_lr"] = args.adam_scalar_lr
    if args.muon_lr is not None:
        params["muon_lr"] = args.muon_lr
    if args.muon_momentum is not None:
        params["muon_momentum"] = args.muon_momentum
    if args.iter_limit is not None:
        params["iter_limit"] = args.iter_limit

    # Call run_experiment with the parameters that were provided.
    run_experiment(**params)

if __name__ == "__main__":
    main()