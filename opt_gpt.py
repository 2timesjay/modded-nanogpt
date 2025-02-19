from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df

# Placeholder: import your experiment function here
from train_gpt import run_experiment  # Replace with your actual import

import os

# Set envvars for distributed training
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["NODE_RANK"] = "0"


# Define Ax search space with placeholder parameters
"""
Default parameters:
adam_head_lr = 0.22
adam_embed_lr = 0.6
adam_scalar_lr = 0.04
muon_lr = 0.05
muon_momentum = 0.95
cooldown_frac = 0.4
"""

COOLDOWN_FRACTION = 0.001
ITER_LIMIT = 500

parameters_config = [
    {
        "name": "adam_head_lr",
        "type": "range",
        "value_type": "float",
        "bounds": [0.05, 1.0],
        "log_scale": True,
    },
    {
        "name": "adam_embed_lr",
        "type": "range",
        "value_type": "float",
        "bounds": [0.05, 1.0],
        "log_scale": True,
    },
    {
        "name": "adam_scalar_lr",
        "type": "range",
        "value_type": "float",
        "bounds": [0.005, 0.2],
        "log_scale": True,
    },
    {
        "name": "muon_lr",
        "type": "range",
        "value_type": "float",
        "bounds": [0.01, 1.0],
        "log_scale": True,
    },
    {
        "name": "muon_momentum",
        "type": "range",
        "value_type": "float",
        "bounds": [0.8, 0.99],
        "log_scale": False,
    },
]

# Initialize the Ax client and create an experiment
ax_client = AxClient()
ax_client.create_experiment(
    name="nanogpt_hpo",
    parameters=parameters_config,
    objectives={"loss": ObjectiveProperties(minimize=True)},
)

total_trials = 30  # Set your total number of trials

for _ in range(total_trials):
    # Ask for the next set of parameters and a trial index
    parameters, trial_index = ax_client.get_next_trial()
    print(f"Running trial {trial_index} with parameters: {parameters}")
    
    # Run the experiment with the provided parameters (replace with your actual function)
    result = run_experiment(**parameters, cooldown_frac=COOLDOWN_FRACTION, iter_limit=ITER_LIMIT)
    print(f"Trial {trial_index} result: {result}")
    
    # Tell Ax the outcome of the trial
    ax_client.complete_trial(trial_index=trial_index, raw_data={"loss": result})

df = exp_to_df(ax_client.experiment)
import os
from datetime import datetime
output_dir = "experiment_results"
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, f"results_{datetime.now().isoformat()}.csv")

print(df)

# Write the DataFrame to a CSV file
df.to_csv(output_file_path, index=False)

# Retrieve and print the best found parameters
best_parameters, best_objectives = ax_client.get_best_parameters()
print("Best parameters:", best_parameters)
print("Best objective:", best_objectives)
