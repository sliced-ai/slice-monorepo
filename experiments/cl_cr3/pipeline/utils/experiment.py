import os
import json
import logging

class ExperimentState:
    def __init__(self, experiment_name, experiment_id, config, current_iteration, current_stage):
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.config = config
        self.current_iteration = current_iteration
        self.current_stage = current_stage

def save_experiment_state(experiment_folder, experiment_id, config, current_iteration, current_stage, filename):
    experiment_state = ExperimentState(
        experiment_name=os.path.basename(experiment_folder),
        experiment_id=experiment_id,
        config=config,
        current_iteration=current_iteration,
        current_stage=current_stage
    )
    state_file = os.path.join(experiment_folder, filename)
    with open(state_file, 'w') as f:
        json.dump(experiment_state.__dict__, f)
    logging.info(f"Saved experiment state at {state_file}")

def load_latest_experiment_state(experiment_folder):
    state_file = os.path.join(experiment_folder, "latest_state.json")
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"No state file found in {experiment_folder}")
    with open(state_file, 'r') as f:
        state_dict = json.load(f)
    experiment_state = ExperimentState(
        experiment_name=state_dict['experiment_name'],
        experiment_id=state_dict['experiment_id'],
        config=state_dict['config'],
        current_iteration=state_dict['current_iteration'],
        current_stage=state_dict['current_stage']
    )
    logging.info(f"Loaded experiment state from {state_file}")
    return experiment_state
