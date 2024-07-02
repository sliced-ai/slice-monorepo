import logging
import uuid
import argparse
import os
from datetime import datetime

from utils.logging import setup_logging, log_execution
from utils.config import load_config
from utils.experiment import save_experiment_state, load_latest_experiment_state, ExperimentState

from cr3_data_module.data_setup import setup_cr3_data
from inference_module.inference import run_inference
from ae_analysis_module.autoencoder import autoencoder_analysis
from dataset_creation_module.create_dataset import create_dataset
from training_module.training import train_model
from testing_module.testing import test_and_analyze

# Setting up logging
setup_logging()

# Constants for stopping configuration
STOP_AT_STAGE = None
STOP_AFTER_ITERATIONS = None
EXPERIMENT_STATE_FILENAME = "latest_state.json"

# Helper function to generate unique experiment IDs
def generate_experiment_id():
    return str(uuid.uuid4())

# Helper function to check and stop at a specific stage
def check_stop(stage, current_iteration):
    if STOP_AT_STAGE == stage and (STOP_AFTER_ITERATIONS is None or current_iteration + 1 >= STOP_AFTER_ITERATIONS):
        logging.info(f"Stopped after stage: {stage} at iteration {current_iteration + 1}")
        return True
    return False

@log_execution
def main_pipeline(experiment_name, cr3_name, datalimit, config_path, stop_at_stage=None, stop_after_iterations=None, resume_from=None):
    global STOP_AT_STAGE, STOP_AFTER_ITERATIONS
    STOP_AT_STAGE = stop_at_stage
    STOP_AFTER_ITERATIONS = stop_after_iterations

    if resume_from:
        experiment_state = load_latest_experiment_state(resume_from)
        config = experiment_state.config
        current_iteration = experiment_state.current_iteration
        experiment_folder = resume_from
        experiment_id = experiment_state.experiment_id
    else:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config = load_config(config_path)
        current_iteration = 0
        experiment_id = generate_experiment_id()
        current_date = datetime.now().strftime("%Y%m%d")
        experiment_folder = os.path.join("experiments", f"{experiment_name}")

        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
    
    logging.info(f"Starting experiment '{experiment_name}' with ID: {experiment_id}")
    logging.info(f"Experiment data will be saved in: {experiment_folder}")

    num_iterations = config['global']['num_iterations']

    try:
        for step in range(current_iteration, num_iterations):
            logging.info(f"Pipeline iteration {step + 1}/{num_iterations}")
            
            # Step 1: Setup raw data from CR3
            cr3_data = setup_cr3_data(cr3_name, datalimit, config['cr3_data'])
            if check_stop("cr3_data", step):
                save_experiment_state(experiment_folder, experiment_id, config, step, "cr3_data", EXPERIMENT_STATE_FILENAME)
                break
            
            # Step 2: Inference and embedding
            inference_embeddings_path = run_inference(config['inference_and_embedding'],experiment_name,step+1)
            if check_stop("inference_and_embedding", step):
                save_experiment_state(experiment_folder, experiment_id, config, step, "inference_and_embedding", EXPERIMENT_STATE_FILENAME)
                break
            
            # Step 3: Autoencoder training and result analysis
            ae_analysis, selected_responses = autoencoder_analysis(config['ae_analysis'],experiment_name,step+1,datalimit)
            if check_stop("ae_analysis", step):
                save_experiment_state(experiment_folder, experiment_id, config, step, "ae_analysis", EXPERIMENT_STATE_FILENAME)
                break
            die
            # Step 4: Dataset creation
            dataset = create_dataset(ae_analysis, selected_responses, config['dataset_creation'])
            if check_stop("dataset_creation", step):
                save_experiment_state(experiment_folder, experiment_id, config, step, "dataset_creation", EXPERIMENT_STATE_FILENAME)
                break
            
            # Step 5: Training step on dataset
            training_metrics = train_model(dataset, config['training'])
            if check_stop("training", step):
                save_experiment_state(experiment_folder, experiment_id, config, step, "training", EXPERIMENT_STATE_FILENAME)
                break
            
            # Step 6: Test and analyze results
            test_results = test_and_analyze(training_metrics, config['testing'])
            if check_stop("testing", step):
                save_experiment_state(experiment_folder, experiment_id, config, step, "testing", EXPERIMENT_STATE_FILENAME)
                break
            
            # Save iteration results
            iteration_folder = os.path.join(experiment_folder, f"iteration_{step + 1}")
            if not os.path.exists(iteration_folder):
                os.makedirs(iteration_folder)
            
            # Assuming results are stored in variables for each step, you would save them here
            # For example:
            # with open(os.path.join(iteration_folder, "inference_results.txt"), "w") as f:
            #     f.write(str(inference_results))
            
            logging.info(f"Iteration {step + 1} completed with test results: {test_results}")

    except Exception as e:
        logging.error(f"Error occurred: {e}. Saving state and stopping.")
        save_experiment_state(experiment_folder, experiment_id, config, step, STOP_AT_STAGE or "unknown", EXPERIMENT_STATE_FILENAME)
        raise

    logging.info("Pipeline execution completed.")
    return experiment_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the LLM training pipeline.')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('cr3_name', type=str, help='Name of the CR3 data')
    parser.add_argument('datalimit', type=int, help='Limit of data to use (0 for all data)')
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')
    parser.add_argument('--stop_at_stage', type=str, default=None, help='Stage to stop at')
    parser.add_argument('--stop_after_iterations', type=int, default=None, help='Stop after a certain number of iterations')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to the experiment folder to resume from')
    
    args = parser.parse_args()
    
    main_pipeline(args.experiment_name, args.cr3_name, args.datalimit, args.config_path, args.stop_at_stage, args.stop_after_iterations, args.resume_from)
