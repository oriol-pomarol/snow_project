import time
import numpy as np
from modules.data_processing import data_processing
from modules.model_selection import model_selection
from modules.model_training import model_training
from modules.forward_simulation import forward_simulation
from modules.simulation_analysis import simulation_analysis
from config import cfg, paths

def generate_configs():
    
    # Define the values for the configuration parameters
    rel_weight_values = np.linspace(0.5, 2.0, 3)
    temporal_split_values = [True, False]

    # Generate the configurations and result paths
    configs = [cfg(rel_weight=(rw,), temporal_split=ts) \
               for rw in rel_weight_values for ts in temporal_split_values]
    config_names = [f"rw_{config.rel_weight}_ts_{config.temporal_split}" for config in configs]
    result_paths = [paths(results = paths.root / "results" / name) for name in config_names]

    # Create the result directories
    for path in result_paths:
        path.results.mkdir(parents=True, exist_ok=True)
        # Create the subdirectories
        for sub_dir in ['models', 'figures', 'outputs']:
            (path.results / sub_dir).mkdir(parents=True, exist_ok=True)

    return configs, result_paths

def run_with_config(config, result_path):
    # Override the global cfg and paths with the new values
    global cfg, paths
    cfg = config
    paths = result_path

    # Record starting run time
    start_time = time.time()

    # Load and process the data
    print('Loading and processing the data...')
    data_processing()
    print('Successfully loaded and processed the data...')

    # Find the best hyperparameters for the models
    print('Finding the best model type and hyperparameters...')
    model_selection()
    print('Successfully found the best model type and hyperparameters...')

    # Train the models with the different setups
    print('Training and evaluating models...')
    model_training()
    print('Successfully trained and evaluated models...')

    # Simulate SWE using ML for each station
    print('Performing forward simulation...')
    forward_simulation()
    print('Successfully performed forward simulation...')

    # Analyze the simulation results
    print('Analyzing simulation results...')
    simulation_analysis()
    print('Successfully analyzed simulation results...')

    # Print execution time
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print('Script finalized.\nExecution time: {:.3g} minutes.'.format(execution_time))

# Run the code with the specified configurations and result paths
configs, result_paths = generate_configs()
for config, result_path in zip(configs, result_paths):
    run_with_config(config, result_path)