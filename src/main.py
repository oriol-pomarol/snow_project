import time
from pathlib import Path
from modules.data_processing import data_processing
from modules.model_selection import model_selection
from modules.model_training import model_training
from modules.forward_simulation import forward_simulation
from modules.simulation_analysis import simulation_analysis
from config import cfg, paths

def generate_configs():
    
    # Define the values for the configuration parameters
    temporal_split_values = [True, False]
    lag_values = [0, 14]

    # Generate the configurations and result paths
    configs = [cfg(temporal_split=ts, lag=lg) for ts in temporal_split_values for lg in lag_values]
    config_names = [f"ts_{config.temporal_split}_lg_{config.lag}" for config in configs]
    root_path = Path(__file__).resolve().parents[1]
    result_paths = [root_path / "results" / name for name in config_names]

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

# Generate the configurations and result paths
configs, result_paths = generate_configs()

for config, result_path in zip(configs, result_paths):

    # Create the results directories if missing
    for sub_dir in ['models', 'figures', 'outputs']:
        (result_path / sub_dir).mkdir(parents=True, exist_ok=True)

    # Save the configuration to a txt file
    with open(result_path / 'config.txt', 'w') as f:
        f.write(str(config))

    # Run the code
    run_with_config(config, result_path)