import time
from modules.data_processing import data_processing
from modules.model_selection import model_selection
from modules.model_training import model_training
from modules.forward_simulation import forward_simulation
from modules.simulation_analysis import simulation_analysis
from config import cfg, paths

# Define the values for the configuration parameters
temporal_split_values = [True, False]
lag_values = [0, 14]

# Loop over each combination of configuration parameters
for lag in lag_values:

    # Update the lag value in the configuration parameters
    cfg.lag = lag

    # Preprocess the data for the given lag value
    print('Loading and processing the data...')
    data_processing()
    print('Successfully loaded and processed the data...')

    for temporal_split in temporal_split_values:

        # Update the split type in the configuration parameters
        cfg.temporal_split = temporal_split

        # Define the new results path based on the configuration parameters
        config_name = f"ts_{cfg.temporal_split}_lg_{cfg.lag}"
        paths.results = paths.root / "results" / config_name

        # Update the other paths depending on the results path
        paths.models = paths.results / "models"
        paths.outputs = paths.results / "outputs"
        paths.figures = paths.results / "figures"
        paths.temp = paths.results / "temp"

        # Create the directories for these paths if they do not exist
        paths.models.mkdir(parents=True, exist_ok=True)
        paths.outputs.mkdir(parents=True, exist_ok=True)
        paths.figures.mkdir(parents=True, exist_ok=True)
        paths.temp.mkdir(parents=True, exist_ok=True)

        # Save the configuration to a txt file
        with open(paths.results / 'config.txt', 'w') as f:
            for key, value in cfg.__dict__.items():
                if not key.startswith('__'):
                    if callable(value):
                        value = value()
                    f.write(f'{key} = {value}\n')

        # Record starting run time
        start_time = time.time()

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