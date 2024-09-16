# Import libraries
import time
from modules.data_processing import data_processing
from modules.model_selection import model_selection
from modules.model_training import model_training
from modules.forward_simulation import forward_simulation
from modules.simulation_analysis import simulation_analysis

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
simulation_analysis(station_years=[])
print('Successfully analyzed simulation results...')

# Print execution time
end_time = time.time()
execution_time = (end_time - start_time)/60
print('Script finalized.\nExecution time: {:.3g} minutes.'.format(execution_time),
      '\nEnd time: {}'.format(time.ctime(end_time)))