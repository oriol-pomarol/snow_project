# Import libraries
import time
from modules.data_preprocessing import data_preprocessing
from modules.data_preprocessing import data_preprocessing
from modules.model_training import model_training
from modules.forward_simulation import forward_simulation

# Record starting run time
start_time = time.time()

# # Load and preprocess data
# print('Loading and preprocess the data...')
# data_preprocessing()
# print('Successfully loaded and preprocessed the data...')

# # Train the models with the different setups
# print('Training models...')
# model_training()
# print('Successfully trained models...')

# Test the models by use of the forward simulation
print('Performing forward simulation...')
forward_simulation()
print('Successfully performed forward simulation...')


station_years=['cdp_2002','rme_2002', 'wfj_2002']

# Print execution time
end_time = time.time()
execution_time = (end_time - start_time)/60
print('Script finalized.\nExecution time: {:.3g} minutes.'.format(execution_time),
      '\nEnd time: {}'.format(time.ctime(end_time)))