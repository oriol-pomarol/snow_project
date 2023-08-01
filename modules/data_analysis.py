import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_analysis(meteo_data, snow_data):
  # Convert the start and end dates to numpy arrays
  start_date_meteo = meteo_data['Date/time start'].to_numpy()
  end_date_meteo = meteo_data['Date/time end'].to_numpy()
  start_date_snow = snow_data['Date/time start'].to_numpy()
  end_date_snow = snow_data['Date/time end'].to_numpy()

  # Find the global start and end date as well as duration
  start_date_all = np.maximum(start_date_meteo, start_date_snow)
  end_date_all = np.minimum(end_date_meteo, end_date_snow)
  duration = end_date_all - start_date_all

  # Obtain and shorten if necessary the station names
  station_names = meteo_data['Event 2'].copy()
  station_names[6] = 'SBBSA'
  station_names[8] = 'SAPS'

  # Create the plot
  fig, ax = plt.subplots(figsize=(8, 4))

  # Plot the bars
  bar_positions = np.arange(len(station_names)) + 0.4
  ax.bar(bar_positions, duration, align='center', width=0.8, edgecolor='black',
        linewidth=0.5, bottom=start_date_all)

  # Label the axes
  ax.set_xticks(bar_positions)
  ax.set_xticklabels(station_names, rotation=45, ha='right')
  ax.set_ylabel('Year')
  # ax.set_xlabel('Station')

  # Add grid lines
  ax.set_axisbelow(True)
  ax.grid(axis='y', linestyle='--')

  # Title of the plot
  ax.set_title('Date Ranges')

  # Adjust the y-axis limits
  ax.set_ylim(start_date_all.min() - 1, end_date_all.max() + 1)

  # Display the plot
  plt.tight_layout()
  plt.savefig('results/station_availability.png')