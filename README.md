# SWE Hybrid Modelling
This repository contains the code used for implementing a hybrid Machine Learning (ML) modeling approach to forecast Snow Water Equivalent (SWE).

## Description
This project explores hybrid ML modeling techniques to forecast Snow Water Equivalent (SWE) from meteorolofical data by integrating snow model simulations with ML. Two approaches are evaluated: a post-processing setup, where outputs from the Crocus snow model are combined with meteorological data as predictors, and a data-augmentation setup, where the ML model is trained on both observed SWE data and simulated SWE from Crocus at additional locations. We evaluate these setups for prediction at left-out years of training stations (temporal split) and at untrained stations (station split). The standalone Crocus model and an ML approach using only measured data are compared as benchmarks.

## Project Structure

- **data/**
  - **processed/** - Processed datasets by station name and lag value
  - **raw/** - Original data sources

- **results/** - Results by type of split and lag value
  - **ts\_[bool]\_lg\_[int]/**
    - **figures/** - Generated visualizations
    - **models/** - Trained models
    - **outputs/** - Analysis outputs
  - ...

- **src/**
  - config.py - Configuration parameters
  - main.py - Main script
  - plot_results.ipynb - Results visualization notebook
  - **modules/** - Custom Python modules executed in the main script

- env_snow_lin.yml - Conda environment file

## Installation
1. Clone this repository:
```
git clone [repository-url]
cd [repository-name]
```

2. Create the conda environment using the provided YAML file:
```
conda env create -f env_snow_lin.yml
conda activate snow_project
```

3. Obtain the raw data:
  - The in-situ SWE and meteorological data can be accessed from Menard and Essery (2019) [[1]](#1)
  - The Crocus snow model simulations can be accessed by correspondance to matthieu.lafaysse@meteo.fr
  - After obtaining both sources of data, place them in the raw directory

## Usage
1. Configure parameters in src/config.py
2. Run the analysis pipeline:
```
python src/main.py
```

## Data Description
This study uses meteorological and SWE data from ten stations across the Northern Hemisphere described in Menard et al. (2019) [[2]](#2). Furthermore, snowpack simulations using the Crocus snow model for the available meteorological data were used as well. The data is processed to a daily time step with time lags of 0 and 14 days, following the naming convention df_[station]\_lag_[days].csv.

## Results
Analysis results are organized by time-series configuration (ts_True/ts_False) and lag period (lg_0/lg_14). Visualizations can be explored using the plot_results.ipynb notebook.

## License
This project is licensed under the terms of the [MIT License](https://github.com/oriol-pomarol/snow_project/blob/main/LICENSE).

## References
<a id="1">[1]</a> 
Menard, Cecile; Essery, Richard (2019): ESM-SnowMIP meteorological and evaluation datasets at ten reference sites (in situ and bias corrected reanalysis data) [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.897575, Supplement to [[2]](#2).

<a id="2">[2]</a> 
Menard, Cecile; Essery, Richard; Barr, Alan; Bartlett, Paul; Derry, Jeff; Dumont, Marie; Fierz, Charles; Kim, Hyungjun; Kontu, Anna; Lejeune, Yves; Marks, Danny; Niwano, Masashi; Raleigh, Mark; Wang, Libo; Wever, Nander (2019): Meteorological and evaluation datasets for snow modelling at 10 reference sites: description of in situ and bias-corrected reanalysis data. Earth System Science Data, 11(2), 865-880, https://doi.org/10.5194/essd-11-865-2019.