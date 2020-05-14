# Tradable Performance Standard Parameterisation
This repository explores techniques that can be used to parameterise a Tradable Performance Standard (TPS). The scheme works by establishing a sectoral emissions intensity baseline. For each unit of output a firm produces they receive an allocation of emissions permits equal to the baseline. Periodically, firms must surrender permits equal to their total emissions. Firms with emissions intensities below the baseline will accrue surplus permits for each unit they produce, while firms with emissions intensities above the baseline will have to purchase additional permits to meet their obligations under the scheme. The trading of permits causes an emissions price to arise, with the scheme facilitating transfers from relatively polluting plant to relatively cleaner technologies. 

This repository contains code describing a mathematical framework that can be used to calibrate the emissions intensity baseline to achieve price targeting objectives.


## Overview
An overview of the repository's contents is as follows:

| Folder name | Description|
| ----- | - |
|`src/1_create_scenarios` | Used to process data used in the calibration protocol|
|`src/2_parameter_selector` | Develops an optimisation model used to calibrate the emissions intensity baseline to achieve price targeting objectives|
|`src/3_process_results` | Extracts results from model output for analysis|
|`src/4_plotting` | Visualises model results|

## Zenodo link
Network and generator datasets used in this analysis are obtained from the following repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1326942.svg)](https://doi.org/10.5281/zenodo.1326942)