# Tradable Performance Standard Parameterisation
This repository explores techniques that can be used to parameterise a Tradable Performance Standard (TPS) [1]. The scheme works by establishing a sectoral emissions intensity baseline, with firms receiving an allocation of emissions permits equal to the baseline for each unit they produce. In the context of a power sector, generators receive permits for each MWh outputted. Firms must periodically surrender permits equal to their total emissions. Note that firms with emissions intensities below the baseline will accrue surplus permits for each unit they produce, while firms with emissions intensities above the baseline will have to purchase additional permits to meet their obligations under the scheme. The trading of permits causes an emissions price to arise, with the scheme facilitating transfers from emissions intensive plant to 'cleaner' technologies. 

Code within this repository describes a mathematical framework that can be used to calibrate the emissions intensity baseline to achieve price targeting objectives under a TPS.

## Overview
An overview of the repository's contents is as follows:

| Folder name | Description|
| ----- | - |
|`src/1_create_scenarios` | Used to process data used in the calibration protocol|
|`src/2_parameter_selector` | Develops an optimisation model used to calibrate the emissions intensity baseline to achieve price targeting objectives|
|`src/3_plotting` | Visualises model results|

## Zenodo link
Network and generator datasets used in this analysis are obtained from the following repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1326942.svg)](https://doi.org/10.5281/zenodo.1326942)

## References
[1] D. Burtraw, J. Linn, K. Palmer, A. Paul, The costs and consequences of Clean Air Act Regulation of CO2 from power plants, American Economic Review 104 (5) (2014) 557-562.
