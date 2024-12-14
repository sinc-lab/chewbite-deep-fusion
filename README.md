Chewbite Deep Fusion Experimentation
====================================

This repository contains all experiments, settings and data processing logic in order to explore different approaches and architectures for a fusion model, capable of detecting and classifying jaw movements from grazing cattle based on acoustic and movement signals.

Installation
------------

This project is Python>=3.10 based. In order to install it locally run the following steps (Ubuntu/MacOS):

1. Clone this repository in your machine using SSH (recommended) or HTTPS.
    - For SSH: git clone git@github.com:sinc-lab/chewbite-deep-fusion.git
    - For HTTPS: https://github.com/sinc-lab/chewbite-deep-fusion.git

Note: We strongly recommend using virtual environments (Python or conda), so creating and/or activating it should be done at this point. For example, using conda virutal environments manager:

    1. conda create --name fusion-cve python=3.10

    2. conda activate fusion-cve

2. Move into repo folder: cd chewbite-deep-fusion

3. Run package setup: pip install -e . --no-cache-dir

Settings
--------

Define paths and settings in the following files:

- src/chewbite_fusion/data/settings.py
- src/chewbite_fusion/experiments/settings.py

Running experiments
-------------------

The use of an external tool called YAER (Yet Another Experiment Runner) allows you to easily run one or several experiments from command line.
In order to do so, you must move into project directory "src/chewbite_fusion" and then run: yaer run -e [experiment_name] -d

For example:
- Running one experiment: yaer run -e bloch2023_v6 -d
- Running three different experiments: yaer run -e rf_v4 -e rf_v5 -e rf_v6 -d
- Running all experiments matching a given regular expression: yaer run -re 'deep_fusion_data_level' -d

Note: Some experiments run only on GPU devices.

Project Organization
--------------------
```
├── LICENSE
├── README.md            <- The top-level README for developers using this project.
│
└── src                  <- Experimental module.
    └── chewbite_fusion  <- Experimental module source.
         ├── data           <- Scripts to generate and transform data.
         ├── experiments    <- Scripts to code experiments that will run with yaer.
         ├── features       <- Scripts to turn raw data into features for modeling (for traditional models).
         └── models      <- Scripts to turn raw data into features for modeling (for traditional models).
```
