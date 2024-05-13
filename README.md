# 02501162-math70076-assessment-2

This repository contains the project for the 2nd assessment. The project focuses on analyzing the forest fires dataset from the UCI Machine Learning Repository to predict the area of forest fires in the Montesinho National Park. The goal is to explore various machine learning models to predict the forest fire area and understand the main drivers. Additionally, this repository contains a file to plot the predicted area based on specific input variables. The full report is located in the outputs folder, as well as this main root folder.

## Table of Contents

- [Data](#data)
- [Analyses](#analyses)
- [Outputs](#outputs)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Data

The dataset used in this project is sourced from the UCI Machine Learning Repository. It includes various features related to weather conditions, time/season and geographical location, among others, the data directory contains both the raw and processed data.

## Analyses

This section contains the analyses performed on the dataset, including exploratory data analysis (EDA), feature selection, and model training and evaluation. Moreover, it contains [forecast.py](./analyses/forecast.py) , which allows forecasted values for specific user input. Additionally, it contains a Bayesian regression with PyMC in the file [bayes_model](./analyses/bayes_model.py), which is not used for predictive inference but might still prove useful because tutorials on Bayesian Gamma regression are often parametrized incorrectly. Apart from regression coefficients or built-in feature importance tools, Partial Dependence Plots (PDPs) are used to understand how the variables affect the outcome

## Outputs

The outputs folder contains the results of the analyses, including visualizations, model predictions, and performance metrics.

## Reports

The reports folder contains the .tex file used to create the report, as well as the final report.

## Installation

To run this project locally, follow these steps:

1. Clone the repository: git clone https://github.com/silaskoemen/02501162-math70076-assessment-2.git
2. Navigate to the project directory: cd 02501162-math70076-assessment-2
3. Install required packages (specific versions for saving and loading of models: pip install -r requirements.txt OR pip3 install -r requirements.txt

## Usage

The data folder contains the raw and processed data. Any of the model training uses the processed data directly. Moreover, the final trained models are saved and loaded for further analysis, so models do not have to be retrained. Should you want to retrain the models or add new ones, just follow the outline of [models.py](./analyses/models.py), but pay close attention to the sklearn and joblib versions in [requirements.txt](./src/requirements.txt) so loading and saving works! The forecasting tool also just uses the final, trained models directly under [saved_models](./src/saved_models)

## Contributing

Contributions are welcome. Please feel free to submit a pull request or open an issue if you encounter problems.

## License

This project is licensed under the MIT License. See the `License.txt` file for details.
      
