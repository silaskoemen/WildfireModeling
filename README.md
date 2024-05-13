# 02501162-math70076-assessment-2

This repository contains the assessment 2 project for the course Math70076. The project focuses on analyzing the forest fires dataset from the UCI Machine Learning Repository to predict the area of forest fires in the Montesinho National Park. The goal is to explore various machine learning models to predict the forest fire area and understand the main drivers. Additionally, this repository contains a file to plot the predicted area based on specific input variables. The full report is located in the outputs folder, as well as this main root folder.

## Table of Contents

- [General Info](#general-info)
- [Data](#data)
- [Analyses](#analyses)
- [Outputs](#outputs)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## General Info

This project utilizes the forest fires dataset available at the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/162/forest+fires). The aim is to apply different machine learning models and data science techniques to predict the area and location of forest fires, and identify the key variables influencing the outcome.

## Data

The dataset used in this project is sourced from the UCI Machine Learning Repository. It includes various features related to weather conditions and geographical location, among others.

## Analyses

This section contains the analyses performed on the dataset, including exploratory data analysis (EDA), feature selection, and model training and evaluation. Moreover, it contains forecast.py, which allows forecasted values for specific user input. Additionally, it contains a Bayesian regression with PyMC, which is not used for predictive inference but might still prove useful because tutorials on Bayesian Gamma regression are often parametrized incorrectly. Apart from regression coefficients or built-in feature importance tools, Partial Dependence Plots (PDPs) are used to understand how the variables affect the outcome

## Outputs

The outputs folder contains the results of the analyses, including visualizations, model predictions, and performance metrics. Most importantly, the full report, which explains all steps taken and conclusions reached, is here next to the root folder.

## Installation

To run this project locally, follow these steps:

1. Clone the repository: git clone https://github.com/silaskoemen/02501162-math70076-assessment-2.git
2. Navigate to the project directory: cd 02501162-math70076-assessment-2
3. Install required packages (specific versions for saving and loading of models: pip install -r requirements.txt OR pip3 install -r requirements.txt

## Usage

The data folder contains the raw and processed data. Any of the model training uses the processed data directly. Moreover, the final trained models are saved and loaded for further analysis, so models do not have to be retrained. Should you want to retrain the models or add new ones, just follow the outline of models.py, but pay close attention to the sklearn and joblib versions in Requirements.txt so loading and saving works! The forecasting tool also just uses the final, trained models directly.

## Contributing

Contributions are welcome. Please feel free to submit a pull request or open an issue if you encounter problems.

## License

This project is licensed under the MIT License. See the `License.txt` file for details.
      
