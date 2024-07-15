# Machine-downtime-analysis

This Streamlit application predicts the downtime probability of fuel pump machines based on various input parameters. The model uses a Random Forest Classifier to provide predictions with high accuracy.

## Features
* **Interactive Input**: Users can input various parameters using sliders and text inputs.
* **Real-time Predictions**: Provides immediate predictions on machine downtime with probabilities.
* **Visualizations**: Displays prediction results and probabilities using interactive Plotly charts.
* **Animations**: Engaging Lottie animations are included to enhance user experience.

## Input Parameters
The following parameters need to be provided for prediction:

* **Month**: Production month (1-31)
* **Hydraulic Pressure**: Enter hydraulic pressure in bars (recommended: 150-300 bars)
* **Coolant Pressure**: Enter coolant pressure in bars (recommended: 1-1.5 bars)
* **Air Pressure**: Enter air pressure in bars (recommended: 6-8 bars)
* **Coolant Temperature**: Enter coolant temperature in °C (recommended: 85-95°C)
* **Hydraulic Oil Temperature**: Enter hydraulic oil temperature in °C (recommended: 40-60°C)
* **Spindle Bearing Temperature**: Enter spindle bearing temperature in °C (recommended: 60-80°C)
* **Spindle Vibration**: Enter spindle vibration in µm (recommended: below 1 µm)
* **Tool Vibration**: Enter tool vibration in µm (recommended: below 2.5 µm)
* **Spindle Speed**: Enter spindle speed in RPM (recommended: 500-3000 RPM)
* **Voltage**: Enter voltage in volts (recommended: 12-14 volts)
* **Torque**: Enter torque in Nm (recommended: 20-30 Nm)
* **Cutting**: Enter cutting value in kN (recommended: 5-10 kN)
* **Machine ID**: Select machine ID from the available options

## Model and Data
* **Dataset**: The model uses a dataset containing various machine parameters and their corresponding downtime status.
* **Model**: A Random Forest Classifier is used for prediction.
* **Data Preprocessing**: Includes one-hot encoding for categorical features and scaling for numerical features.

## Conclusion
This application provides a robust solution for predicting machine downtime, helping to minimize unexpected failures and optimize maintenance schedules. By leveraging machine learning, the app aims to enhance operational efficiency and reduce the economic impact of machine downtime.
