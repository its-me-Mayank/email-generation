# Email Text Generation using TensorFlow and Keras

Author: Mayank Sharma

## Overview
This Python script showcases the application of TensorFlow and Keras to construct a language model tailored for generating text based on the [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset). The comprehensive functionality spans data preprocessing, model training, and text generation, providing a versatile tool for experimenting with natural language generation tasks.

## Features
- **Data Loading and Preprocessing:** Utilizes the pandas library to load the first 1000 rows of email data from the [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset), focusing on the "message" column. Text data is preprocessed to lowercase for standardization.
- **Model Training:** Employs a recurrent neural network (RNN) architecture with an LSTM layer for training the language model. The model is trained on the preprocessed email text data, and both the model and tokenizer are saved for later use.
- **Text Generation:** Implements a function to generate text based on a seed text and the trained language model. The generated text reflects the learned patterns from the email data.
- **Model Evaluation:** Evaluates the accuracy of the trained model on a subset of the test data, providing insights into its performance.

## Requirements
Ensure the following Python libraries are installed:
- pandas
- numpy
- pickle
- tensorflow
- scikit-learn

You can install the required dependencies using the command:
```bash
pip install -r requirements.txt

Usage

Run the script using the command: python email_text_generation.py.
The script loads, preprocesses, and trains on the email data. It then evaluates the model's accuracy on a subset of the test data and generates sample text based on a seed text.
Feel free to customize the script parameters, explore different datasets, or adapt the code for other text generation tasks.
Notes
The script is designed to work with a CSV file named "emails.csv" from the Enron Email Dataset containing an appropriate "message" column.
Make sure to customize the script or adapt the model architecture for specific text generation requirements.
