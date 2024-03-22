# PCL Detection with Fine-Tuned HuggingFace Models

## Project Overview
This project focuses on developing a binary classification model for detecting Patronising and Condescending Language (PCL) in text. PCL, characterized by a superior attitude or compassionately patronizing manner towards others, poses a nuanced challenge in natural language processing (NLP). Utilizing a dataset on PCL towards vulnerable communities, this project leverages fine-tuned HuggingFace models to address this complexity in text analysis.

## Requirements
- Python version: 3.10
- Dependencies are listed in `requirements.txt`.

## Data Analysis
- Significant class imbalances were observed in the dataset, impacting the training process.
- PCL severity ranged from 0 (no PCL) to 4 (clear PCL), with binary classification considering 0-1 as 'no-PCL' and 2-4 as 'PCL'.
- The dataset comprised text extracts from various news media outlets across 20 countries for diversity in model training and evaluation.

## Modeling Approach
- The project utilized Decoding-enhanced BERT with disentangled attention (DeBERTa) from Microsoft, chosen for its superior attention mechanism.
- DeBERTa's ability to use content and position vectors enhances its capacity to understand contextual relationships.
- Model weights were frozen, and a binary classification head was added for specific PCL detection tasks.

## Model Improvements and Hyperparameter Tuning
- Techniques such as pre-processing, sampling, and data augmentation (e.g., synonym and contextual replacement) were explored.
- Systematic manual search was employed for hyperparameter tuning, including adjustments in learning rate and weight decay.

## Results
- The final custom DeBERTa model achieved an F1-score of 0.550 and an accuracy of 0.914.
- Context replacement coupled with upsampling emerged as the most effective strategy.
- The model's performance showed a dependency on factors like keyword categories and input sequence length.

## Conclusion and Future Work
- The project made significant strides in PCL detection, outperforming baseline models.
- Future work includes combining multiple data preprocessing and augmentation techniques for increased performance, as well as exploring ensemble methods for better generalization.

## Repository Contents
- Code for data processing and model training.
- Jupyter notebooks for analysis.
- Dataset used for the project (subject to data privacy and ethical considerations).

## How to Use
- Clone the repository and install dependencies.
- Run the Python scripts for training and evaluating the model.
- Explore the Jupyter notebooks for in-depth analysis.
