Parkinson's Disease Detection using XGBoost
This project uses machine learning to predict the presence of Parkinson's Disease based on voice features. The model employs the XGBoost algorithm for classification and is trained on a dataset containing 22 voice features, such as fundamental frequency, jitter, and shimmer.

Project Overview
Parkinson’s Disease is a neurodegenerative disorder that affects movement and speech. Early diagnosis is crucial for better treatment outcomes. This model aims to help healthcare professionals detect Parkinson's Disease using voice feature analysis.

Features
Dataset: 22 voice features (e.g., fundamental frequency, jitter, shimmer) are used to predict whether an individual has Parkinson’s Disease.
Model: XGBoost algorithm for classification.
Accuracy: Achieved 94.87% accuracy after preprocessing and normalizing the data.
Technologies Used
Python: Primary programming language.
XGBoost: Algorithm used for model training and classification.
scikit-learn: For data preprocessing, splitting the dataset, and evaluating model performance.
Pandas: For data manipulation and cleaning.
Matplotlib/Seaborn: For visualizing data and model results.
Usage
Preprocess the data using the data_preprocessing.py script.
Train the model with train_model.py using the preprocessed dataset.
Evaluate the model using evaluate_model.py and review the accuracy and other metrics.
Results
Accuracy: 94.87%
Confusion Matrix: The model shows a high true positive rate, accurately predicting the presence of Parkinson’s Disease.


Accuracy: 94.87%
Precision: 0.95
Recall: 0.94
F1-score: 0.94
