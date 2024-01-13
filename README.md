# Credit Card Fraud Detection

This repository focuses on credit card fraud detection, employing various techniques, including deep learning models, random forest, decision tree, and sampling methods. The primary goal is to develop effective models for identifying fraudulent transactions in credit card datasets.

## Step 1: Importing Libraries and Data
The initial step involves importing essential libraries such as pandas, numpy, matplotlib, seaborn, and TensorFlow. The dataset (creditcard.csv) is loaded into a pandas DataFrame for further analysis.

## Step 2: Data Preprocessing
### Feature Scaling
To ensure consistent scales among features, feature scaling is performed using the StandardScaler from scikit-learn. The 'Amount' column is standardized, and the 'Time' column is dropped from the dataset, as it is not deemed essential for the analysis.

### Splitting into Training and Testing
The dataset is split into training and testing sets using the train_test_split function from scikit-learn. This division facilitates training and evaluating machine learning models.

## Step 3: Deep Learning Model
A deep learning model is constructed using the Keras library. The architecture comprises multiple layers with activation functions like 'relu' and 'sigmoid.' The model is compiled using the Adam optimizer and binary crossentropy loss function. Subsequently, it is trained on the training data.
