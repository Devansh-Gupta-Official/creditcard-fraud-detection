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

## Step 4: Evaluating the Deep Learning Model
The trained deep learning model is evaluated on the testing set to assess its performance. Metrics such as loss and accuracy are calculated and printed.

### Step 5: Random Forest Model
A Random Forest classifier is employed to model the data. The classifier is trained on the training set and evaluated on the testing set. The accuracy score and a confusion matrix are generated for further analysis.

### Step 6: Decision Tree Model
Similarly, a Decision Tree classifier is utilized for modeling. The classifier is trained on the training set and evaluated on the testing set. The accuracy score and a confusion matrix are generated for analysis.

### Step 7: Sampling
Given the imbalanced nature of the dataset, two sampling techniques are explored:

### Undersampling and Training the Deep Learning Model on it
Undersampling involves reducing the number of instances in the over-represented class (non-fraudulent transactions). A balanced dataset is created by randomly selecting a subset of non-fraudulent transactions equal to the number of fraudulent transactions. The deep learning model is then trained on this balanced dataset.

### Oversampling (SMOTE) and Training the Deep Learning Model on it
Oversampling involves increasing the number of instances in the under-represented class (fraudulent transactions). The Synthetic Minority Over-sampling Technique (SMOTE) is applied to generate synthetic instances of the minority class. The deep learning model is trained on this oversampled dataset.

