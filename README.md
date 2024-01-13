# Credit Card Fraud Detection

This repository focuses on credit card fraud detection, employing various techniques, including deep learning models, random forest, decision tree, and sampling methods. The primary goal is to develop effective models for identifying fraudulent transactions in credit card datasets.

## Step 1: Importing Libraries and Data
### Libraries
The first step involves importing necessary libraries to facilitate data analysis and model development. These include:

- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- matplotlib: For creating visualizations.
- seaborn: For enhancing the visual appeal of plots.
- tensorflow: For building and training deep learning models.

### Data
The dataset (creditcard.csv) is loaded into a pandas DataFrame using pd.read_csv(). This dataset presumably contains anonymized information about credit card transactions.

## Step 2: Data Preprocessing
### Feature Scaling
To ensure consistent scales among features, feature scaling is performed using the StandardScaler from scikit-learn. The 'Amount' column is standardized, and the 'Time' column is dropped from the dataset, as it is not deemed essential for the analysis.

### Splitting into Training and Testing
The dataset is split into training and testing sets using the train_test_split function from scikit-learn. This division facilitates training and evaluating machine learning models.

## Step 3: Deep Learning Model
### Model Architecture
A deep learning model is constructed using the Keras library. The architecture consists of multiple layers, including dense layers with 'relu' activation functions and a final layer with a 'sigmoid' activation function.

### Compilation and Training
The model is compiled using the Adam optimizer and binary crossentropy loss function. It is then trained on the training data using the fit method. The training process is conducted for a specified number of epochs, and the model's performance is monitored.

## Step 4: Evaluating the Deep Learning Model
The trained deep learning model is evaluated on the testing set to assess its performance. Metrics such as loss and accuracy are calculated and printed.

## Step 5: Random Forest Model
A Random Forest classifier is employed to model the data. The classifier is trained on the training set and evaluated on the testing set. The accuracy score and a confusion matrix are generated for further analysis.
### Results
The accuracy score of the Random Forest model is calculated, providing an indication of its performance on the testing data. Additionally, a confusion matrix is generated to visualize the model's ability to correctly classify fraudulent and non-fraudulent transactions.

## Step 6: Decision Tree Model
Similarly, a Decision Tree classifier is utilized for modeling. The classifier is trained on the training set and evaluated on the testing set. The accuracy score and a confusion matrix are generated for analysis.
### Results
The accuracy score and confusion matrix help assess the Decision Tree model's effectiveness in identifying fraudulent transactions.

## Step 7: Sampling
Given the imbalanced nature of the dataset, two sampling techniques are explored:

### Undersampling and Training the Deep Learning Model on it
Undersampling involves reducing the number of instances in the over-represented class (non-fraudulent transactions). A balanced dataset is created by randomly selecting a subset of non-fraudulent transactions equal to the number of fraudulent transactions. The deep learning model is then trained on this balanced dataset.
#### Results
The model's performance on the undersampled testing set is evaluated using a confusion matrix, providing insights into its ability to detect both fraudulent and non-fraudulent transactions.

### Oversampling (SMOTE) and Training the Deep Learning Model on it
Oversampling involves increasing the number of instances in the under-represented class (fraudulent transactions). The Synthetic Minority Over-sampling Technique (SMOTE) is applied to generate synthetic instances of the minority class. The deep learning model is trained on this oversampled dataset.
#### Results
The model's performance on the oversampled testing set is assessed using a confusion matrix. This evaluation helps understand the model's performance when trained on a more balanced dataset.


## Conclusion
The combination of deep learning models, traditional machine learning algorithms (Random Forest and Decision Tree), and sampling techniques offers a comprehensive approach to credit card fraud detection. The results obtained from each method provide insights into their respective strengths and weaknesses. Additionally, the sampling methods demonstrate the importance of handling imbalanced datasets to enhance model performance in real-world scenarios. The choice of approach depends on the specific requirements and constraints of the credit card fraud detection problem at hand.
