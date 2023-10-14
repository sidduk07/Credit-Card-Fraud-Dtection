# Credit Card Fraud Detection Project

## Overview
This project focuses on building a machine learning model to detect fraudulent credit card transactions. The dataset used for this project is sourced from Kaggle and contains anonymized credit card transactions labeled as fraudulent or non-fraudulent.

## Dataset
The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains the following features:
- `Time`: The number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1` to `V28`: Anonymized features resulting from a PCA transformation. Due to privacy reasons, the original features and more background information about the data cannot be provided.
- `Amount`: The transaction amount.
- `Class`: 1 for fraudulent transactions, 0 otherwise.

## Project Structure
- **Notebook:** The main analysis, data preprocessing, model training, and evaluation are conducted in the Jupyter Notebook titled `Credit_Card_Fraud_Detection.ipynb`.
- **Dataset:** The dataset (`creditcard.csv`) is available on the Kaggle link provided.
- **Requirements:** Required Python packages and versions are listed in the `requirements.txt` file.
- **Model:** The trained machine learning model (e.g., Random Forest, Logistic Regression) is serialized using pickle and stored as `fraud_detection_model.pkl`.

## Technical Details
### Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Data Preprocessing
- Checked for missing values (there were none).
- Explored the data distribution.
- Handled class imbalance, if necessary, using techniques like oversampling or undersampling.

### Model Building
- Split the data into training and testing sets.
- Explored different machine learning algorithms (e.g., Random Forest, Logistic Regression).
- Chose the best-performing algorithm based on evaluation metrics.

### Model Evaluation
- Used metrics like accuracy, precision, recall, F1-score, and confusion matrix to evaluate the model's performance.
- Tuned hyperparameters to optimize the model.

## Conclusion
The machine learning model developed in this project demonstrates a robust ability to detect fraudulent credit card transactions. The model's performance was evaluated using various metrics, ensuring its reliability in real-world applications. This project serves as a foundation for further enhancements and deployment into a production environment.
