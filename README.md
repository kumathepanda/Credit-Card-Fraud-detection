# Credit Card Fraud Detection

This project addresses the credit card fraud detection problem using machine learning techniques. The dataset is highly imbalanced, with a small percentage of fraudulent transactions.

## Dataset

The dataset contains credit card transactions over two days, with the target variable `Class` indicating whether a transaction is fraudulent (1) or not (0).  Due to confidentiality, the features are the result of a PCA transformation, except for `Time` and `Amount`.

-   `Time`: Seconds elapsed between each transaction and the first transaction.
-   `Amount`: Transaction amount.
-   `V1-V28`: Principal components obtained from PCA.
-   `Class`: Target variable (0: non-fraudulent, 1: fraudulent).

**Note:** The dataset is highly imbalanced.

## Libraries Used

-   pandas
-   numpy
-   sklearn (scikit-learn)
-   matplotlib
-   seaborn
-   imblearn (for handling imbalanced data)

## Methodology

1.  **Data Loading and Exploration:**
    -   Loaded the dataset using pandas.
    -   Checked for missing values (`data.info()`).
    -   Examined the data distribution and basic statistics (`data.describe()`).

2.  **Data Preprocessing:**
    -   Scaled the `Amount` and `Time` features using `StandardScaler`.
    -   Split the data into training and testing sets.

3.  **Handling Imbalanced Data:**
    -   **Oversampling (SMOTE):** Applied SMOTE to oversample the minority class (fraudulent transactions) in the training set.
    -   **Undersampling (RandomUnderSampler):** Applied RandomUnderSampler to undersample the majority class (non-fraudulent transactions) in the training set.
    -   **Combined Sampling:** Experimented with combining oversampling and undersampling techniques.

4.  **Model Training and Evaluation:**
    -   **Logistic Regression:**
        -   Trained a Logistic Regression model on the original, oversampled, undersampled, and combined sampled data.
        -   Evaluated performance using classification reports and ROC curves.
        -   Performed Grid Search for hyperparameter tuning.
    -   **Random Forest Classifier:**
        -   Trained a Random Forest Classifier on the original, oversampled, undersampled, and combined sampled data.
        -   Evaluated performance using classification reports and ROC curves.
        -   Performed Grid Search for hyperparameter tuning.

5.  **Threshold Tuning:**
    -   Visualized the ROC curve and AUC for the best performing Random Forest model with oversampling.
    -   (Further threshold tuning could be explored to optimize for specific precision/recall trade-offs)

## Results

-   The notebook provides a comprehensive analysis of the credit card fraud dataset.
-   It demonstrates the impact of class imbalance on model performance.
-   It showcases the effectiveness of oversampling and undersampling techniques in addressing the class imbalance problem.
-   Both Logistic Regression and Random Forest models were trained and evaluated, with Random Forest generally showing better performance, especially after oversampling.
-   The notebook includes visualizations (ROC curves) to aid in understanding model performance.

## Conclusion

This project provides a solid foundation for credit card fraud detection. Further improvements could include:

-   More extensive hyperparameter tuning.
-   Trying other advanced models (e.g., XGBoost, LightGBM).
-   Feature engineering.
