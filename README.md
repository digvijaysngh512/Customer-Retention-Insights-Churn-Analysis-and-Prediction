Customer Churn Analysis – EDA and Modeling
Overview

This project focuses on Exploratory Data Analysis (EDA) and predictive modeling to understand and forecast customer churn.

Customer churn (or attrition) occurs when customers discontinue using a company’s products or services. It is a crucial business metric as it directly affects revenue and profitability. High churn rates often signal dissatisfaction, poor experiences, or lack of engagement.

## Dataset

The dataset used in this project is [Data Source](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction/data).

It contains the following features: 

 1. RowNumber
 2. CustomerId
 3. Surname
 4. CreditScore
 5. Geography
 6. Gender
 7. Age
 8. Tenure
 9. Balance
 10. NumOfProducts
 11. HasCrCard
 12. IsActiveMember
 13. EstimatedSalary
 14. Exited

The main variables of interest is **Exited**.

## Requirements

The following libraries are required to run the notebook:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Key Components

Handling Class Imbalance: Applied oversampling techniques like SMOTE and class weighting to improve predictions on the minority churn class.

Exploratory Data Analysis (EDA): Performed detailed data exploration to identify key trends, patterns, and factors influencing churn.

Classification Models: Built and compared multiple models including Logistic Regression, Random Forest, KNN, SVM, XGBoost, and Gradient Boosting, with imbalance handling techniques applied.

## Results

| Model                   | Accuracy | Recall Score | F1 Score | ROC AUC Score |
|-------------------------|----------|--------------|----------|---------------|
| Logistic Regression     | 0.703667 | 0.683219     | 0.473029 | 0.764076      |
| Random Forest           | 0.862000 | 0.414384     | 0.538976 | 0.852447      |
| K-Nearest Neighbors     | 0.752333 | 0.667808     | 0.512147 | 0.776639      |
| Support Vector Machine  | 0.785667 | 0.662671     | 0.546224 | 0.822503      |
| XGBoost                 | 0.833000 | 0.609589     | 0.586974 | 0.841784      |
| Gradient Boosting       | 0.817000 | 0.700342     | 0.598391 | 0.859767      |

From the results of the classification models on the churn prediction dataset, we can infer the following:

1. **Gradient Boosting** has the highest F1 score (0.598391) and the highest ROC AUC score (0.859767) among all the models. This suggests that Gradient Boosting is the most effective model in balancing precision and recall and has the best ability to distinguish between the churned and non-churned customers.

2. **XGBoost** also performs well, with a relatively high F1 score (0.586974) and a good ROC AUC score (0.841784). This indicates that XGBoost is another strong model for this task.

3. **Random Forest** has a high accuracy (0.862000) but a lower F1 score (0.538976) compared to Gradient Boosting and XGBoost. This suggests that while Random Forest is good at predicting the majority class (non-churned customers), it might not be as effective at identifying the minority class (churned customers).

4. **Support Vector Machine** and **K-Nearest Neighbors** have moderate F1 scores and ROC AUC scores. They perform better than Logistic Regression but are not as effective as Gradient Boosting or XGBoost for this dataset.

5. **Logistic Regression** has the lowest accuracy (0.703667), F1 score (0.473029), and ROC AUC score (0.764076) among all the models. This indicates that Logistic Regression is the least effective model for predicting customer churn in this dataset.

### Conclusion
Streamlined features to the most important predictors, improving model efficiency and interpretability.
Achieved 83.3% accuracy and 58.7% F1-score with XGB on churn prediction, balancing precision and recall.
Optimized XGB, reaching 81.7% accuracy, 70.0% recall, and 86.0% ROC AUC, making it effective model.

#### Overall:

Gradient Boosting appears to be the best model for this churn prediction task, followed closely by XGBoost. These models are able to better handle the class imbalance and provide a good balance between precision and recall. 




