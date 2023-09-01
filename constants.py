'''
This module contains the parameters used in the pipeline

Author: Hossein Mousavi
Date: 2023-08-25
'''
# random state
RANDOM_STATE = 42
# data path
DATA_PATH = 'data/bank_data.csv'
# eda/results performance dir
IMAGES_DIR = 'images'
# trained models dir
MODEL_DIR = 'model'
# Categorical variables to be converted to numerical features
# using mean of response
CATEGORY_LST = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']
# features used for training
TRAIN_FEATURES = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                'Income_Category_Churn', 'Card_Category_Churn']
# name of target column
OUTPUT_COLUMN = 'Churn'
# proportion of test set
TEST_SIZE = 0.3
## params grid
LOG_REG_PARAM_GRID = {
    'max_iter': [3000]
}

RFC_PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}
##