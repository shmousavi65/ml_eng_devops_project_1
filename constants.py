'''
This module contains the parameters used in the pipeline

Author: Hossein Mousavi
Date: 2023-08-25
'''
# random state
random_state = 42
# data path
data_path = 'data/bank_data.csv'
# eda/results performance dir
images_dir = 'images'
# trained models dir
model_dir = 'model'
# Categorical variables to be converted to numerical features
# using mean of response
category_lst = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']
# features used for training
train_features = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                'Income_Category_Churn', 'Card_Category_Churn']
# name of target column
output_column = 'Churn'
# proportion of test set
test_size = 0.3
#max iteration in logistic regression
log_reg_max_iter = 3000
# random forest parameters' grid
# param_grid = {
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt'],
#     'max_depth': [4, 5, 100],
#     'criterion': ['gini', 'entropy']
# }
rfc_param_grid = {
    'n_estimators': [200],
    'max_features': ['auto'],
    'max_depth': [4],
    'criterion': ['gini']
    }