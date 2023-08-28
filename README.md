# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project trains a random-forest and logistic-regression model to predict the customer churn of a bank, for which data is provided. 

## Files and data description
- churn_library : contains all the functions used in the pipeline
- test_churn_library : contains the unittest for the churn_library functions.
- constants : contians the input parameters and their values used in the pipeline.
- data/bank_data.csv : the data used for training and testing.
- images/ : contains the model training and performance evaluation results
- model : contains the trained 
    - logistic-regression model
    - best random-forest model obtained from grid search
- log/ : contains the logging file from the runs.

## Environment setup
create a new environment in the project dir and activate it:
```
cd ml_eng_devops_project_1

python3.8 -m venv env

source env/bin/activate
```   
install the required package:
```
pip install -r requirements_py3.8.txt
```

**Note**: You can also use python 3.6 instead of python 3.8 in the above instruction.

## Running Files
To run the unit tests:
```
python test_churn_library 
```
The related logging script is stored in log/test_churn_library.log.


To run the piepline:
```
python churn_library 
```
The related logging script is stored in log/main.log.





