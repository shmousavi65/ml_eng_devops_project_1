'''
This module implements the functions required for 
data processing, training, analysis of a model for credit card 
chrun prediction

Author: Hossein Mousavi
Date: 2023-08-25
'''

# import libraries
import logging
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)
    return df


def perform_eda(df, save_path):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    fig = plt.figure(figsize=(20,10)) 
    df['Churn'].hist()
    plt.savefig(os.path.join(save_path, 'churn_histogram.png'))
    plt.close(fig)


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == "__main__":
    # parameters
    data_path = 'data/bank_data.csv'

    logging.basicConfig(filename="./logs/main.txt",
                    level=logging.INFO,
                    filemode='w',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Started")

    logging.info("reading the data from {} ...".format(data_path))
    df = import_data("./data/bank_data.csv")
    logging.info("head of the dataframe:\n {}".format(df.head()))
    logging.info("shape of dataframe: {}".format(df.shape))

    perform_eda(df, 'images')