'''
This module implements the tests for the functions of churn_library

Author: Hossein Mousavi
Date: 2023-08-25
'''
import os
import logging
import pytest
import pandas as pd
import churn_library as cls
import constants as cnts

logging.basicConfig(
    filename='./logs/test_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def declare_pytest_scope_varaibles():
    '''declare the variables defined in the pytest scope'''
    pytest.df = pd.DataFrame()

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        pytest.df = cls.import_data(cnts.data_path)
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert pytest.df.shape[0] > 0
        assert pytest.df.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_add_churn_column():
    '''
    test add_churn_column
    '''
    try:
        pytest.df = cls.add_churn_column(pytest.df)
        assert 'Churn' in pytest.df.columns
        logging.info("Testing add_churn_column: SUCCESS")
    except AssertionError as err:
        logging.error("Tesing add_churn_column: the Churn column is not added to the dataframe")
        raise err


def test_eda():
    '''
    test perform_eda
    '''
    try:
        cls.perform_eda(pytest.df)
        assert os.path.isfile(os.path.join(cnts.images_dir, 'churn.png'))
        assert os.path.isfile(os.path.join(cnts.images_dir, 'customer_age.png'))
        assert os.path.isfile(os.path.join(cnts.images_dir, 'marital_status.png'))
        assert os.path.isfile(os.path.join(cnts.images_dir, 'total_trans_ct.png'))
        assert os.path.isfile(os.path.join(cnts.images_dir, 'correlation.png'))
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The fig does not exist in the directory %s",
            cnts.images_dir)
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    try:
        pytest.df = cls.encoder_helper(pytest.df, cnts.category_lst)
        for feature in cnts.category_lst:
            assert feature + '_' + cnts.output_column in pytest.df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing endoder_helper: The columns for the categorical features are not added \
                  properly to the dataframe")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:
        pytest.X_train, pytest.X_test, pytest.y_train, pytest.y_test = \
              cls.perform_feature_engineering(
            pytest.df, cnts.train_features, cnts.output_column)
        assert pytest.X_train.shape[0] >= cnts.test_size * pytest.X_test.shape[0]
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The train/test data are not generated properly")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        cls.train_models(
            pytest.X_train,
            pytest.X_test,
            pytest.y_train,
            pytest.y_test,
            cnts.rfc_param_grid)
        assert os.path.isfile(
            os.path.join(
                cnts.images_dir,
                'classification_reports.png'))
        assert os.path.isfile(os.path.join(cnts.images_dir, 'roc_curves.png'))
        assert os.path.isfile(os.path.join(cnts.model_dir, 'rfc_model.pkl'))
        assert os.path.isfile(os.path.join(cnts.model_dir, 'logistic_model.pkl'))
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: The classification report do not exist in the directory %s",
            'images')
        raise err


if __name__ == "__main__":
    declare_pytest_scope_varaibles()
    pytest.main()
