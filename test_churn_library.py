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
import matplotlib.pyplot as plt

logging.basicConfig(
    filename='./logs/test_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def declare_pytest_scope_varaibles():
    '''declare the variables defined in the pytest scope'''
    # pytest.churn_df_builder = cls.ChurnDfBuilder()
    pass


def test_import_data():
    '''
    test import data - this example is completed for you to assist with the other test functions
    '''
    try:
        pytest.df = cls.import_data(cnts.DATA_PATH)
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


class TestChurnDfBuilder:
    '''
    test ChuenBuilder class methods
    '''

    def test_constructor(self):
        '''
        test the class constructor
        '''
        try:
            pytest.churn_df_builder = cls.ChurnDfBuilder(cnts.DATA_PATH)
            logging.info("Testing ChurnDfBuilder constructor: SUCCESS")
        except BaseException as err:
            logging.error("testing ChuenBuilder constructor!")
            raise err

    def test_add_churn_column(self):
        '''
        test add_churn_column method
        '''
        try:
            pytest.churn_df_builder.add_churn_column()
            assert 'Churn' in pytest.churn_df_builder.get_df().columns
            logging.info("Testing add_churn_column: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Tesing add_churn_column: the Churn column is not added to the dataframe")
            raise err

    def test_perform_eda(self):
        '''
        test perform_eda method
        '''
        try:
            pytest.churn_df_builder.perform_eda(cnts.IMAGES_DIR)
            assert os.path.isfile(os.path.join(cnts.IMAGES_DIR, 'churn.png'))
            assert os.path.isfile(
                os.path.join(
                    cnts.IMAGES_DIR,
                    'customer_age.png'))
            assert os.path.isfile(
                os.path.join(
                    cnts.IMAGES_DIR,
                    'marital_status.png'))
            assert os.path.isfile(
                os.path.join(
                    cnts.IMAGES_DIR,
                    'total_trans_ct.png'))
            assert os.path.isfile(
                os.path.join(
                    cnts.IMAGES_DIR,
                    'correlation.png'))
            logging.info("Testing perform_eda: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing perform_eda: The fig does not exist in the directory %s",
                cnts.IMAGES_DIR)
            raise err

    def test_encoder_helper(self):
        '''
        test encoder_helper method
        '''
        try:
            pytest.churn_df_builder.encoder_helper(cnts.CATEGORY_LST)
            for feature in cnts.CATEGORY_LST:
                assert feature + '_' + cnts.OUTPUT_COLUMN in pytest.churn_df_builder.get_df()\
                    .columns
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
        pytest.churn_df = pytest.churn_df_builder.get_df()
        pytest.X_train, pytest.X_test, pytest.y_train, pytest.y_test = \
            cls.perform_feature_engineering(
                pytest.churn_df, cnts.TRAIN_FEATURES, cnts.OUTPUT_COLUMN)
        assert pytest.X_train.shape[0] >= cnts.TEST_SIZE * \
            pytest.X_test.shape[0]
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The train/test data are not generated properly")
        raise err


class TestGridTrainer:
    '''
    test the GridTrainer class methods
    '''

    def test_constructor(self):
        '''
        test GridTrainer constructor
        '''
        try:
            pytest.log_reg_trainer = cls.GridTrainerFactory.logreg(
                cnts.LOG_REG_PARAM_GRID)
            pytest.random_forest_trainer = cls.GridTrainerFactory\
                .randfor(cnts.RFC_PARAM_GRID, cnts.RANDOM_STATE)
            logging.info("Testing GridTrainer constructor: SUCCESS")
        except BaseException as err:
            logging.error("Testing GridTrainer constructor: Failed!")
            raise err

    def test_train_log_reg(self):
        '''
        test train method for logistic regression
        '''
        try:
            _, pytest.ax = plt.subplots(figsize=(15, 8))
            pytest.log_reg_trainer.train(pytest.X_train, pytest.X_test,
                                         pytest.y_train, pytest.y_test,
                                         cnts.IMAGES_DIR, cnts.MODEL_DIR, roc_ax=pytest.ax)
            assert os.path.isfile(
                os.path.join(
                    cnts.IMAGES_DIR,
                    'logistic_reg_roc_curves.png'))
            assert os.path.isfile(
                os.path.join(
                    cnts.IMAGES_DIR,
                    'roc_curves.png'))
            assert os.path.isfile(os.path.join(cnts.IMAGES_DIR,
                                               'logistic_reg_classification_reports.png'))
            assert os.path.isfile(
                os.path.join(
                    cnts.MODEL_DIR,
                    'logistic_reg_model.pkl'))
            logging.info("Testing logistic regression trainer.train: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing logistic regression trainer.train: Failed!")
            raise err

    def test_train_rand_forest(self):
        '''
        test train method for random forest
        '''
        try:
            pytest.random_forest_trainer.train(pytest.X_train, pytest.X_test,
                                               pytest.y_train, pytest.y_test,
                                               cnts.IMAGES_DIR, cnts.MODEL_DIR, roc_ax=pytest.ax)
            assert os.path.isfile(
                os.path.join(
                    cnts.IMAGES_DIR,
                    'random_forest_roc_curves.png'))
            assert os.path.isfile(
                os.path.join(
                    cnts.IMAGES_DIR,
                    'roc_curves.png'))
            assert os.path.isfile(os.path.join(cnts.IMAGES_DIR,
                                               'random_forest_classification_reports.png'))
            assert os.path.isfile(
                os.path.join(
                    cnts.MODEL_DIR,
                    'random_forest_model.pkl'))
            logging.info("Testing random_forest trainer.train: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing random_forest trainer.train: Failed!")
            raise err


if __name__ == "__main__":
    declare_pytest_scope_varaibles()
    pytest.main()
