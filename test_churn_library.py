import pytest
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/test_churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def initialize_pytest_params():
	pytest.figs_dir = 'images'


def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		pytest.df = cls.import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert pytest.df.shape[0] > 0
		assert pytest.df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	try:
		cls.perform_eda(pytest.df, pytest.figs_dir)
		assert os.path.isfile(os.path.join(pytest.figs_dir, 'churn_histogram.png'))
		logging.info("Testing perform_eda: SUCCESS")
	except AssertionError as err:
		logging.error("Testing perform_eda: The fig does not exist in the directory %s", pytest.figs_dir)
		raise err



# def test_encoder_helper(encoder_helper):
# 	'''
# 	test encoder helper
# 	'''


# def test_perform_feature_engineering(perform_feature_engineering):
# 	'''
# 	test perform_feature_engineering
# 	'''


# def test_train_models(train_models):
# 	'''
# 	test train_models
# 	'''


if __name__ == "__main__":
	initialize_pytest_params()
	pytest.main()








