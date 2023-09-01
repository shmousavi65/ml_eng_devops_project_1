'''
This module implements the functions required for
data processing, training, analysis of a model for credit card
chrun prediction

Author: Hossein Mousavi
Date: 2023-08-25
'''

# import libraries
import os
import logging
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import shap
import constants as cnts

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


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


class ChurnDfBuilder:
    '''
    class for building a churn df object
    '''

    def __init__(self, pth):
        '''
        class constructor

        input:
                pth: pth to the data csv
        output:
                None
        '''
        self._dataframe = import_data(pth)

    def add_churn_column(self):
        '''
        add column 'Churn' to the _dataframe
        '''
        self._dataframe['Churn'] = self._dataframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

    def perform_eda(self, save_path='images'):
        '''
        perform eda on _datafrrame and save figures to save_path

        input:
                save_path: where to save the eda images. Defaults to 'image' directory
        '''
        fig = plt.figure(figsize=(20, 10))
        self._dataframe['Churn'].hist()
        plt.savefig(os.path.join(save_path, 'churn.png'))
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))
        self._dataframe['Customer_Age'].hist()
        plt.savefig(os.path.join(save_path, 'customer_age.png'))
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))
        self._dataframe.Marital_Status.value_counts(
            'normalize').plot(kind='bar')
        plt.savefig(os.path.join(save_path, 'marital_status.png'))
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))
        # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
        # using a kernel density estimate
        sns.histplot(
            self._dataframe['Total_Trans_Ct'],
            stat='density',
            kde=True)
        plt.savefig(os.path.join(save_path, 'total_trans_ct.png'))
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))
        sns.heatmap(
            self._dataframe.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.savefig(os.path.join(save_path, 'correlation.png'))
        plt.close(fig)

    def encoder_helper(self, category_lst, response='Churn'):
        '''
        helper function to turn each categorical column in the category_lst into a new column with
        propotion of response for each category

        input:
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming
                variables or index y column]. Defaults to 'Churn'.
        '''
        for feature in category_lst:
            feature_lst = []
            feature_groups = self._dataframe.groupby(feature).mean()[response]
            for val in self._dataframe[feature]:
                feature_lst.append(feature_groups.loc[val])
            self._dataframe[feature + '_' + response] = feature_lst

    def get_df(self):
        '''
        return the built dataframe
        '''
        return self._dataframe


def perform_feature_engineering(
        df, feature_cols, response, test_size=0.3, random_state=42):
    '''
    generate train/test split based on the input params

    input:
            df: the input dataframe
            feature_cols: list of columns used as training features
            response: string of output column name. Defaults to 'Churn'.
            test_size: proportion size of test_set. Defaults to 0.3.
            random_state : seed for the random generator. Defaults to 42.
    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    X = pd.DataFrame()
    X[feature_cols] = df[feature_cols]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                model_name='model',
                                save_dir='images'):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions
            y_test_preds: test predictions
            model_name: name of the model to be used as the name of the report to be saved. Defaults
             to 'model'
            save_dir: dir to save the classfication report image. Defaults to 'images' directory
    output:
             None
    '''
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.text(0.01, 1.25, str(model_name + ' Train '), {
        'fontsize': 10}, fontproperties='monospace')
    ax.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    ax.text(0.01, 0.6, str(model_name + ' Test '), {
        'fontsize': 10}, fontproperties='monospace')
    ax.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    ax.axis('off')
    fig.savefig(
        os.path.join(
            save_dir,
            model_name +
            "_classification_reports.png"))
    plt.close(fig)


def tree__based_feature_importance_plot(model, X_data, model_name, save_dir):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            model_name: name of the model to be used as the name of the report to be saved. Defaults
             to 'model'
            save_dir: dir to store the importance plot figure
    output:
             None
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    fig = plt.figure(figsize=(20, 10))
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    plt.savefig(
        os.path.join(
            save_dir,
            model_name +
            '_shap_feature_importance.png'))
    plt.close(fig)


class GridTrainer:
    '''
    class for training a model based on grid search
    '''

    def __init__(self, model, param_grid, model_name):
        self.model = model
        self.param_grid = param_grid
        self.model_name = model_name

    def train(self, X_train, X_test, y_train, y_test,
              scores_save_dir='images', models_save_dir='model', roc_ax=None, num_cv=5):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
                num_cv: number of folds used for cross validation in grid search. Defaults to 5.
                scores_save_dir: directory to save the modeling performance and scores. Defaults to
                    'images'.
                models_save_dir: directory to save the best trained model.
                 Defaults to 'model'.
                 roc_ax: figure for plotting roc curve. If None new figure is generated. Defaults
                 to None.
        output:
                None
        '''

        model_grid = GridSearchCV(estimator=self.model,
                                  param_grid=self.param_grid, cv=num_cv)
        model_grid.fit(X_train, y_train)
        self.best_model = model_grid.best_estimator_
        # save best model
        joblib.dump(
            self.best_model,
            os.path.join(models_save_dir, self.model_name + '_model.pkl')
        )

        y_train_preds = self.best_model.predict(X_train)
        y_test_preds = self.best_model.predict(X_test)
        classification_report_image(y_train,
                                    y_test,
                                    y_train_preds,
                                    y_test_preds,
                                    self.model_name,
                                    scores_save_dir)
        if roc_ax is None:
            fig, ax = plt.subplots(figsize=(15, 8))
        else:
            ax = roc_ax
        plot_roc_curve(self.best_model, X_test, y_test, ax=ax, alpha=0.8)
        fig = ax.get_figure()
        fig.savefig(os.path.join(scores_save_dir, 'roc_curves.png'))

    def best_model_feature_importance_plot(self, X, save_dir='images'):
        '''
        find and save the shap feature importance
        '''
        tree__based_feature_importance_plot(
            self.best_model, X, self.model_name, save_dir)


class GridTrainerFactory:
    """
    generates different model for grid search based training
    """
    @staticmethod
    def logreg(param_grid):
        '''
        return a trainer for logistic regression
        '''
        model = LogisticRegression()
        return GridTrainer(model, param_grid, 'logistic_reg')

    @staticmethod
    def randfor(param_grid, random_state):
        '''
        return a trainer for random forest
        '''
        model = RandomForestClassifier(random_state=random_state)
        return GridTrainer(model, param_grid, 'random_forest')


if __name__ == "__main__":

    logging.basicConfig(filename="./logs/main.txt",
                        level=logging.INFO,
                        filemode='w',
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Pipeline Started")

    # Data Preprocessing
    logging.info("reading the data from %s ...", cnts.DATA_PATH)
    churn_df_builder = ChurnDfBuilder(cnts.DATA_PATH)

    logging.info(
        "head of the dataframe:\n %s",
        churn_df_builder.get_df().head())
    logging.info("shape of dataframe: %s", churn_df_builder.get_df().shape)

    logging.info("'Churn' column is added to the dataframe\n")
    churn_df_builder.add_churn_column()

    logging.info('performing eda ...')
    churn_df_builder.perform_eda(cnts.IMAGES_DIR)
    logging.info(
        'results images are saved to the directory %s \n', cnts.IMAGES_DIR)

    logging.info("The following categorical features are being encoded ....:\n %s\n",
                 cnts.CATEGORY_LST)
    churn_df_builder.encoder_helper(cnts.CATEGORY_LST, cnts.OUTPUT_COLUMN)
    logging.info("The column names after adding the encoded columns are as follows:\n %s\n",
                 churn_df_builder.get_df().columns)

    churn_df = churn_df_builder.get_df()
    ##

    # Generate train/test data
    logging.info("Train/Test data are generated. \n  input features: %s\n  output feature: %s\n ",
                 cnts.TRAIN_FEATURES, cnts.OUTPUT_COLUMN)
    X_train_df, X_test_df, y_train_df, y_test_df = perform_feature_engineering(churn_df,
                                                                               cnts.TRAIN_FEATURES,
                                                                               cnts.OUTPUT_COLUMN,
                                                                               cnts.TEST_SIZE,
                                                                               cnts.RANDOM_STATE)
    ##

    # Train logistic regression
    figure, axis = plt.subplots(figsize=(15, 8))
    logging.info("Training the logistic regression with the grid search \n %s \n",
                 cnts.LOG_REG_PARAM_GRID)
    GridTrainerFactory.logreg(cnts.LOG_REG_PARAM_GRID).train(X_train_df, X_test_df,
                                                             y_train_df, y_test_df,
                                                             cnts.IMAGES_DIR, cnts.MODEL_DIR,
                                                             roc_ax=axis
                                                             )
    logging.info("Resutls are saved to %s \n. best models are saved to %s \n", cnts.IMAGES_DIR,
                 cnts.MODEL_DIR)
    ##

    # Train random forest
    logging.info("Training the radom forest with the grid search \n %s \n",
                 cnts.RFC_PARAM_GRID)
    random_forest_trainer = GridTrainerFactory.randfor(
        cnts.RFC_PARAM_GRID, cnts.RANDOM_STATE)
    random_forest_trainer.train(X_train_df, X_test_df, y_train_df, y_test_df,
                                cnts.IMAGES_DIR, cnts.MODEL_DIR, roc_ax=axis)
    random_forest_trainer.best_model_feature_importance_plot(
        X_train_df, cnts.IMAGES_DIR)
    logging.info("Resutls are saved to %s \n. best models are saved to %s \n", cnts.IMAGES_DIR,
                 cnts.MODEL_DIR)
    plt.close(figure)
    ##

    logging.info("Pipeline Finished")
