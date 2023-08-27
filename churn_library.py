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


def add_churn_column(df):
    '''
    add column 'Churn' to the dataframe

    input:
            df: pandas dataframe
    output:
            df: pandas dataframe
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df, save_path='images'):
    '''
    perform eda on df and save figures to images folder
    
    input:
            df: pandas dataframe
            save_path: where to save the eda images. Defaults to 'image' directory
    output:
            None
    '''
    fig = plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(os.path.join(save_path, 'churn.png'))
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(os.path.join(save_path, 'customer_age.png'))
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(save_path, 'marital_status.png'))
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(save_path, 'total_trans_ct.png'))
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(save_path, 'correlation.png'))
    plt.close(fig)


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming
              variables or index y column]. Defaults to 'Churn'.
    output:
            df: pandas dataframe with new columns for
    '''
    for feature in category_lst:
        feature_lst = []
        feature_groups = df.groupby(feature).mean()[response]
        for val in df[feature]:
            feature_lst.append(feature_groups.loc[val])
        df[feature + '_' + response] = feature_lst
    return df


def perform_feature_engineering(df, feature_cols, response, test_size=0.3, random_state=42):
    '''
    input:
              df: pandas dataframe
              feature_cols: list of columns used as training features
              response: string of response name [optional argument that could be used for naming
                variables or index y column]
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
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                save_path='images'):
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
            save_path: where to save the eda images. Defaults to 'images' directory
    output:
             None
    '''
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
    # ax[0].rc('figure', figsize=(5, 10))
    ax[0].text(0.01, 1.25, str('Random Forest Train'), {
               'fontsize': 10}, fontproperties='monospace')
    ax[0].text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
               'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    ax[0].text(0.01, 0.6, str('Random Forest Test'), {
               'fontsize': 10}, fontproperties='monospace')
    ax[0].text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
               'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    ax[0].axis('off')

    ax[1].text(0.01, 1.25, str('Logistic Regression Train'),
               {'fontsize': 10}, fontproperties='monospace')
    ax[1].text(0.01, 0.05, str(classification_report(y_test, y_test_preds_lr)), {
               'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    ax[1].text(0.01, 0.6, str('Logistic Regression Test'), {
               'fontsize': 10}, fontproperties='monospace')
    ax[1].text(0.01, 0.7, str(classification_report(y_train, y_train_preds_lr)), {
               'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    ax[1].axis('off')

    fig.savefig(os.path.join(save_path, "classification_reports.png"))
    plt.close(fig)


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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    fig = plt.figure(figsize=(20, 10))
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    plt.savefig(output_pth)
    plt.close(fig)


def train_models(X_train, X_test, y_train, y_test, rfc_param_grid, log_reg_max_iter=3000,
                  random_state=42, scores_save_path='images', models_save_path='model'):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              log_reg_max_iter: maximum number of iterations in logistic regression model training. 
               Defaults to 3000.
              rfc_param_grid: dictionay used for grid search in random forest model training
              random_state : seed for the random generator (optional, defaults to 42)
              scores_save_path: directory to save the modeling performance and scores. Defaults to
                'images'.
              models_save_path: directory to save the trained models. Defaults to 'model'.
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=random_state)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=log_reg_max_iter)

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=rfc_param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                save_path=scores_save_path)

    fig, ax = plt.subplots(figsize=(15, 8))
    plot_roc_curve(lrc, X_test, y_test, ax=ax)
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    fig.savefig(os.path.join(scores_save_path, 'roc_curves.png'))
    plt.close(fig)

    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        os.path.join(
            scores_save_path,
            'rfc_shap_feature_importance.png'))

    # save best model
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            models_save_path,
            'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(models_save_path, 'logistic_model.pkl'))


if __name__ == "__main__":

    logging.basicConfig(filename="./logs/main.txt",
                    level=logging.INFO,
                    filemode='w',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Started")

    logging.info("reading the data from {} ...".format(cnts.data_path))
    df = import_data(cnts.data_path)
    logging.info("head of the dataframe:\n {}".format(df.head()))
    logging.info("shape of dataframe: {}".format(df.shape))
    
    logging.info("'Churn' column is added to the dataframe\n")
    df = add_churn_column(df)

    logging.info('performing eda ...')
    perform_eda(df, 'images')
    logging.info('results images are saved to the directory {} \n'.format(cnts.images_dir))

    logging.info("The following categorical features are being encoded ....:\n {}\n".format(
        cnts.category_lst))
    df = encoder_helper(df, cnts.category_lst, cnts.output_column)
    logging.info("The column names after adding the encoded columns are as follows:\n {}\n".format(
        df.columns))

    logging.info("Train/Test data are generated. \n  input features: {}\n  output feature: {}\n ".
                 format(cnts.train_features, cnts.output_column))
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, cnts.train_features,
                                                                    cnts.output_column,
                                                                    cnts.test_size,
                                                                    cnts.random_state)
    
    logging.info("Training the logistic regression with the log_reg_max_iter {} \n \
                  and random forest model with the grid search \n {} \n".
                  format(cnts.log_reg_max_iter, cnts.rfc_param_grid))
    train_models( X_train, X_test, y_train, y_test, cnts.rfc_param_grid,
                cnts.log_reg_max_iter, cnts.random_state,
                cnts.images_dir, cnts.model_dir)
    logging.info("Resutls are saved to {} \n. best models are saved to {} \n".format(
        cnts.images_dir, cnts.model_dir))