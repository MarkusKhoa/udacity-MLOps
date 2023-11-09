import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, roc_curve, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


EDA_PATH = "./images/eda"
RESULTS_PATH = "./images/results"
MODEL_PATH = "./models"
CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

def import_data(data_path):
    """
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv

    output:
        data: pandas dataframe

    raises:
        FileNotFoundError: if the file is not found
    """
    try:
        data = pd.read_csv(data_path)
        logging.info("SUCCESS: Import_data")
        return data
    except FileNotFoundError as err:
        logging.error("File not found")
        raise err


def perform_eda(data):
    """
    perform eda on data and save figures to images folder
    input:
        data: pandas dataframe

    output:
        None
    """
    logging.info(data.shape)
    logging.info(data.isnull().sum())
    logging.info(data.describe())

    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    data['Churn'].hist()
    plt.savefig(os.path.join(EDA_PATH, "churn_histogram.png"))

    plt.figure(figsize=(20, 10))
    data['Customer_Age'].hist()
    plt.savefig(os.path.join(EDA_PATH, "customer_age_histogram.png"))

    plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(EDA_PATH, "marital_status_counts.png"))

    plt.figure(figsize=(20, 10))
    sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(EDA_PATH, "total_transaction_histogram.png"))

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        data.corr(
            numeric_only=True),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(os.path.join(EDA_PATH, "heatmap.png"))


def encoder_helper(data, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        data: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for naming \
            variables or index y column]

    output:
        data: pandas dataframe with new columns for
    """
    for category in category_lst:
        cat_lst = []
        cat_groups = data.groupby(category).mean(numeric_only=True)[response]
        for val in data[category]:
            cat_lst.append(cat_groups.loc[val])
        data[category + "_" + response] = cat_lst
        logging.info("SUCCESS: encoder %s", category)
    return data


def perform_feature_engineering(data):
    """
    input:
        data: pandas dataframe

    output:
        features_train: features training data
        features_test: features testing data
        labels_train: labels training data
        labels_test: labels testing data
    """
    labels = data['Churn']
    features = pd.DataFrame()
    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]
    features[keep_cols] = data[keep_cols]

    # Split into train and test
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.3, random_state=42)
    return (features_train, features_test, labels_train, labels_test)


def classification_report_image(
    labels_train,
    labels_test,
    labels_train_preds_lr,
    labels_train_preds_rf,
    labels_test_preds_lr,
    labels_test_preds_rf
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        labels_train: training response values
        labels_test:  test response values
        labels_train_preds_lr: training predictions from logistic regression
        labels_train_preds_rf: training predictions from random forest
        labels_test_preds_lr: test predictions from logistic regression
        labels_test_preds_rf: test predictions from random forest

    output:
        None
    """
    plt.close()
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(labels_test, labels_test_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                labels_train, labels_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_PATH, "rf_classification_report.png"))
    logging.info("SUCCESS: classification_report_image: random forest")

    # Save the logistic regression classification report image.
    plt.close()
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                labels_train, labels_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(labels_test, labels_test_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_PATH, "lr_classification_report.png"))
    logging.info(
        msg="SUCCESS: classification_report_image: logistic regression")


def feature_importance_plot(model, features_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        features_data: pandas dataframe of features values
        output_pth: path to store the figure

    output:
        None
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [features_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(features_data.shape[1]), importances[indices])

    # Add feature names as features-axis labels
    plt.xticks(range(features_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    logging.info(
        "%s : %s",
        "SUCCESS: feature_importance_plot and saved feature report to ",
        output_pth)


def train_models(features_train, features_test, labels_train, labels_test):
    """
    train, store model results: images + scores, and store models
    input:
        features_train: features training data
        features_test: features testing data
        labels_train: labels training data
        labels_test: labels testing data
    output:
        None
    """
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(features_train, labels_train)

    lrc.fit(features_train, labels_train)

    labels_train_preds_rf = cv_rfc.best_estimator_.predict(features_train)
    labels_test_preds_rf = cv_rfc.best_estimator_.predict(features_test)

    labels_train_preds_lr = lrc.predict(features_train)
    labels_test_preds_lr = lrc.predict(features_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(labels_test, labels_test_preds_rf))
    print('train results')
    print(classification_report(labels_train, labels_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(labels_test, labels_test_preds_lr))
    print('train results')
    print(classification_report(labels_train, labels_train_preds_lr))

    # Save the models
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            MODEL_PATH,
            "rfc_model.pkl"))
    joblib.dump(lrc, os.path.join(MODEL_PATH, "logistic_model.pkl"))

    # Save the classification report images
    classification_report_image(
        labels_train=labels_train,
        labels_test=labels_test,
        labels_train_preds_lr=labels_train_preds_rf,
        labels_train_preds_rf=labels_train_preds_lr,
        labels_test_preds_lr=labels_test_preds_lr,
        labels_test_preds_rf=labels_test_preds_rf
    )

    # Save the feature importance plot
    feature_importance_plot(
        model=cv_rfc,
        features_data=features_train,
        output_pth=os.path.join(RESULTS_PATH, "cv_feature_importance.png")
    )

    lrc_plot = RocCurveDisplay.from_estimator(
        lrc, features_test, labels_test)
    plt.savefig(os.path.join(RESULTS_PATH, "lr_roc_curve.png"))
    logging.info("SUCCESS: model_train: save lr roc curve")

    # Combine both lrc ans rfc roc plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    _ = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, features_test, labels_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(os.path.join(RESULTS_PATH, "lr_rf_roc_curves.png"))
    logging.info("SUCCESS: model_train: save lr and rf roc curves")


if __name__ == '__main__':
    data = import_data("./data/bank_data.csv")
    logging.info(data)
    perform_eda(data)
    logging.info("SUCCESS: perform_eda")
    logging.info("START: encoder_helper")
    data = encoder_helper(data, CAT_COLUMNS, "Churn")
    logging.info("SUCCESS: encoder_helper")
    logging.info("START: perform_feature_engineering")
    features_train, features_test, labels_train, labels_test = perform_feature_engineering(data)
    logging.info("SUCCESS: perform_feature_engineering")
    logging.info("START: train_models")
    train_models(features_train, features_test, labels_train, labels_test)
    logging.info("SUCCESS: train_models")