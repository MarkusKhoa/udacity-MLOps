import os
import logging
import pytest
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)
DATA_PATH = "./data/bank_data.csv"


@pytest.fixture(scope="module")
def import_data():
    '''
    Fixture - The test function test_import() will
    use the return of import_data() as an argument
    '''
    return cls.import_data


def test_import(import_data):
    '''
    test data import
    '''
    try:
        data = import_data(DATA_PATH)
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture(scope="module")
def perform_eda():
    '''
    Fixture - The test function test_eda() will
    use the return of eda() as an argument
    '''
    return cls.perform_eda


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        data_local = cls.import_data(DATA_PATH)
        perform_eda(data_local)
        assert os.path.exists('./images/eda/churn_histogram.png')
        assert os.path.exists('./images/eda/heatmap.png')
        assert os.path.exists('./images/eda/total_transaction_histogram.png')
        assert os.path.exists('./images/eda/customer_age_histogram.png')
        assert os.path.exists('./images/eda/marital_status_counts.png')
        logging.info("Testing perform_eda: SUCCESS")

    except AssertionError as err:
        logging.error(
            "FAILED: Testing perform_eda")
        raise err


@pytest.fixture(scope="module")
def encoder_helper():
    '''
    Fixture - The test function test_eda() will
    use the return of eda() as an argument
    '''
    return cls.encoder_helper


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    data_test = cls.import_data(DATA_PATH)
    cls.perform_eda(data_test)

    try:
        data_test = encoder_helper(data_test, cls.CAT_COLUMNS, "Churn")
        assert data_test.shape[0] == 10127
        logging.info("Testing encoder_helper: SUCCESS")

    except AssertionError as err:
        logging.error("Testing encoder_helper: FAILED!")
        raise err


@pytest.fixture(scope="module")
def perform_feature_engineering():
    '''
    Fixture - The test function test_perform_feature_engineering() will
    use the return of perform_feature_engineering() as an argument
    '''
    return cls.perform_feature_engineering


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    data_test = cls.import_data('./data/bank_data.csv')
    cls.perform_eda(data_test)

    data_test = cls.encoder_helper(
        data=data_test,
        category_lst=cls.CAT_COLUMNS,
        response="Churn")

    try:
        features_train, features_test, labels_train, labels_test = perform_feature_engineering(data_test)
        assert features_train.shape[0] and labels_train.shape[0] == 7088 \
            and features_test.shape[0] and labels_test.shape[0] == 3039
        logging.info("SUCCESS: Testing perform_feature_engineering")

    except AssertionError as err:
        logging.error("FAILED: Testing perform_feature_engineering!")
        raise err


@pytest.fixture(scope="module")
def train_models():
    '''
    Fixture - The test function test_perform_feature_engineering() will
    use the return of perform_feature_engineering() as an argument
    '''
    return cls.train_models


def test_train_models(train_models):
    '''
    test train_models
    '''
    data_test = cls.import_data('./data/bank_data.csv')
    cls.perform_eda(data_test)

    data_test = cls.encoder_helper(
        data=data_test,
        category_lst=cls.CAT_COLUMNS,
        response="Churn")

    features_train, features_test, labels_train, labels_test = cls.perform_feature_engineering(
        data=data_test)

    try:
        train_models(features_train, features_test, labels_train, labels_test)
        assert os.path.exists('models/logistic_model.pkl')
        assert os.path.exists('models/rfc_model.pkl')
        logging.info("SUCCESS: Testing train_models")

    except AssertionError as err:
        logging.error("FAILED: Testing train_models!")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)