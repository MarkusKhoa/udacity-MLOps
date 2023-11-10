# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is the project to practice clean code in a Machine learning project.
## Files and data description
Here is the files and folders structure built in this project
`
├── README.md
├── churn_library.py
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── churn_histogram.png
│   │   ├── customer_age_histogram.png
│   │   ├── heatmap.png
│   │   ├── marital_status_counts.png
│   │   └── total_transaction_histogram.png
│   └── results
│       ├── cv_feature_importance.png
│       ├── lr_classification_report.png
│       ├── lr_rf_roc_curves.png
│       ├── lr_roc_curve.png
│       └── rf_classification_report.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
└── requirements.txt
`

## Running Files
To install environment variables, run this command
`pip install -r requirements.txt`

Usage
`python churn_library.py`

There are two different ways to test the Python script
 - Directly execute the script by using command (main() functon required)
 `ipython churn_script_logging_and_tests.py`
 - Using `pytest` command:
 `pytest churn_script_logging_and_tests.py`