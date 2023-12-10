import os
import json

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_configuration(production = False):
    with open("config.json") as config_file:
        config = json.load(config_file)
    if production:
        config["input_folder_path"] = "sourcedata"
        config["output_folder_path"] = "models"
    
    create_folder(config["output_model_path"])
    create_folder(config["prod_deployment_path"])
    
    config["test_data_csv_path"] = os.path.join(
        os.getcwd(),
        config["test_data_path"],
        'testdata.csv'
    )
    config["final_data_path"] = os.path.join(
        os.getcwd(),
        config["output_folder_path"],
        'finaldata.csv'
    )
    config["api_returns_path"] = os.path.join(
        os.getcwd(),
        config["output_model_path"],
        'apireturns2.txt' if production else 'apireturns.txt'
    )
    config["cfm_path"] = os.path.join(
        os.getcwd(),
        config["output_model_path"],
        'confusionmatrix2.png' if production else 'confusionmatrix.png'
    )
    return config

if __name__ == 'main':
    config = get_configuration()