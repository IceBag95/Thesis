import json
import sys
import os
from django.apps import AppConfig
from pathlib import Path

class MlModelConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_model'
    model = None
    model_columns = None
    scaler = None

    @classmethod
    def ready(cls):
        print("setting up path....")
        
        ml_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ML')
        sys.path.append(ml_folder_path)
        
        print("Import path....")
        from train_logistic_regression_model import train_model

        print("Train model....")
        results = train_model()
        cls.model = results.get('model')
        cls.scaler = results.get('scaler')
        print('Getting the model columns....')
        model_columns_file = open(Path.cwd() / "Assets" / "columns.json", 'r')
        model_columns_content:dict = json.load(model_columns_file)
        print(model_columns_content)
        cls.model_columns = model_columns_content.get('dataset_columns')
        model_columns_file.close()

        print('Model set!')

    @classmethod
    def get_model(cls):
        return cls.model
    
    @classmethod
    def get_columns(cls) -> list:
        return cls.model_columns
    
    @classmethod
    def get_scaler(cls):
        return cls.scaler