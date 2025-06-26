import json
import sys
import os
from django.apps import AppConfig
from pathlib import Path

class MlModelConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_model'
    model = None
    model_columns_and_limits = None
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
        model_columns_file = open(Path.cwd() / "Assets" / "Initial_data_for_predict.json", 'r')
        initial_data_content:dict = json.load(model_columns_file)
        print(initial_data_content)    # this can go

        columns_and_limits = []
        for item in initial_data_content.get('qna'):
            column = item.get("for_column")
            if len(item.get("limits")) > 0:
                limits = [item.get("limits").get("lower_limit"), item.get("limits").get("upper_limit")]
            else:
                limits = [0, len(item.get("answers"))-1]
            columns_and_limits.append({
                "column": column,
                "limits": limits 
            })

        cls.model_columns_and_limits = columns_and_limits[:]
        model_columns_file.close()

        print('Model set!')

    @classmethod
    def get_model(cls):
        return cls.model
    
    @classmethod
    def get_columns_and_limits(cls) -> list:
        return cls.model_columns_and_limits
    
    @classmethod
    def get_scaler(cls):
        return cls.scaler