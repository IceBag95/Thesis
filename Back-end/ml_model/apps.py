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

    @classmethod
    def ready(cls):
        print("setting up path....")
        
        ml_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ML')
        sys.path.append(ml_folder_path)
        
        print("Import path....")
        from train_with_adaboost import train_model

        print("Train model....")
        cls.model = train_model()

        print('Getting the model columns....')
        model_columns_file = open(Path.cwd() / "Assets" / "columns.json", 'r')
        model_columns_content = json.load(model_columns_file)
        print(model_columns_content)
        cls.model_columns = model_columns_content.get('dataset_columns')
        model_columns_file.close()

        print('Model set!')

    @classmethod
    def get_model(cls):
        return cls.model
    
    @classmethod
    def get_columns(cls):
        return cls.model_columns