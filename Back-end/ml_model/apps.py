import sys
import os
from django.apps import AppConfig

class MlModelConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_model'

    def ready(self):

        print("setting up path")

        # Correct the path to the 'ML' folder by going up three levels
        ml_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ML')
        sys.path.append(ml_folder_path)
        
        print("Import path")
        from train_with_adaboost import train_model

        print("Train model")
        self.model = train_model()
