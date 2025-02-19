from django.apps import AppConfig
from ML.train_with_adaboost import train_model


class MlModelConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_model'

    def ready(self):
        self.model = train_model()
