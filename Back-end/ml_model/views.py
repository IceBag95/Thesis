from django.http import FileResponse
from django.conf import settings
from django.http import JsonResponse
from pathlib import Path
import json

from django.shortcuts import render

def heart_attack_prediction(request):

    index_file = Path(settings.FRONT_END_DIR).resolve() / 'build' / 'index.html'
    print(f'Batman - index_file path: {index_file}')
    
    return FileResponse(open(index_file, 'rb'), content_type='text/html')

def make_heart_attack_prediction(request):

    return JsonResponse({
        "error": "no data",
    })

def send_initial_info(request):

    initial_data_path = Path(settings.BASE_DIR).resolve() / 'Assets' / 'Initial_data_for_predict.json'
    print(f'Batman - Initial_data path: {initial_data_path}')

    file = open(initial_data_path, 'r')
    initial_data = json.load(file)

    return JsonResponse(initial_data)
