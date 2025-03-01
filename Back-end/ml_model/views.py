import json
from django.http import FileResponse
from django.conf import settings
from django.http import JsonResponse
import os

def heart_attack_prediction(request):

    index_file = os.path.join(settings.FRONT_END_DIR, 'index.html')
    
    # Open and serve the index.html file as a response
    return FileResponse(open(index_file, 'rb'), content_type='text/html')

def make_heart_attack_prediction(request):

    return JsonResponse({
        "error": "no data",
    })
