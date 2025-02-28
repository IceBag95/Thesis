# from django.http import HttpResponse
# from django.shortcuts import render


# # Create your views here.
# def heart_attack_prediction(request):
#     return render(request, 'index.html')

from django.http import FileResponse
from django.conf import settings
import os

# Define the view to serve React's index.html
def heart_attack_prediction(request):
    # Construct the path to the index.html file in your React build folder
    print(f'Batman - FRONT_END_DIR: {settings.FRONT_END_DIR}')
    index_file = os.path.join(settings.FRONT_END_DIR, 'index.html')
    
    # Open and serve the index.html file as a response
    return FileResponse(open(index_file, 'rb'), content_type='text/html')
