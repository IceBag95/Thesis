from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def heart_attack_prediction(request):
    return HttpResponse("This is my custom view!")