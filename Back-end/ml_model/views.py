from django.http import FileResponse
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path

import numpy as np
import pandas as pd
from ml_model.apps import MlModelConfig
import json


def heart_attack_prediction(request):

    index_file = Path(settings.FRONT_END_DIR).resolve() / 'build' / 'index.html'
    print(f'Batman - index_file path: {index_file}')
    
    return FileResponse(open(index_file, 'rb'), content_type='text/html')

@csrf_exempt
def make_heart_attack_prediction(request):

    if request.method == 'POST':

        ml_model = MlModelConfig.get_model()
        columns = MlModelConfig.get_columns()

        print(f'ML MODEL: {ml_model}')
        print(f'COLUMNS: {columns}')


        try:
            data = json.loads(request.body)
          
            user_data_list = data.get('usr_ans_list')
            if user_data_list is None:
                raise json.JSONDecodeError("Missing 'usr_ans_list' key. No answers received from user.", "", 0)
            
            user_data_for_model = {}
            columns_copy = columns[:]
            for obj in user_data_list:
                curr_col = obj.get('for_column')
                if curr_col in columns_copy:
                    
                    # Here we check x type if can be converted to anything else than string
                    x = obj.get('current_answer')
                    print(x)
                    
                    # We try float 
                    try:
                        x = float(x)
                    except Exception as e:
                        pass
                    
                    # Then int
                    try:
                        x = int(x)
                    except Exception as e:
                        pass
                    
                    # Then bool
                    if x == "True":
                        x = True
                    
                    if x == "False":
                        x = False

                    # If nothing then let it be str
                    print(x)

                    user_data_for_model[obj.get('for_column')] = x
                    columns_copy.remove(curr_col)
                else:
                    raise json.JSONDecodeError(f"Can't make prediction. Column {curr_col} does not seem to exist in the dataset.", "", 0)
            
            if len(columns_copy) > 0:
                raise json.JSONDecodeError(f"Can't make prediction. No data received for columns {columns_copy}", "", 0)

            print(f"User_data_for_model: {user_data_for_model}")
            try:
                df = pd.DataFrame([user_data_for_model])
                prediction = ml_model.predict(df)
                print(type(prediction))
                response_data = {'target': prediction.tolist()[0]}
                print(response_data)
            except Exception as e:
                print(f'Something went wrong. Could be format. Is this the format you need for the data:\n{user_data_for_model}')
            
            # Send a JSON response back to the frontend
            return JsonResponse(response_data)
        except json.JSONDecodeError as e:
            return JsonResponse({'status': 'error', 'message': f'Invalid JSON data {e.__str__}'}, status=400)

    return JsonResponse({
        "error": "no data",
    })

def send_initial_info(request):

    initial_data_path = Path(settings.BASE_DIR).resolve() / 'Assets' / 'Initial_data_for_predict.json'
    print(f'Initial_data path: {initial_data_path}')

    file = open(initial_data_path, 'r')
    initial_data = json.load(file)
    file.close()

    return JsonResponse(initial_data)
