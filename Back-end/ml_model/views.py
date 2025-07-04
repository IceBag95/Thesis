from typing import Any, Dict, List
from django.http import FileResponse, HttpRequest
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
import pandas as pd
from ml_model.apps import MlModelConfig
import json


def heart_attack_prediction(request: HttpRequest) -> FileResponse:

    index_file = Path(settings.FRONT_END_DIR).resolve() / 'build' / 'index.html'
    print(f'heart_attack_prediction - index_file path: {index_file}')
    
    return FileResponse(open(index_file, 'rb'), content_type='text/html')

@csrf_exempt
def make_heart_attack_prediction(request: HttpRequest) -> JsonResponse:

    if request.method == 'POST':

        ml_model  = MlModelConfig.get_model()
        ml_scaler = MlModelConfig.get_scaler()
        columns_and_limits = MlModelConfig.get_columns_and_limits()

        print(f'ML MODEL: {ml_model}')
        print(f'ML SCALER: {ml_scaler}')
        print(f'COLUMNS AND LIMITS: {columns_and_limits}')


        try:
            data:Dict[str, Any] = json.loads(request.body)
          
            user_data_list:List[Dict[str, Any]] = data.get('usr_ans_list')
            if user_data_list is None:
                raise json.JSONDecodeError("Missing 'usr_ans_list' key. No answers received from user.", "", 0)
            
            user_data_for_model = {}
            
            columns_list = []
            for item in columns_and_limits:
                columns_list.append(item.get("column"))
            
            columns_list_copy = columns_list[:]
            
            for obj in user_data_list:
                curr_col = obj.get('for_column')
                if curr_col in columns_list:
                    idx = columns_list.index(curr_col)
                    # Here we check x type if can be converted to anything else than string
                    x = obj.get('current_answer')
                    lower_limit = columns_and_limits[idx].get("limits")[0]
                    upper_limit = columns_and_limits[idx].get("limits")[1]
                    print(x)
                    print(lower_limit)
                    print(upper_limit)
                    
                    # We try float 
                    try:
                        x = float(x)
                        if not lower_limit <= x <= upper_limit:
                            return JsonResponse({'error': f'Answer out of bounds, please refresh and try again'}, status=400)
                    except Exception as e:
                        pass
                    
                    # Then int
                    try:
                        x = int(x)
                        if not lower_limit <= x <= upper_limit:
                            return JsonResponse({'error': f'Answer out of bounds, please refresh and try again'}, status=400)
                    except Exception as e:
                        pass
                    
                    # Then bool
                    if x == "True":
                        x = True
                    
                    if x == "False":
                        x = False


                    user_data_for_model[obj.get('for_column')] = x
                    columns_list_copy.remove(curr_col)

                    print(f"END FOR {curr_col}")
                else:
                    raise json.JSONDecodeError({f"Can't make prediction. Column {curr_col} does not seem to exist in the dataset.", "", 0})
            
            if len(columns_list_copy) > 0:
                raise json.JSONDecodeError(f"Can't make prediction. No data received for columns {columns_list_copy}", "", 0)

            print(f"User_data_for_model: {user_data_for_model}")
            try:
                df = pd.DataFrame([user_data_for_model])
                print(df.head())
                if ml_scaler is not None:
                    print('Performing scaling.')
                    df = pd.DataFrame(ml_scaler.transform(df), columns=df.columns)
                    print('Scaling performed.')
                    print(df.head())
                else:
                    print('No need to scale.')
                prediction = ml_model.predict(df)
                print(prediction)
                response_data = {'target': prediction.tolist()[0]}
                print(response_data)
            except Exception as e:
                print(f'Something went wrong. Could be format. Is this the format you need for the data:\n{user_data_for_model}')
            
            # Send a JSON response back to the frontend
            return JsonResponse(response_data, status=200)
        except json.JSONDecodeError as e:
            return JsonResponse({'error': f'Invalid JSON data {e.__str__}'}, status=400)

    return JsonResponse({
        "error": "no data",
    },status=404)

def send_initial_info(request: HttpRequest) -> JsonResponse:

    initial_data_path = Path(settings.BASE_DIR).resolve() / 'Assets' / 'Initial_data_for_predict.json'
    print(f'Initial_data path: {initial_data_path}')

    file = open(initial_data_path, 'r')
    initial_data = json.load(file)
    file.close()

    return JsonResponse(initial_data)
