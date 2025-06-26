from typing import Dict, List, Optional, Union
import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from  pathlib import Path
import setup



class LrcModel:

    counter = 0

    def __init__(self, model, cls_rprt, accuracy, roc_auc):
        LrcModel.counter += 1
        self.id = LrcModel.counter
        self.model = model
        self.cls_rprt = cls_rprt
        self.accuracy = accuracy
        self.roc_auc = roc_auc

    def get_id(self):
        return self.id
    
    def get_model(self):
        return self.model

    def get_classification_report(self):
        return self.cls_rprt

    def get_accuracy(self):
        return self.accuracy

    def get_roc_auc(self):
        return self.roc_auc



def train_nth_model(penalty: str,
                    X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_test: pd.DataFrame) -> LrcModel:
    
    logs = Path.cwd().parent / "Dataset" / "Observations" / "Logistic Regression Metrics.txt"

    logfile = logs.open('a')

    params = {
        "cv": 10,
        "Cs": 50,
        "solver": 'saga',
        "penalty": penalty,
        "max_iter": 1000,
        "random_state": 101,
        "class_weight": 'balanced',
        "verbose": 3
    }

    if penalty == 'elasticnet':
        params['l1_ratios'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    lrc = LogisticRegressionCV(**params)


    lrc.fit(X_train, y_train)
    preds = lrc.predict(X_test)
    cls_rprt = classification_report(y_test,preds)
    accuracy = accuracy_score(y_test,preds)
    roc_auc = roc_auc_score(y_test,preds)
    print(f'\n\n✍️ ======= LogisticRegressionCV with penalty: {penalty}')
    print(cls_rprt)
    print(f'Precise accuracy {accuracy}')
    print(f'Roc-Auc: {roc_auc}')
    logfile.write(f'\n\n======= LogisticRegressionCV with penalty: {penalty} =======\n')
    logfile.write(cls_rprt)
    logfile.write(f'\nPrecise accuracy {accuracy}\n')
    logfile.write(f'Roc-Auc: {roc_auc}\n')
    logfile.write('\n')

    logfile.close()

    return LrcModel(lrc, cls_rprt, accuracy, roc_auc)



def pick_best_model_based_on(metric: str, models_list: List[LrcModel]) -> List[LrcModel]:
    metrics_list = []
    best_models = []
    
    # We define a dict mapping the metric to the class method NAME
    metric_getters = {
        'accuracy': 'get_accuracy',
        'roc_auc': 'get_roc_auc',
        'classification_report': 'get_classification_report'
    }
    
    # We get the method name 
    getter_name = metric_getters[metric]

    
    for model in models_list:
        metric_method = getattr(model, getter_name)   # get the actual class method of each 'model' obj
        metrics_list.append(metric_method())          # execute the method and append the result to the list
    

    max_value =  max(metrics_list)
    for idx, val in enumerate(metrics_list):
        if val == max_value:
            best_models.append(models_list[idx])

    return best_models



def train_model() -> Dict[str, Union[LogisticRegressionCV, Optional[StandardScaler]]]:

    print('Entered LogisticRegressionCV training')

    model_name = 'Logistic Regression'
    
    setup.setup_dataset()


    logs = Path.cwd().parent / "Dataset" / "Observations" / "Logistic Regression Metrics.txt"
    
    if logs.exists():
        logs.unlink()
    
    logs.touch()

    df = pd.read_csv("../Dataset/clean_data.csv")

    # Removing outliers for Logistic Regression
    print(f'\n⏳ Removing columns with outliers for {model_name}')
    idx_list = []
    for col in df.columns:
        zscores = np.abs(stats.zscore(df[col]))
        temp = df[zscores > 3]                   # anything that scores above 3 is considered an outlier
        current_idx_list = temp.index            # get the indexes of the rows returned and store them into a variable 
        for idx in current_idx_list:
            if idx not in idx_list:
                idx_list.append(idx)
    print(f'>> Found and proceeding to remove {len(idx_list)} rows with outlier values in at least one of their columns...')
    df = df.drop(index=idx_list).reset_index(drop=True)
    print('✅ Removal SUCCESS\n')

    X = pd.get_dummies(df.drop('target',axis=1),drop_first=True)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    scaler = StandardScaler()

    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    penalties = ['l1', 'l2', 'elasticnet']
    models_list = []
    for penalty in penalties:
        lrc = train_nth_model(penalty, X_train, y_train, X_test, y_test)
        models_list.append(lrc)

    best_models = pick_best_model_based_on('accuracy', models_list)
    best_models = pick_best_model_based_on('roc_auc', best_models)

    logfile = logs.open('a')

    # If multiple best model found, create a message to display them
    print('\n\n✍️\t ======= Results =======\n')
    if len(best_models) > 1:
        msg = 'Tie between models: '
        for model in best_models:
            msg += f'lr{model.get_id()} '
        print(f'\n{msg}\n')

    print(f'Selected model: {best_models[0].get_model()}')
    print(f'{best_models[0].get_classification_report()}')
    print('\nPrecise Accuracy:'.ljust(20) + f'{best_models[0].get_accuracy()}')
    print(f'ROC_AUC: {best_models[0].get_roc_auc()}\n')


    logfile.write('\n\n✍️\t======= Results =======\n')
    if len(best_models) > 1:
        logfile.write(f'\n{msg}\n\n')
    logfile.write(f'Selected model: {best_models[0].get_model()}\n\n')
    logfile.write(f'{best_models[0].get_classification_report()}\n')
    logfile.write('\nPrecise Accuracy:'.ljust(20) + f'{best_models[0].get_accuracy()}\n')
    logfile.write(f'ROC_AUC: {best_models[0].get_roc_auc()}\n')

    logfile.close()

    selected_model = best_models[0].get_model()

    return {
            'model': selected_model,
            'scaler': scaler
        }



if __name__ == '__main__':
    train_model()
