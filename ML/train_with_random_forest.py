from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, RocCurveDisplay
import setup



class RfcModel:
    counter = 0

    def __init__(self, model, cls_rprt, accuracy, roc_auc, cv_results):
        RfcModel.counter += 1
        self.id = f'rfc{RfcModel.counter}'
        self.model = model
        self.cls_rprt = cls_rprt
        self.accuracy = accuracy
        self.roc_auc = roc_auc
        self.cv_results = cv_results

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
    
    def get_cv_results(self):
        return self.cv_results
    
    


# In case we need it
def plot_errors_for_n_estimators(X_train: pd.DataFrame, y_train: pd.DataFrame, criterion: str) -> int:
    skf = StratifiedKFold(n_splits=5)
    best_estimators: List[int] = []
    total_errors: List[List[float]] = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        errors = []
        prev_err = float('inf')
        stable_n = 100
        got_stable_idx = False

        for n in range(100, 501, 50):
            rfc = RandomForestClassifier(n_estimators=n, random_state=101, criterion=criterion, class_weight='balanced')
            rfc.fit(X_tr, y_tr)
            preds = rfc.predict(X_val)
            err = 1 - accuracy_score(y_val, preds)
            errors.append(err)

            if abs(prev_err - err) < 0.1 and not got_stable_idx:
                stable_n = n
                got_stable_idx = True
            prev_err = err

        best_estimators.append(stable_n)
        total_errors.append(errors)

    stable_index = (int(np.median(best_estimators)) // 50) * 50

    # Plot all folds errors on the same graph
    for fold_idx, errors in enumerate(total_errors, 1):
        plt.plot(range(100, 501, 50), errors, label=f'Fold {fold_idx}')

    plt.xlabel('Number of Estimators')
    plt.ylabel('Validation Error')
    plt.title(f'Random Forest Validation Error per Fold\nCriterion "{criterion}"')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../Observations/Random Forest/plot_of_errors_random_forest_all_folds_with_criterion_{criterion}.png')
    plt.close()

    return stable_index



def evaluate_with_cv(model:RandomForestClassifier, X:pd.DataFrame, y:pd.DataFrame, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies, aucs, precisions, recalls, f1s = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

    results = {
        'Metric': ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1 Score'],
        'Mean': [np.mean(accuracies), np.mean(aucs), np.mean(precisions), np.mean(recalls), np.mean(f1s)],
        'Std Dev': [np.std(accuracies), np.std(aucs), np.std(precisions), np.std(recalls), np.std(f1s)]
    }

    results_df = pd.DataFrame(results)
    return results_df



def train_nth_model(criterion: str,
                    X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_test: pd.DataFrame) -> RfcModel:
    
    logs = Path.cwd().parent / "Observations" / "Random Forest" / "Random Forest Metrics.txt"
    logfile = logs.open(mode='a')
    logfile.write(f"\n✍️ ========= {criterion.upper()} =========\n\n")
    rfc = RandomForestClassifier(random_state=101,criterion=criterion, class_weight='balanced')
    stable_idx = plot_errors_for_n_estimators(X_train, y_train, criterion)

    grid_params = {
        'n_estimators'    : list(range(stable_idx,501, 50)),
        'max_depth'       : [4,5],
        'max_features'    : ['sqrt', 'log2'],
        'min_samples_leaf': [1,2,3,4,5],
        'bootstrap'       : [True, False]
    }

    grid = GridSearchCV(rfc ,param_grid=grid_params, verbose=3)
    grid.fit(X_train,y_train)

    print(grid.best_params_)

    best_max_depth = grid.best_params_.get('max_depth')
    best_use_bootstrap = grid.best_params_.get('bootstrap')
    best_n_estimators = grid.best_params_.get('n_estimators')
    best_max_features = grid.best_params_.get('max_features')
    best_min_samples_leaf = grid.best_params_.get('min_samples_leaf')


    rfc = RandomForestClassifier(n_estimators=best_n_estimators, 
                                max_depth=best_max_depth,
                                max_features=best_max_features,
                                min_samples_leaf=best_min_samples_leaf,
                                bootstrap=best_use_bootstrap,
                                class_weight='balanced',
                                random_state=101)


    rfc.fit(X_train,y_train)

    cv_results = evaluate_with_cv(rfc, X_train, y_train)

    preds = rfc.predict(X_test)

    acc = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1])
    cls_rprt = classification_report(y_test,preds)
    print(classification_report(y_test,preds))
    print(f'\nAccuracy: {acc}')
    logfile.write(classification_report(y_test,preds))
    logfile.write(f'\nAccuracy: {acc}\n')
    print(f'ROC-AUC: {roc_auc}\n')
    logfile.write(f'ROC-AUC: {roc_auc}\n\n')
    print(f'\nCross Validation Results {rfc}:\n\n{cv_results}\n')
    logfile.write(f'\nCross Validation Results {rfc}:\n\n{cv_results}\n')
    logfile.close()

    # Make and save image for confusion matrix for this model
    filepath = Path.cwd().parent / "Observations" / "Random Forest" / f"Confusion_Matrix_RFC_with_criterion_{criterion}"
    cm = confusion_matrix(y_test,preds)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix\nRandom Forest with criterion: {criterion}')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.savefig(filepath)
    plt.close()

    # Make and save Roc Curve graph for this model
    filepath = Path.cwd().parent / "Observations" / "Random Forest" / f"Roc_Curve_RFC_with_criterion_{criterion}"
    RocCurveDisplay.from_estimator(rfc, X_test, y_test)

    plt.savefig(filepath)
    plt.close()

    ret_model = RfcModel(rfc, cls_rprt, acc, roc_auc, cv_results)
    
    return ret_model



def pick_best_model_based_on(metric: str, models_list: List[RfcModel]) -> List[RfcModel]:
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




def train_model() -> Dict[str, Union[RandomForestClassifier, Optional[StandardScaler]]]:

    print('Entered random forest training')

    setup.setup_dataset()
    rf_obsrv_path = Path.cwd().parent / "Observations" / "Random Forest"

    if not rf_obsrv_path.exists():
        rf_obsrv_path.mkdir(parents=True, exist_ok=True)

    logs = Path.cwd().parent / "Observations" / "Random Forest" / "Random Forest Metrics.txt"
    if (logs.exists()):
        logs.unlink()

    logs.touch()

    df = pd.read_csv("../Dataset/clean_data.csv")

    X = pd.get_dummies(df.drop('target',axis=1),drop_first=True)

    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    models_list = []

    for criterion in ['gini', 'entropy', 'log_loss']:
        print (f"\n✍️ ========= {criterion.upper()} =========\n")
        res = train_nth_model(criterion, X_train, y_train, X_test, y_test)
        models_list.append(res)

    
    # Start checks to find best model
    # First for accuracy
    best_models = pick_best_model_based_on('accuracy', models_list)
    
    # If we get more than one we go for roc_auc
    if len(best_models) > 1:
        best_models= pick_best_model_based_on('roc_auc', best_models)

    # ....
    # If we have more than one again we can continue here for more metrics, but the whole module
    # will need to be adjusted to take into consideration these metrics



    logfile = logs.open(mode='a')
    
    print("\n✍️ ========= Results =========\n")
    logfile.write("\n✍️ ========= Results =========\n\n")

    if len(best_models) > 1:
        msg = 'Tie between:   '
        for model in best_models:
            msg += f'rfc{model.get_id()} '
        
        print(msg)
        logfile.write(f'\n{msg}\n')
    
    print(best_models[0].get_classification_report())
    logfile.write(f'{best_models[0].get_classification_report()}\n')
    print('\nAccuracy:'.ljust(15) + str(best_models[0].get_accuracy()))
    logfile.write('\nAccuracy:'.ljust(15) + str(best_models[0].get_accuracy()) + '\n')
    print(f'ROC-AUC:'.ljust(15) + str(best_models[0].get_roc_auc()) + '\n')
    logfile.write(f'ROC-AUC:'.ljust(15) + str(best_models[0].get_roc_auc()) + '\n\n')
    print(f'\n✍️ Cross Validation Results {best_models[0].get_id()}:\n{best_models[0].get_cv_results()}\n')
    logfile.write(f'\n✍️ Cross Validation Results {best_models[0].get_id()}:\n{best_models[0].get_cv_results()}\n')
    
    if len(best_models) > 1:
        print(f'Picking {best_models[0].get_id()} to continue')
        logfile.write(f'Picking {best_models[0].get_id()} to continue\n\n')
    
    rfc = best_models[0].get_model()
    print(f'Best model: {rfc}')

    return {
            'model': rfc,
            'scaler': None
        }




if __name__ == '__main__':
    train_model()

