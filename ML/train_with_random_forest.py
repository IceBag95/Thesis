from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import setup

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
            rfc = RandomForestClassifier(n_estimators=n, random_state=101, criterion=criterion)
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
    plt.title(f'Random Forest Validation Error per Fold\nCriterion {criterion.upper}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../Dataset/Observations/plot_of_errors_random_forest_all_folds_with_criterion_{criterion}.png')
    plt.close()

    return stable_index


def train_nth_model(criterion: str,
                    X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_test: pd.DataFrame) -> Dict[str, Union[RandomForestClassifier, Optional[StandardScaler]]]:
    
    logs = Path.cwd().parent / "Dataset" / "Observations" / "Random Forest Metrics.txt"
    logfile = logs.open(mode='a')
    logfile.write(f"\n✍️ ========= {criterion.upper()} =========\n\n")
    rfc = RandomForestClassifier(random_state=101,criterion=criterion)
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
                                random_state=101)


    rfc.fit(X_train,y_train)

    preds = rfc.predict(X_test)

    acc = accuracy_score(y_test, preds)
    roc_auc =roc_auc_score(y_test, preds)
    print(classification_report(y_test,preds))
    print(f'\nAccuracy: {acc}')
    logfile.write(classification_report(y_test,preds))
    logfile.write(f'\nAccuracy: {acc}\n')
    print(f'ROC-AUC: {roc_auc}\n')
    logfile.write(f'ROC-AUC: {roc_auc}\n\n')
    logfile.close()
    return {
        'model': rfc,
        'accuracy': acc,
        'roc_auc': roc_auc
    }

def train_model() -> Dict[str, Union[RandomForestClassifier, Optional[StandardScaler]]]:

    print('Entered random forest training')

    logs = Path.cwd().parent / "Dataset" / "Observations" / "Random Forest Metrics.txt"
    if (logs.exists()):
        logs.unlink()

    logs.touch()

    setup.setup_dataset()

    df = pd.read_csv("../Dataset/clean_data.csv")

    X = pd.get_dummies(df.drop('target',axis=1),drop_first=True)

    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    models_list = []
    accs_list = []
    roc_auc_list = []

    for criterion in ['gini', 'entropy', 'log_loss']:
        print (f"\n✍️ ========= {criterion.upper()} =========\n")
        res1 = train_nth_model(criterion, X_train, y_train, X_test, y_test)
        models_list.append(res1.get('model'))
        accs_list.append(res1.get('accuracy'))
        roc_auc_list.append(res1.get('roc_auc'))


    max_acc = max(accs_list)
    best_models = []
    for i in range(len(models_list)):
        if accs_list[i] == max_acc:
            best_models.append(i)
    
    rfc = models_list[best_models[0]]

    logfile = logs.open(mode='a')
    
    print("\n✍️ ========= Results =========\n")
    logfile.write("\n✍️ ========= Results =========\n\n")

    if len(best_models) > 1:
        msg = 'Tie between:   '
        for idx in best_models:
            msg += f'rfc{idx+1} '
        
        print(msg)
        logfile.write(f'\n{msg}\n')
        print('Accuracy:'.ljust(15) + str(max_acc))
        logfile.write('\nAccuracy:'.ljust(15) + str(max_acc) + '\n')
        print(f'ROC-AUC: {roc_auc_list[best_models[0]]}\n')
        logfile.write(f'ROC-AUC: {roc_auc_list[best_models[0]]}\n\n')
        print(f'Picking rfc{best_models[0]+1} to continue')
        logfile.write(f'Picking rfc{best_models[0]+1} to continue\n\n')
    else:
        print(f"Best model: ".ljust(15) + f'{best_models[0]+1}')
        logfile.write(f"Best model: ".ljust(15) + f'{best_models[0]+1}\n')
        print('Accuracy:'.ljust(15) + str(max_acc))
        logfile.write('Accuracy:'.ljust(15) + str(max_acc) + '\n\n')
        print(f'ROC-AUC: {roc_auc_list[best_models[0]]}\n')
        logfile.write(f'ROC-AUC: {roc_auc_list[best_models[0]]}\n\n')
    

    return {
            'model': rfc,
            'scaler': None
        }


if __name__ == '__main__':
    train_model()

