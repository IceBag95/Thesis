# import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import setup



def plot_errors_for_n_estimators(base_learner: DecisionTreeClassifier, 
                                 X_train: pd.DataFrame,
                                 y_train: pd.DataFrame) -> int:
    
    skf = StratifiedKFold(n_splits=5)
    best_estimators = []
    total_errors:List[List[float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        sample_weights = compute_sample_weight(class_weight='balanced', y=y_tr)

        errors = []
        prev_err = float('inf')
        stable_n = 50
        got_stable_idx = False

        for n in range(50, 501, 50):
            model = AdaBoostClassifier(estimator=base_learner, n_estimators=n, random_state=101)
            model.fit(X_tr, y_tr, sample_weight=sample_weights)
            preds = model.predict(X_val)
            err = 1 - accuracy_score(y_val, preds)
            errors.append(err)

            if abs(prev_err - err) < 0.01 and not got_stable_idx:
                stable_n = n
                got_stable_idx = True
            prev_err = err

        best_estimators.append(stable_n)
        total_errors.append(errors)

    stable_index = int(np.median(best_estimators))

    # Plot all folds errors on the same graph
    for fold_idx, errors in enumerate(total_errors, 1):
        plt.plot(range(50, 501, 50), errors, label=f'Fold {fold_idx}')

    plt.xlabel('Number of Estimators')
    plt.ylabel('Validation Error')
    plt.title('AdaBoost Validation Error per Fold')
    plt.legend()
    plt.grid(True)
    plt.savefig('../Dataset/Observations/plot_of_errors_adaboost_all_folds.png')
    plt.close()

    print(f'Stable Index: {stable_index}')
    return stable_index


def train_model() -> Dict[str, Union[AdaBoostClassifier, Optional[StandardScaler]]]:

    print('Entered adaboost training')

    logs = Path.cwd().parent / "Dataset" / "Observations" / "Adaboost Metrics.txt"
    logs.touch()
    
    logfile = logs.open(mode='w')

    setup.setup_dataset()

    df = pd.read_csv("../Dataset/clean_data.csv")

    X = pd.get_dummies(df.drop('target',axis=1),drop_first=True)

    y = df['target']

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    base_learner = DecisionTreeClassifier(max_depth=1, class_weight='balanced')

    ada_boost = AdaBoostClassifier(estimator=base_learner, random_state=101)

    current_least_error_index = plot_errors_for_n_estimators(base_learner, X_train, y_train) # should be > 50

    if current_least_error_index == -1: 
        current_least_error_index = 100 # I will assign as default the 100 n_estimators


    # Grid Search for best params after the stabilazation point of the plot
    grid_params = {
        'n_estimators': list(range(current_least_error_index,501))
    }

    grid = GridSearchCV(ada_boost ,param_grid=grid_params, verbose=3)
    grid.fit(X_train,y_train, sample_weight=sample_weights)

    print(grid.best_params_)

    ada_boost_model = AdaBoostClassifier(estimator=base_learner, 
                                         n_estimators=grid.best_params_.get('n_estimators'), 
                                         random_state=101)
    ada_boost_model.fit(X_train,y_train, sample_weight=sample_weights)
    preds = ada_boost_model.predict(X_test)
    print('\n\n✍️ ======= AdaBoost model =======\n')
    logfile.write('\n\n✍️ ======= AdaBoost model =======\n\n')
    print(classification_report(y_test,preds))
    logfile.write('\n' + classification_report(y_test,preds) + '\n\n')
    print(f'Accuracy achieved: {accuracy_score(y_test, preds)}\n')
    logfile.write(f'Accuracy achieved: {accuracy_score(y_test, preds)}\n')
    print(f'ROC-AUC: {roc_auc_score(y_test, preds)}\n')
    logfile.write(f'ROC-AUC: {roc_auc_score(y_test, preds)}\n')

    return {
            'model': ada_boost_model,
            'scaler': None
        }


if __name__ == '__main__':
    train_model()