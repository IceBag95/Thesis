# import joblib
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import setup



def plot_errors_for_n_estimators(base_learner, X_train, y_train, X_test, y_test):
    errors = []
    missclassifications = []

    prev_err = float('inf')
    stable_index = -1


    for n in range(50,500, 50):
        rfc = AdaBoostClassifier(estimator=base_learner, n_estimators=n, random_state=101)
        rfc.fit(X_train, y_train)
        preds = rfc.predict(X_test)
        err =  1 - accuracy_score(y_test,preds)
        n_missed = np.sum( preds != y_test)
        
        errors.append(err)
        missclassifications.append(n_missed)

        print(f'Index:{n}\nError:{err}\nPrevious Error:{prev_err}\nError diff:{abs(prev_err - err)}\nIs Prev erro diff that current {prev_err != err}')

        if abs(prev_err - err) < 0.1 and prev_err != err:
            stable_index = n
        
        prev_err = err

    # print(f'Min missclassifications {min(errors)} at index {missclassifications.index(min(errors))}')
    # plt.plot(range(50,500),errors)
    # plt.show()
    print(f'Stable Index: {stable_index}')
    return stable_index

def train_model():

    print('Entered adaboost training')

    setup.setup_dataset()

    df = pd.read_csv("../Dataset/clean_data.csv")

    X = pd.get_dummies(df.drop('target',axis=1),drop_first=True)

    y = df['target']

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    base_learner = DecisionTreeClassifier(max_depth=1)

    ada_boost = AdaBoostClassifier(estimator=base_learner, random_state=101)

    current_least_error_index = plot_errors_for_n_estimators(base_learner, X_train, y_train, X_test, y_test) # should be > 50

    if current_least_error_index == -1: 
        current_least_error_index = 100 # I will assign as default the 100 n_estimators


    # Grid Search for best params after the stabilazation point of the plot
    grid_params = {
        'n_estimators': list(range(current_least_error_index,501,10))
    }

    grid = GridSearchCV(ada_boost ,param_grid=grid_params, verbose=3)
    grid.fit(X_train,y_train)

    print(grid.best_params_)

    ada_boost_model = AdaBoostClassifier(estimator=base_learner, n_estimators=grid.best_params_.get('n_estimators'), random_state=101)
    ada_boost_model.fit(X_train,y_train)
    preds = ada_boost_model.predict(X_test)
    print(classification_report(y_test,preds))

    return ada_boost_model
