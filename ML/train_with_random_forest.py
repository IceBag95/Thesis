import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import setup

# In case we need it
def plot_errors_for_n_estimators(X_train, y_train, X_test, y_test):
    errors = []
    missclassifications = []

    prev_err = float('inf')
    stable_index = -1


    for n in range(100, 500, 50):
        rfc = RandomForestClassifier(n_estimators=n, random_state=101)
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

    plt.plot(range(100,500,50),errors)
    plt.savefig('../Dataset/Observations/plot_of_errors_random_forest.png')
    print(f'Stable Index: {stable_index}')
    return stable_index


def train_nth_model(criterion, X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier(random_state=101,criterion=criterion)
    stable_idx = plot_errors_for_n_estimators(X_train, y_train, X_test, y_test)

    grid_params = {
        'n_estimators': list(range(stable_idx,501, 50)),
        'max_depth'   : [4,5],
        'bootstrap'   : [True]
    }

    grid = GridSearchCV(rfc ,param_grid=grid_params, verbose=3)
    grid.fit(X_train,y_train)

    print(grid.best_params_)

    best_max_depth = grid.best_params_.get('max_depth')
    best_use_bootstrap = grid.best_params_.get('bootstrap')
    best_n_setimators = grid.best_params_.get('n_estimators')

    rfc = RandomForestClassifier(n_estimators=best_n_setimators, 
                                max_depth=best_max_depth,
                                bootstrap=best_use_bootstrap,
                                random_state=101)


    rfc.fit(X_train,y_train)

    preds = rfc.predict(X_test)

    print(classification_report(y_test,preds))
    acc = accuracy_score(y_test, preds)
    return {
        'model': rfc,
        'accuracy': acc
    }

def train_model():

    print('Entered random forest training')

    setup.setup_dataset()

    df = pd.read_csv("../Dataset/clean_data.csv")

    X = pd.get_dummies(df.drop('target',axis=1),drop_first=True)

    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    models_list = []
    accs_list = []

    for criterion in ['gini', 'entropy', 'log_loss']:
        print (f"\n✍️ ========= {criterion.upper()} =========\n")
        res1 = train_nth_model(criterion, X_train, y_train, X_test, y_test)
        models_list.append(res1.get('model'))
        accs_list.append(res1.get('accuracy'))

    max_acc = max(accs_list)
    best_models = []
    for i in range(1, len(models_list)):
        if accs_list[i] == max_acc:
            best_models.append(i)
    
    rfc = models_list[best_models[0]]
    
    print ("\n✍️ ========= Results =========\n")

    if len(best_models) > 1:
        msg = 'Tie between:   '
        for idx in best_models:
            msg += f'rfc{idx+1} '
        
        print(msg)
        print('Accuracy:'.ljust(15) + str(max_acc))
        print(f'Picking rfc{best_models[0]+1} to continue')
    else:
        print(f"Best model: ".ljust(15) + f'{best_models[0]+1}')
        print('Accuracy:'.ljust(15) + str(max_acc))
    

    return {
            'model': rfc,
            'scaler': None
        }


if __name__ == '__main__':
    train_model()

