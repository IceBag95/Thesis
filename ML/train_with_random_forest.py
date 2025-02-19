import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay, accuracy_score
import setup

# In case we need it
def plot_errors_for_n_estimators(max_featues, bootstrap):
    errors = []
    missclassifications = []

    # Plotted the whole thing, seems to start smothing out at arround the 200.
    # Found the min missclassifications at exactly the 200 mark. So we will go with that.
    for n in range(100,500):
        rfc = RandomForestClassifier(n_estimators=n, max_features=max_featues, bootstrap=bootstrap, random_state=101)
        rfc.fit(X_train, y_train)
        preds = rfc.predict(X_test)
        err =  1 - accuracy_score(y_test,preds)
        n_missed = np.sum( preds != y_test)
        
        errors.append(err)
        missclassifications.append(n_missed)

    # print(f'Min missclassifications {min(missclassifications)} at index {missclassifications.index(min(missclassifications))}')
    plt.plot(range(100,500),errors)
    plt.show()

    # return missclassifications.index(min(missclassifications))



setup.setup_dataset()

df = pd.read_csv("./Dataset/clean_data.csv")

X = pd.get_dummies(df.drop('target',axis=1),drop_first=True)

y = df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# To avoid recreating completely random X-Y splits when testing against different models 
# it's important to assign a random state so the comparison is fair  

rfc = RandomForestClassifier(random_state=101)

# Do a grid search for our params. Because the machine could handle everything I did the search for all estimators from 10 to 120 n_estimators
grid_params = {
    'n_estimators': list(range(100,501)),
    'max_depth'   : [4,5, None],
    'bootstrap'   : [True, False]
}

grid = GridSearchCV(rfc ,param_grid=grid_params, verbose=3)
grid.fit(X_train,y_train)

print(grid.best_params_)

best_max_depth = grid.best_params_.get('max_depth')
best_use_bootstrap = grid.best_params_.get('bootstrap')
best_n_setimators = grid.best_params_.get('n_estimators')
# plot_errors_for_n_estimators(best_max_depth, best_use_bootstrap)

rfc = RandomForestClassifier(n_estimators=best_n_setimators, 
                             max_depth=best_max_depth,
                             bootstrap=best_use_bootstrap,
                             random_state=101)


rfc.fit(X_train,y_train)

preds = rfc.predict(X_test)

cm = confusion_matrix(y_test, preds, labels=rfc.classes_)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)

# disp.plot()

# plt.show()

print(classification_report(y_test,preds))

joblib.dump(rfc, '../Back-end/ml_model/RandomForestModel.pkl')


