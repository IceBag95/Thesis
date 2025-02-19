from sklearn.ensemble import StackingClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import train_with_random_forest
import train_with_adaboost
import setup



def train_model():
    
    setup.setup_dataset()

    df = pd.read_csv("./Dataset/clean_data.csv")

    X = pd.get_dummies(df.drop('target',axis=1),drop_first=True)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    print('Calling adaboost training')
    base_model_1 = train_with_adaboost.train_model()
    print('Finished adaboost training')

    print('Calling random forest training')
    base_model_2 = train_with_random_forest.train_model()
    print('Finished random forest training')

    meta_model = XGBClassifier()

    stacked_model = StackingClassifier(
        estimators=[('ada_boost', base_model_1), ('random_forest', base_model_2)],
        final_estimator=meta_model
    )

    grid_params = {
        "final_estimator__n_estimators": list(range(50,500,50))
    }

    grid_search = GridSearchCV(estimator=stacked_model, param_grid=grid_params, verbose=3)
    grid_search.fit(X_train,y_train)

    stacked_model = grid_search.best_estimator_
    stacked_model.fit(X_train,y_train)
    stacked_model_preds = stacked_model.predict(X_test)
    accur = accuracy_score(y_test,stacked_model_preds)
    print(f'Stackined model accuracy: {accur:.4f}')

    

    return stacked_model

# During full run this should be removed
train_model()
