from typing import Dict, Optional, Union
import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from  pathlib import Path
import setup



def train_model() -> Dict[str, Union[LogisticRegressionCV, Optional[StandardScaler]]]:

    print('Entered LogisticRegressionCV training')

    model_name = 'Logistic Regression'
    
    setup.setup_dataset()

    models_metrics = {
        'model': [],
        'cls_rprt': [],
        'accuracy': [],
        'roc_auc': []
    }

    logs = Path.cwd().parent / "Dataset" / "Observations" / "Logistic Regression Metrics.txt"
    logs.touch()
    
    logs = open(logs, 'w')

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

    lr1 = LogisticRegressionCV(cv=10,
                               Cs=50,
                               solver='saga',
                               penalty='l1',
                               max_iter=1000,
                               random_state=101,
                               class_weight='balanced',
                               verbose=3)
    lr1.fit(X_train, y_train)
    preds = lr1.predict(X_test)
    models_metrics.get('model').append(lr1)
    cls_rprt_1 = classification_report(y_test,preds)
    accuracy1 = accuracy_score(y_test,preds)
    roc_auc_1 = roc_auc_score(y_test,preds)
    models_metrics.get('cls_rprt').append(cls_rprt_1)
    models_metrics.get('accuracy').append(accuracy1)
    models_metrics.get('roc_auc').append(roc_auc_1)
    print('\n\n✍️ ======= LogisticRegressionCV with penalty: l1')
    print(cls_rprt_1)
    print(f'Precise accuracy {accuracy1}')
    print(f'Roc-Auc: {roc_auc_1}')
    logs.write('\n\n======= LogisticRegressionCV with penalty: l1 =======\n')
    logs.write(cls_rprt_1)
    logs.write(f'\nPrecise accuracy {accuracy1}\n')
    logs.write(f'Roc-Auc: {roc_auc_1}\n')
    logs.write('\n')

    lr2 = LogisticRegressionCV(cv=10, 
                               Cs=50,
                               solver='saga',
                               penalty='l2',
                               max_iter=1000,
                               random_state=101,
                               class_weight='balanced',
                               verbose=3)
    lr2.fit(X_train, y_train)
    preds = lr2.predict(X_test)
    models_metrics.get('model').append(lr2)
    cls_rprt_2 = classification_report(y_test,preds)
    accuracy2 = accuracy_score(y_test,preds)
    roc_auc_2 = roc_auc_score(y_test,preds)
    models_metrics.get('cls_rprt').append(cls_rprt_2)
    models_metrics.get('accuracy').append(accuracy2)
    models_metrics.get('roc_auc').append(roc_auc_2)
    print('\n\n✍️ ======= LogisticRegressionCV with penalty: l2 =======\n')
    print(cls_rprt_2)
    print(f'Precise accuracy {accuracy2}')
    print(f'Roc-Auc: {roc_auc_2}')
    logs.write('\n\n======= LogisticRegressionCV with penalty: l2 =======\n')
    logs.write(cls_rprt_2)
    logs.write(f'\nPrecise accuracy {accuracy2}\n')
    logs.write(f'Roc-Auc: {roc_auc_2}\n')
    logs.write('\n')

    lr3 = LogisticRegressionCV(cv=10,
                               Cs=100, 
                               solver='saga', 
                               penalty='elasticnet', 
                               max_iter=1000, 
                               l1_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                               random_state=101,
                               class_weight='balanced',
                               verbose=3)
    lr3.fit(X_train, y_train)
    preds = lr3.predict(X_test)
    models_metrics.get('model').append(lr3)
    cls_rprt_3 = classification_report(y_test,preds)
    accuracy3 = accuracy_score(y_test,preds)
    roc_auc_3 = roc_auc_score(y_test,preds)
    models_metrics.get('cls_rprt').append(cls_rprt_3)
    models_metrics.get('accuracy').append(accuracy3)
    models_metrics.get('roc_auc').append(roc_auc_3)
    print('\n\n✍️ ======= LogisticRegressionCV with penalty: elasticnet =======')
    print(cls_rprt_3)
    print(f'Precise accuracy {accuracy3}')
    print(f'Roc-Auc: {roc_auc_3}')
    logs.write('\n\n======= LogisticRegressionCV with penalty: elasticnet =======\n')
    logs.write(cls_rprt_3)
    logs.write(f'\nPrecise accuracy {accuracy3}\n')
    logs.write(f'Roc-Auc: {roc_auc_3}\n')
    logs.write('\n')
    
    best_accuracy = max(models_metrics.get('accuracy'))

    # Check if there are more than one best accuracies to log it
    best_accuracy_idxs = []
    accs = models_metrics.get('accuracy')
    for idx in range(len(accs)):
        if accs[idx] == best_accuracy:
            best_accuracy_idxs.append(idx)

    print('\n\n✍️\t ======= Best Accuracy =======\n')
    print(f'All model Accuracies:'.ljust(24) + f'{accs}')
    print('Best Accuracy:'.ljust(24) + f'{best_accuracy}')
    logs.write('\n\n\t======= Best Accuracy =======\n')
    logs.write('All model Accuracies:'.ljust(24) + f'{accs}\n')
    logs.write(f'Best Accuracy:'.ljust(24) + f'{best_accuracy}\n')

    best_accuracy_idx = models_metrics.get('accuracy').index(best_accuracy)
    selected_model = models_metrics.get('model')[best_accuracy_idx]

    # If multiple best model found, create a message to display them
    if len(best_accuracy_idxs) > 1:
        msg = 'Tie between models: ['
        for idx in best_accuracy_idxs[:-1]:
            msg += f'lr{idx+1}, '
        msg += f'lr{best_accuracy_idxs[-1] + 1}]'

        print('Achieved by Models:'.ljust(24) + f'{ msg[msg.index('[') + 1 : -1] }')
        logs.write('Achieved by Models:'.ljust(24) + f'{ msg[msg.index('[') + 1 : -1] }\n')
    else:
        print('Achieved by Models:'.ljust(24) + f'lr{best_accuracy_idx+1}\n')
        logs.write('Achieved by Models:'.ljust(24) + f'lr{best_accuracy_idx+1}\n')
    
    print('\n\n✍️\t ======= Selected Best Model =======\n')
    if len(best_accuracy_idxs) > 1:
        print(f'\n{msg}\n')
    print(f'✅\tSelected best model: lr{best_accuracy_idx+1}')
    print(f'{models_metrics.get('cls_rprt')[best_accuracy_idx]}')
    print(f'\nPrecise Accuracy: {best_accuracy}')
    print(f'ROC_AUC: {models_metrics.get('roc_auc')[best_accuracy_idx]}\n')
    logs.write('\n\n\t======= Selected Best Model =======\n')
    if len(best_accuracy_idxs) > 1:
        logs.write(f'\n{msg}\n\n')
    logs.write(f'Selected best model: lr{best_accuracy_idx+1}\n')
    logs.write(f'{models_metrics.get('cls_rprt')[best_accuracy_idx]}')
    logs.write(f'\nPrecise Accuracy: {best_accuracy}\n')
    logs.write(f'ROC_AUC: {models_metrics.get('roc_auc')[best_accuracy_idx]}')

    logs.close()

    
    return {
            'model': selected_model,
            'scaler': scaler
        }

if __name__ == '__main__':
    train_model()
