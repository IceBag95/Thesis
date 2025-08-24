# import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, RocCurveDisplay
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
	plt.savefig('../Observations/Adaboost/plot_of_errors_adaboost_all_folds.png')
	plt.close()

	print(f'Stable Index: {stable_index}')
	return stable_index



def evaluate_with_cv(model:AdaBoostClassifier, X: pd.DataFrame, y: pd.Series, k=5)  -> pd.DataFrame:
    scoring = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']

    cv_results = cross_validate(model, X, y, cv=k, scoring=scoring)

    results = {
        'Metric': scoring,
        'Mean': [np.mean(cv_results[f'test_{m}']) for m in scoring],
        'Std Dev': [np.std(cv_results[f'test_{m}']) for m in scoring]
    }

    return pd.DataFrame(results)



def external_test(model:AdaBoostClassifier, scaler:StandardScaler=None, remove_outliers=False) -> pd.DataFrame:

	print("\n\n🎯Proceeding to external testing\n\n")

	setup.setup_external_dataset()

	df = pd.read_csv("../Dataset/External_dataset/cleveland_clean_data.csv")

	if remove_outliers == True:
		# Removing outliers for Logistic Regression
		print(f'\n⏳ Removing columns with outliers for Logistic Regression in external dataset')
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

	if remove_outliers == True:
		X = pd.DataFrame(scaler.transform(X), columns=X.columns)

	preds = model.predict(X)
	probs = model.predict_proba(X)[:, 1]

	results = {
		'Metric': ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1 Score'],
		'Value': [
			accuracy_score(y, preds),
			roc_auc_score(y, probs),
			precision_score(y, preds),
			recall_score(y, preds),
			f1_score(y, preds)
		]
	}

	results = pd.DataFrame(results)

	return results



def train_model() -> Dict[str, Union[AdaBoostClassifier, Optional[StandardScaler]]]:

	print('Entered adaboost training')

	ab_obsrv_path = Path.cwd().parent / "Observations" / "Adaboost"

	if not ab_obsrv_path.exists():
		ab_obsrv_path.mkdir(parents=True, exist_ok=True)

	logs = Path.cwd().parent / "Observations" / "Adaboost" / "Adaboost Metrics.txt"
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

	grid = GridSearchCV(ada_boost,
					 	param_grid=grid_params,
						verbose=3,
						cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=101))
	grid.fit(X_train,y_train, sample_weight=sample_weights)

	print(grid.best_params_)

	ada_boost_model = AdaBoostClassifier(estimator=base_learner, 
										n_estimators=grid.best_params_.get('n_estimators'), 
										random_state=101)
	
	cv_results = evaluate_with_cv(ada_boost_model, X_train, y_train)

	ada_boost_model.fit(X_train,y_train, sample_weight=sample_weights)
	preds = ada_boost_model.predict(X_test)

	external_testing_results = external_test(ada_boost_model)

	print('\n\n✍️ ======= AdaBoost model =======\n')
	logfile.write('\n\n✍️ ======= AdaBoost model =======\n\n')
	print(classification_report(y_test,preds))
	logfile.write('\n' + classification_report(y_test,preds) + '\n\n')
	print(f'Accuracy achieved: {accuracy_score(y_test, preds)}\n')
	logfile.write(f'Accuracy achieved: {accuracy_score(y_test, preds)}\n')
	print(f'ROC-AUC: {roc_auc_score(y_test, ada_boost_model.predict_proba(X_test)[:, 1])}\n')
	logfile.write(f'ROC-AUC: {roc_auc_score(y_test, ada_boost_model.predict_proba(X_test)[:, 1])}\n')
	print(f'\n✍️ Cross Validation Results for AdaBoost model:\n\n{cv_results}\n\n')
	logfile.write(f'\n✍️ Cross Validation Results for AdaBoost model:\n\n{cv_results}\n\n')
	print(f'\n\n🎯External testing Results:\n\n{external_testing_results}\n')
	logfile.write(f'\n\n🎯External testing Results :\n\n{external_testing_results}\n')
	logfile.close()

	# Make and save image for confusion matrix for this model
	filepath = Path.cwd().parent / "Observations" / "Adaboost" / f"Confusion_Matrix_AdaBoost"
	cm = confusion_matrix(y_test,preds)
    
	plt.figure(figsize=(6,5))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
	plt.title(f'Confusion Matrix\nAdaBoost')
	plt.xlabel('Predicted Class')
	plt.ylabel('Actual Class')
	plt.savefig(filepath)
	plt.close()

    # Make and save Roc Curve graph for this model
	filepath = Path.cwd().parent / "Observations" / "Adaboost" / f"Roc_Curve_AdaBoost"
	RocCurveDisplay.from_estimator(ada_boost_model, X_test, y_test)

	plt.savefig(filepath)
	plt.close()

	return {
			'model': ada_boost_model,
			'scaler': None
		}


if __name__ == '__main__':
	train_model()