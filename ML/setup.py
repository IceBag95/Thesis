# https://data.mendeley.com/datasets/yrwd336rkz/2


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from  pathlib import Path

def setup_dataset():
    print('\n\n⏳ Loading the original dataset...')
    df = pd.read_csv("../Dataset/raw_data.csv")
    print('\n✅ Load SUCCESS\n')

    print('\n⏳ Removing corrupted or dublicate data...')
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)   
    df.drop('ST slope', axis=1, inplace=True)
    df.drop(df[(df['target'] != 1) & (df['target'] != 0)].index, inplace=True)
    df.drop(df[df['cholesterol'] == 0].index, inplace=True)
    df.drop(df[df.duplicated()].index, inplace=True)
    print('\n✅ Removal SUCCESS\n')

    print('\n⏳ Adjusting data to be better suited for training...')
    df['fasting blood sugar'] = df['fasting blood sugar'].astype(bool)
    df['exercise angina'] = df['exercise angina'].astype(bool)
    df['target'] = df['target'].astype(bool)

    dummy_df = pd.get_dummies(df['sex'])
    dummy_df.columns = ['Male', 'Female']
    df = df.drop('sex', axis=1)
    df = pd.concat([df[['age']], dummy_df[['Male']], df.drop(['age'], axis=1)], axis=1)
    print('\n✅ Clean dataset READY\n\n>> Please check the format of the entries (first 5 as example)')
    print(df.head())
    print(f'>> Clean dataset length: {len(df)} enties\n')
    print(f'\n‼️ Outliers kept! Depending on the model, they will be removed before training \n')
    print('\n⏳ Proceding to store the clean dataset...')
    df.to_csv("../Dataset/clean_data.csv", index=False)
    print('\n✅ Clean Dataset SAVED\n')

    print('\n⏳ Storing the column names of the dataset in order...')
    columns = {"dataset_columns" : df.drop('target', axis=1).columns.tolist()}
    columns_file = open(Path.cwd().parent / "Back-end" / "Assets" / "columns.json" , 'w')
    json.dump(columns,columns_file)
    columns_file.close()
    print('\n✅ Store SUCCESS\n')
    #print(df)

    
    #------------------------------------------------------------------------------------------
    print('\n⏳ Creating images of data relationships to review...')
    observations_path = Path.cwd().parent / "Dataset" / "Observations"
    if not observations_path.exists():
        observations_path.mkdir(parents=True, exist_ok=True)
        

    correlation_matrix = df.corr()

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.savefig('../Dataset/Observations/heatmap_of_dataset.png')

    # from the plot it seems that there is not much linear correlation so linear regression 
    # is not an option. Also the correlation table never exeeds 0.4 for any value when correlated
    # to target so nothing indicates a linear relationship between this metrics. 
    # We will try it out though just to make sure
    
    # Visualize the correlation matrix
    sns.pairplot(df, hue='target')
    plt.savefig('../Dataset/Observations/pairplot_of_dataset.png')
    print('\n✅ Images CREATED\n')

if __name__ == '__main__':
    setup_dataset()