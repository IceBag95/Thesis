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
    df = pd.read_csv("../Dataset/raw_data.csv")

    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)   
    df.drop('ST slope', axis=1, inplace=True)
    df.drop(df[(df['target'] != 1) & (df['target'] != 0)].index, inplace=True)
    df.drop(df[df['cholesterol'] == 0].index, inplace=True)

    df['fasting blood sugar'] = df['fasting blood sugar'].astype(bool)
    df['exercise angina'] = df['exercise angina'].astype(bool)
    df['target'] = df['target'].astype(bool)

    dummy_df = pd.get_dummies(df['sex'])
    dummy_df.columns = ['Male', 'Female']
    df = df.drop('sex', axis=1)
    df = pd.concat([df[['age']], dummy_df[['Male']], df.drop(['age'], axis=1)], axis=1)

    df.drop(df[df.duplicated()].index, inplace=True)
    print(df.head())
    print(len(df))
    df.to_csv("../Dataset/clean_data.csv", index=False)


    columns = {"dataset_columns" : df.drop('target', axis=1).columns.tolist()}
    columns_file = open(Path.cwd().parent / "Back-end" / "Assets" / "columns.json" , 'w')
    json.dump(columns,columns_file)
    columns_file.close()
    #print(df)

    
    #------------------------------------------------------------------------------------------
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

    # from this plot we can observe that the data is quite mixed there are not good indicators
    # of target. Ideally we would like to see target being two quite separate groups when a
    # columns is selected as a base.
    
    #------------------------------------------------------------------------------------------
    
    # Outliers
    idx_list = []
    for col in df.columns:
        zscores = np.abs(stats.zscore(df[col]))
        # print(f'Col: { col }')
        temp = df[zscores > 3]
        current_idx_list = temp.index
        for idx in current_idx_list:
            if idx not in idx_list:
                idx_list.append(idx)
            else:
                print(f'{idx} already in idx_list')
        # print(temp[col])

    # print(idx_list)
    # print(len(idx_list))

    # Not the cleanest implementation of trying to find all the different entries that 
    # can be considered outliers. Don't need that for desision tree based models but it 
    # will be good for algorithms that have issues with outliers.
    #
    # 34 rows with outliers found. Due to them being not that significant for for desision tree based 
    # algorithm i'll keep them into that csv file / df. Also these outliers are valid there is no
    # reason to get rid of them.
    #

if __name__ == '__main__':
    setup_dataset()