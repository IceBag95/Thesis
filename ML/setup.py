# https://data.mendeley.com/datasets/yrwd336rkz/2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def setup_dataset():
    df = pd.read_csv("/Users/panagiotispagonis/Documents/ΠΜΣ/ΕΡΓΑΣΙΑ ΔΕ/Heart Attack Dataset/Heart_Attack.csv")

    df.to_csv("./Dataset/raw_data.csv")

    #-----------------------------------------------------------------------------------------------
    # I should probably get rid of the Slope column since the values represented there are at best 
    # questionable. According to the dataset descreption there should be 2 categpries of Slope but
    # there are four. Them being 4 makes no sense, since according to my reaserch online they
    # are 3 categories of slopes flat upgoing downgoing.
    # For certainty Ι'll have to ask the proffessor so I do not make a mess out of the dataset.
    #
    df = df.drop('ST slope', axis=1)
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
    df.to_csv("./Dataset/clean_data.csv", index=False)

    print(df)

    #
    # ------------------------------------------------------------------------------------------
    #
    # pivot_df = pd.pivot_table(df, values='target', index='age', aggfunc='count')
    # print(pivot_df)

    # plt.scatter(pivot_df.index, pivot_df['target'])
    # plt.show()

    correlation_matrix = df.corr()

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.show()

    # from the plot it seems that there is not much linear correlation so linear regression 
    # is not an option. Also the correlation table never exeeds 0.4 for any value when correlated
    # to target so nothing indicates a linear relationship between this metrics
    #
    # ------------------------------------------------------------------------------------------
    #
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

    print(idx_list)
    print(len(idx_list))

    # Not the cleanest implementation of trying to find all the different entries that 
    # can be considered outliers. Don't need that for random forest but it will be good 
    # for algorithms that have issues with outliers.
    #
    # 34 rows with outliers found. Due to them being not that significant for random forest 
    # algorithm i'll keep them into that csv file / df. Also these outliers are valid there is no
    # reason to get rid of them.
    #
