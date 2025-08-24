# https://data.mendeley.com/datasets/yrwd336rkz/2


import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  pathlib import Path

def setup_dataset() -> None:
    print('\n\n⏳ Loading the original dataset...')
    df = pd.read_csv("../Dataset/raw_data.csv")
    print('\n✅ Load SUCCESS\n')

    print('\n⏳ Removing corrupted or dublicate data...')
    df = df.dropna()
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)   
    df.drop(df[(df['target'] != 1) & (df['target'] != 0)].index, inplace=True)
    df.drop(df[df['cholesterol'] == 0].index, inplace=True)
    df = df[df['ST slope'] <= 2]
    df.drop(df[df.duplicated()].index, inplace=True)
    print('\n✅ Removal SUCCESS\n')
    df = df[~(df < 0).any(axis=1)]

    print('\n⏳ Adjusting data to be better suited for training...')
    df['fasting blood sugar'] = df['fasting blood sugar'].astype(bool)
    df['exercise angina'] = df['exercise angina'].astype(bool)
    df['Chest pain type'] = df['Chest pain type'] - 1
    df['target'] = df['target'].astype(bool)

    df = df.rename(columns={"sex": "Male"})
    df["Male"] = df["Male"].map({1.0: True, 0.0: False})
    print('\n👀Trues and Falses in column Target:')
    df2 = df['target'].value_counts()
    df2_normaliazed = df['target'].value_counts(normalize=True)
    print(f'True: {df2[True]} -> {df2_normaliazed[True]:.2f}%')
    print(f'False: {df2[False]} -> {df2_normaliazed[False]:.2f}%')
    df_len = len(df)
    if 30 <= df2[True]/df_len*100 <= 70 and 30 <= df2[False]/df_len*100 <= 70:
        print('\nTarget values seem  balanced, no need to oversample')
    else:
        print("\n‼️ Proceeding with IMBALANCED Target values! Results may be better if oversampling techniques are applied")
    print('\n✅ Clean dataset READY\n\n>> Please check the format of the entries (first 5 as example)')
    print(df.head())
    print(f'>> Clean dataset length: {len(df)} entries\n')
    print(f'\n‼️ Outliers kept! Depending on the model, they will be removed before training \n')
    print('\n⏳ Proceding to store the clean dataset...')
    df.to_csv("../Dataset/clean_data.csv", index=False)
    print('\n✅ Clean Dataset SAVED\n')


    
    #------------------------------------------------------------------------------------------
    print('\n⏳ Creating images of data relationships to review...')
    observations_path = Path.cwd().parent / "Observations"
    if not observations_path.exists():
        observations_path.mkdir(parents=True, exist_ok=True)
        

    correlation_matrix = df.corr()

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.savefig('../Observations/heatmap_of_dataset.png')
    plt.close()

    # from the plot it seems that there is not much linear correlation so linear regression 
    # is not an option. Also the correlation table never exeeds 0.4 for any value when correlated
    # to target so nothing indicates a linear relationship between this metrics. 
    # We will try it out though just to make sure
    
    # Visualize the correlation matrix
    sns.pairplot(df, hue='target')
    plt.savefig('../Observations/pairplot_of_dataset.png')
    print('\n✅ Images CREATED\n')
    plt.close()


def setup_external_dataset():
    print('\n\n⏳ Loading external dataset...')
    df = pd.read_csv("../Dataset/External_dataset/cleveland_raw_data.csv")
    print('\n✅ Load SUCCESS\n')

    print('Matching columns between external and original dataset')
    naming_map_json = open(Path.cwd().parent / "Dataset" / "External_dataset" / "name_mapping.json")
    naming_map:dict = json.load(naming_map_json)
    naming_map_json.close()
    columns_to_drop = []

    for key, value in naming_map.items():
        if value:
            df.rename(columns={key: value}, inplace=True)
        else:
            columns_to_drop.append(key)
    
    df = df.drop(columns=columns_to_drop)

    # Tryied external Validation with uci dataset. These are the changes needed to be applied so that the uci Dataset matches the original.
    # Also you will need to change the name_mapping.json and un-commenting these lines. 
    # print('\n⏳ Matching the way data are register between the original and the external dataset...')
    # df['sex'] = df['sex'].map({'Male': 1.0, 'Female': 0.0})
    # df['Chest pain type'] = df['Chest pain type'].map({"asymptomatic": 0, "atypical angina": 1, "typical angina": 2, "non-anginal": 3})
    # df['resting ecg'] = df['resting ecg'].map({'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2}) 
    # df['ST slope'] = df['ST slope'].map({'upsloping': 0, 'flat': 1, 'downsloping': 2})
    

    print('\n⏳ Removing corrupted, matching and dublicate data...')
    df = df.dropna()
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.drop(df[(df['target'] != 1) & (df['target'] != 0)].index, inplace=True)
    df.drop(df[df['cholesterol'] == 0].index, inplace=True)
    df = df[df['ST slope'] <= 2]
    df.drop(df[df.duplicated()].index, inplace=True)


    # Since this is a proof of concept these lines will be commented out but they are necessary in real life replication
    # Finding common rows
    # original_df = pd.read_csv("../Dataset/raw_data.csv")
    # common_df = df[df.isin(original_df.to_dict(orient='list')).all(axis=1)]
    # print('Matcing rows:\n', common_df)

    # # Removing common rows from external test
    # df = pd.concat([df, common_df]).drop_duplicates(keep=False)
    # print('df with removed duplicates:\n', df)

    print('\n✅ Removal SUCCESS\n')

    print('\n⏳ Adjusting data to be better suited for training...')
    df['target'] = df['target'].astype(bool)                           # Reminder: In the case of uci, any non 0 value will become True by default, so nw for categories 1-4
    df = df[~(df < 0).any(axis=1)]

    df = df.rename(columns={"sex": "Male"})
    df["Male"] = df["Male"].map({1.0: True, 0.0: False})
    print('\n👀Trues and Falses in column Target:')
    df2 = df['target'].value_counts()
    df2_normaliazed = df['target'].value_counts(normalize=True)
    print(f'True: {df2[True]} -> {df2_normaliazed[True]:.2f}%')
    print(f'False: {df2[False]} -> {df2_normaliazed[False]:.2f}%')
    df_len = len(df)
    if 30 <= df2[True]/df_len*100 <= 70 and 30 <= df2[False]/df_len*100 <= 70:
        print('\nTarget values seem  balanced, no need to oversample')
    else:
        print("\n‼️ Proceeding with IMBALANCED Target values! Results may be better if oversampling techniques are applied")
    print('\n✅ Clean dataset READY\n\n>> Please check the format of the entries (first 5 as example)')
    print(df.head())
    print(f'>> Clean dataset length: {len(df)} entries\n')
    print(f'\n‼️ Outliers kept! Depending on the model, they will be removed before training \n')
    print('\n⏳ Proceding to store the clean dataset...')
    df.to_csv("../Dataset/External_dataset/cleveland_clean_data.csv", index=False)
    print('\n✅ Clean Dataset SAVED\n')


    
    #------------------------------------------------------------------------------------------
    print('\n⏳ Creating images of data relationships to review...')
    observations_path = Path.cwd().parent / "Observations"
    if not observations_path.exists():
        observations_path.mkdir(parents=True, exist_ok=True)
        

    correlation_matrix = df.corr()

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.savefig('../Observations/heatmap_of_external_dataset.png')
    plt.close()
    
    # Visualize the correlation matrix
    sns.pairplot(df, hue='target')
    plt.savefig('../Observations/pairplot_of_external_dataset.png')
    print('\n✅ Images CREATED\n')
    plt.close()

if __name__ == '__main__':
    setup_dataset()
    setup_external_dataset()