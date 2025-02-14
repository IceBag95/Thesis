import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import setup

setup.setup_dataset()

df = pd.read_csv("./Dataset/clean_data.csv")

X = pd.get_dummies(df.drop('target',axis=1),drop_first=True)

y = df['target']

print(y.head())

