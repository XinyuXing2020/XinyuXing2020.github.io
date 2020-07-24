import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import os
import pickle

raw = pd.read_csv('heloc_dataset_v1.csv', na_values = [-9])
# raw = pd.read_csv('heloc_dataset_v1.csv')

#### transform target
raw.loc[raw['RiskPerformance'] == 'Good', 'RiskPerformance'] = 1
raw.loc[raw['RiskPerformance'] == 'Bad', 'RiskPerformance'] = 0

#### transforming data
# see = raw[raw.isnull().any(axis=1)]

data = raw.copy(deep=True)
data.dropna(axis=0, inplace=True)
data.reset_index(drop=True, inplace=True)
imputer_mode = SimpleImputer(missing_values=-8, strategy='most_frequent')
data_y = data.pop('RiskPerformance')
data_X_impute = pd.DataFrame(imputer_mode.fit_transform(data), columns=data.columns, index=data.index)
data_X_impute.insert(0, 'RiskPerformance', data_y)
data = data_X_impute.copy(deep=True)

# data.loc[data['MSinceMostRecentDelq'] == -7, 'MSinceMostRecentDelq'] = max(data['MSinceMostRecentDelq'])
# data.loc[data['MSinceMostRecentInqexcl7days'] == -7, 'MSinceMostRecentInqexcl7days'] = max(data['MSinceMostRecentInqexcl7days'])
data.replace(-7, 0, inplace=True)

#### build dummies for cat
cat_var = data.loc[:, ['MaxDelq2PublicRecLast12M', 'MaxDelqEver']]
data.drop(['MaxDelq2PublicRecLast12M', 'MaxDelqEver'], axis=1, inplace=True)

encoder_MaxDelq2 = OneHotEncoder()
MaxDelq2 = encoder_MaxDelq2.fit_transform(cat_var.iloc[:, 0].values.reshape(-1, 1)).toarray()
MaxDelq2_Cols = []
for i in range(0, 9):
    col = 'MaxDelq2PublicRecLast12M_' + str(i)
    MaxDelq2_Cols.append(col)
MaxDelq2_df = pd.DataFrame(MaxDelq2, columns=MaxDelq2_Cols)
MaxDelq2_df.drop('MaxDelq2PublicRecLast12M_0', axis=1, inplace=True)

encoder_MaxDelqEver = OneHotEncoder()
MaxDelqEver = encoder_MaxDelqEver.fit_transform(cat_var.iloc[:, 1].values.reshape(-1, 1)).toarray()
MaxDelqEver_Cols = []
for i in range(2, 9):
    col = 'MaxDelqEver_' + str(i)
    MaxDelqEver_Cols.append(col)
MaxDelqEver_df = pd.DataFrame(MaxDelqEver, columns=MaxDelqEver_Cols)
MaxDelqEver_df.drop('MaxDelqEver_2', axis=1, inplace=True)

dummy_var = pd.get_dummies(cat_var, columns=['MaxDelq2PublicRecLast12M', 'MaxDelqEver'])
# dummy_var.insert(5,'MaxDelq2PublicRecLast12M_5_6', dummy_var.iloc[:, 6] + dummy_var.iloc[:, 7])
# dummy_var.drop(['MaxDelq2PublicRecLast12M_5.0', 'MaxDelq2PublicRecLast12M_6.0'], axis=1, inplace=True)
MaxDelq2_df.insert(5,'MaxDelq2PublicRecLast12M_5_6', MaxDelq2_df.iloc[:, 6] + MaxDelq2_df.iloc[:, 7])
MaxDelq2_df.drop(['MaxDelq2PublicRecLast12M_5', 'MaxDelq2PublicRecLast12M_6'], axis=1, inplace=True)

data_final = pd.concat([data, MaxDelq2_df, MaxDelqEver_df], axis=1)
data_final.to_csv('data_set_cleaned_v4.csv', index=False)

pickle.dump(imputer_mode, open('imputer_mode.sav', 'wb'))
pickle.dump(encoder_MaxDelq2, open('OneHot_MaxDelq2.sav', 'wb'))
pickle.dump(encoder_MaxDelqEver, open('OneHot_MaxDelqEver.sav', 'wb'))