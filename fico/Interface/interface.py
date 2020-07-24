import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Risk Prediction')
dic = {0: 'Bad', 1: 'Good'}

data= pd.read_csv('heloc_dataset_v1.csv')
X=data.iloc[:,1:]
n=X.shape[1]
input=pd.DataFrame()

for i in range(n):
    name= X.columns[i]
    max=X.iloc[:,i].max()
    header = name + ': (You can pick a number from -8, -7 and any integer from 0 to ' + str(max)+ ')'
    num = st.number_input(header,step=1)
    input[name] = [num]

impute = pickle.load(open('imputer_mode.sav', 'rb'))
encoder_MaxDelqEver = pickle.load(open('OneHot_MaxDelqEver.sav', 'rb'))
encoder_MaxDelq2 = pickle.load(open('OneHot_MaxDelq2.sav', 'rb'))
bestmodel = pickle.load(open('best_model_rf.sav','rb'))

input_imputed = pd.DataFrame(impute.transform(input), columns=input.columns, index=input.index)
input = input_imputed.copy(deep=True)
MaxDelq2 = encoder_MaxDelq2.transform(input['MaxDelq2PublicRecLast12M'].values.reshape(-1, 1)).toarray()
MaxDelqEver = encoder_MaxDelqEver.transform(input['MaxDelqEver'].values.reshape(-1,1)).toarray()

MaxDelq2_Cols = []
for i in range(0, 9):
    col = 'MaxDelq2PublicRecLast12M_' + str(i)
    MaxDelq2_Cols.append(col)
MaxDelq2_df = pd.DataFrame(MaxDelq2, columns=MaxDelq2_Cols)
MaxDelq2_df.drop('MaxDelq2PublicRecLast12M_0', axis=1, inplace=True)

MaxDelqEver_Cols = []
for i in range(2, 9):
    col = 'MaxDelqEver_' + str(i)
    MaxDelqEver_Cols.append(col)
MaxDelqEver_df = pd.DataFrame(MaxDelqEver, columns=MaxDelqEver_Cols)
MaxDelqEver_df.drop('MaxDelqEver_2', axis=1, inplace=True)

MaxDelq2_df.insert(5,'MaxDelq2PublicRecLast12M_5_6', MaxDelq2_df.iloc[:, 6] + MaxDelq2_df.iloc[:, 7])
MaxDelq2_df.drop(['MaxDelq2PublicRecLast12M_5', 'MaxDelq2PublicRecLast12M_6'], axis=1, inplace=True)

input.drop(['MaxDelq2PublicRecLast12M', 'MaxDelqEver'], axis=1, inplace=True)
input_clean = pd.concat([input, MaxDelq2_df, MaxDelqEver_df], axis=1)


if st.button('Run Model'):
    res = bestmodel.predict(input_clean)
    st.write('Prediction:  ', dic[res[0]])
    prob= bestmodel.predict_proba(input_clean)
    st.write('The probability of default is',prob[0][0]) 
    # important= bestmodel.feature_importances_()
    # st.write('The importance of features is',important)

st.write('Below is the feature importance:')
st.text('''
Variable: ExternalRiskEstimate                     Importance: 0.32
Variable: NetFractionRevolvingBurden               Importance: 0.12
Variable: AverageMInFile                           Importance: 0.09
Variable: PercentTradesNeverDelq                   Importance: 0.06
Variable: NumBank2NatlTradesWHighUtilization       Importance: 0.06
Variable: MSinceOldestTradeOpen                    Importance: 0.05
Variable: MSinceMostRecentInqexcl7days             Importance: 0.04
Variable: PercentTradesWBalance                    Importance: 0.04
Variable: MaxDelqEver_8                            Importance: 0.04
Variable: NumSatisfactoryTrades                    Importance: 0.03
Variable: NumTrades60Ever2DerogPubRec              Importance: 0.03
Variable: MSinceMostRecentDelq                     Importance: 0.02
Variable: NumTotalTrades                           Importance: 0.02
Variable: MaxDelq2PublicRecLast12M_7               Importance: 0.02
Variable: PercentInstallTrades                     Importance: 0.01
Variable: NumInqLast6M                             Importance: 0.01
Variable: NetFractionInstallBurden                 Importance: 0.01
Variable: NumRevolvingTradesWBalance               Importance: 0.01
Variable: MaxDelq2PublicRecLast12M_4               Importance: 0.01
Variable: MSinceMostRecentTradeOpen                Importance: 0.0
Variable: NumTrades90Ever2DerogPubRec              Importance: 0.0
Variable: NumTradesOpeninLast12M                   Importance: 0.0
Variable: NumInqLast6Mexcl7days                    Importance: 0.0
Variable: NumInstallTradesWBalance                 Importance: 0.0
Variable: MaxDelq2PublicRecLast12M_1               Importance: 0.0
Variable: MaxDelq2PublicRecLast12M_2               Importance: 0.0
Variable: MaxDelq2PublicRecLast12M_3               Importance: 0.0
Variable: MaxDelq2PublicRecLast12M_5_6             Importance: 0.0
Variable: MaxDelq2PublicRecLast12M_8               Importance: 0.0
Variable: MaxDelqEver_3                            Importance: 0.0
Variable: MaxDelqEver_4                            Importance: 0.0
Variable: MaxDelqEver_5                            Importance: 0.0
Variable: MaxDelqEver_6                            Importance: 0.0
Variable: MaxDelqEver_7                            Importance: 0.0''')

note1=pd.read_excel('8f3c89894ce48371.xlsx', sheetname='Data Dictionary')
st.write('Below is a Data Dictionary:')
st.table(note1.iloc[:24,:])

note2=pd.read_excel('8f3c89894ce48371.xlsx', sheetname='Max Delq')
st.write('Below is a Categorical Variable Explanation:')
st.write('(1)MaxDelq2PublicRecLast12M')
st.table(note2)

note3=pd.read_excel('8f3c89894ce48371.xlsx', sheetname='Sheet1')
st.write('(2)MaxDelqEver')
st.table(note3)

note4=pd.read_excel('8f3c89894ce48371.xlsx', sheetname='SpecialValues')
st.write('Below is a Special Values Explanation:')
st.table(note4)

