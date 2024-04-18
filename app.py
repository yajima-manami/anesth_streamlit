#基本ライブラリ
from fastapi import FastAPI
app = FastAPI()
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
#目標値
df = pd.read_csv('./2024.4.16-2 train, val data general_anesth.csv')
#予測モデル構築
anesth = pd.DataFrame(df.values[0:],columns=df.iloc[0])
x = df.drop(labels=['target'], axis=1).values
y = df['target'].values
#xgboost
clf = XGBClassifier()
clf.fit(x,y)
# セレクトボックスのタイトルと選択肢のリストを定義
sex = st.sidebar.selectbox('Please select your gender.',('Male', 'Female'))
# 選択された性別を表示
st.sidebar.write('あなたは:', sex)
#サイドバー
st. sidebar. header('Input Features')
#sex = st.sidebar.checkbox('Male')
#sex = st.sidebar.checkbox('Female')
age = st.sidebar.slider('age', min_value=0, max_value = 99, step=1)
bmi = st.sidebar.slider('BMI', min_value=10.0, max_value = 50.0, step=0.1)
bun = st.sidebar.slider('BUN', min_value=1.0, max_value = 50.0, step=0.1)
gtp = st.sidebar.slider('GTP', min_value=5.0, max_value = 1000.0, step=0.1)
ast = st.sidebar.slider('AST', min_value=5.0, max_value = 100.0, step=0.1)
alt = st.sidebar.slider('ALT', min_value=5.0, max_value = 100.0, step=0.1)
cre = st.sidebar.slider('Cre', min_value=0.0, max_value=5.0, step=0.1)
st. sidebar. header('Please check if you have.')
asthma = st.sidebar.checkbox('Asthma (喘息)')
ht = st.sidebar.checkbox('HyperTension (高血圧)')
ponv = st.sidebar.checkbox('乗り物酔い')
#メインパネル
st. title('General_Anesth Classifier')
st. write('##Input Value')
#インプットデータ
#value_df = pd.DataFrame([], columns=['data', 'Sex', 'Age', 'BMI', 'BUN', 'GTP', 'AST', 'ALT', 'Cre', 'Asthma(喘息)', 'HyperTension(高血圧)', '乗り物酔い'])
#record = pd.Series(['data',sex, age, bmi, bun, gtp, ast, alt, cre, asthma, ht, ponv], index=value_df.columns)
#value_df=value_df.append(record, ignore_index=True)
#value_df.set_index('data', inplace=True)
#value_df.head()
#value_dfの各列が文字列（object型）なので数値型に変換する。
value_df = pd.DataFrame([], columns=['data', 'Sex', 'Age', 'BMI', 'BUN', 'GTP', 'AST', 'ALT', 'Cre', 'Asthma(喘息)', 'HyperTension(高血圧)', '乗り物酔い'])
record = pd.Series(['data', sex, age, bmi, bun, gtp, ast, alt, cre, asthma, ht, ponv], index=value_df.columns)
value_df = pd.concat([value_df, record.to_frame().T], ignore_index=True)
value_df.set_index('data', inplace=True)
# Sex列のデータ型を数値型に変換する関数
def convert_sex_to_numeric(sex):
    if sex.lower() == 'male':
        return 0
    elif sex.lower() == 'female':
        return 1
    else:
        return None  # その他の場合はNaNなどにする
# Sex列のデータ型を数値型に変換する
value_df['Sex'] = value_df['Sex'].apply(convert_sex_to_numeric)
# 列名を指定して空のDataFrameを作成
###columns = ['data', 'Sex', 'Age', 'BMI', 'BUN', 'GTP', 'AST', 'ALT', 'Cre', 'Asthma(喘息)', 'HyperTension(高血圧)', '乗り物酔い']
###value_df = pd.DataFrame(columns=columns)
# 数値型に変換可能な列のリストを指定
numeric_columns = ['Age', 'BMI', 'BUN', 'GTP', 'AST', 'ALT', 'Cre', 'Asthma(喘息)', 'HyperTension(高血圧)', '乗り物酔い']
# 指定した列のデータ型をfloat型に変換する
value_df[numeric_columns] = value_df[numeric_columns].astype(float)
#入力値の値
st.write(value_df)
#予測値のデータフレーム
pred_probs = clf.predict_proba(value_df)
pred_df=pd.DataFrame(pred_probs, columns=['Inhaled Anesth with Sevoflurare (rapid)', 'Inhaled Anesth with Sevoflurare (slow)','Inhaled Anesth with Desflurare (rapid)','TIVA with Propofol', 'TIVA with Remimazolam'], index=['probability'])
st.write('##Prediction')
st.write(pred_df)
#予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('##Result')
st.write('全身麻酔の方法として',str(name[0]),'を選択するのがよいでしょう。')