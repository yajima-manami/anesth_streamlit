import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split                                        
df = pd.read_csv('./2024.4.10 train, val data 全身麻酔方法の選択.csv')
df.head()
df_t = pd.read_csv('./2024.4.10 test data 全身麻酔方法の選択.csv')
df_t.head()
x = df.drop('target', axis=1).values
t = df['target'].values
print(x.shape, t.shape)
x_test = df_t.drop('target', axis=1).values
t_test = df_t['target'].values
print(x_test.shape, t_test.shape)
from sklearn.model_selection import train_test_split
x_train, x_val, t_train, t_val = train_test_split(x, t, test_size=0.3, random_state=0)
#from sklearn.tree import DecisionTreeClassifier
#dtree = DecisionTreeClassifier(random_state=0)
#dtree.fit(x_train, t_train)
#print('train score : ', dtree.score(x_train, t_train))
#print('val score :', dtree.score(x_val, t_val))
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit (x_train, t_train)
print('train score : ', model.score(x_train, t_train))
print('val score : ', model.score(x_val, t_val))
#print('train score : ', dtree.score(x_train, t_train))
#print('test score : ', dtree.score(x_test, t_test))
#dtree_m = DecisionTreeClassifier(max_depth=7, min_samples_split=15, random_state=0)
#dtree_m.fit(x_train, t_train)
#print('train score:', dtree_m.score(x_train, t_train))
#print('validation score:', dtree_m.score(x_val, t_val))
#print('test score:', dtree_m.score(x_test, t_test))
import pickle
pickle.dump(model, open('general_anesth', 'wb'))
