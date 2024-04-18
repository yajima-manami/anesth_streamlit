from fastapi import FastAPI
app = FastAPI()
from pydantic import BaseModel
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
#学習済みモデルの読み込み
model = pickle.load(open('general_anesth', 'rb'))
#入力するデータの型の定義
class general_anesth(BaseModel):
    age:float
    sex:float
    bmi:float
    asthma:float
    ht:float
    ponv:float
    bun:float
    gtp:float
    ast:float
    alt:float
    cre:float
#トップページ
@app.get('/')
async def index():
    return {"Anesth":'general_anesth_prediction'}
#POSTが送信されたとき（入力)と予測値(出力)の定義
@app.post('/make_predictions')
async def make_predictions(features: general_anesth):
    return({'prediction':str(model.predict([[features.age, features.sex, features.bmi, features.asthma, features.ht, features.ponv, features.bun, features.gtp, features.ast, features.alt, features.cre]])[0])})
