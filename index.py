from typing import Union

from fastapi import FastAPI, File,UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from sklearn.preprocessing import StandardScaler

from keras.models import load_model
from fastapi.encoders import jsonable_encoder
from typing import Union

import pandas as pd
import numpy as np



# returns a compiled model
# identical to the previous one
model = load_model('model/lstmclass.h5')




import tensorflow as tf

app = FastAPI()


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(x: pd.DataFrame) -> np.ndarray:
        scaler = StandardScaler().fit(x)
        x_scaled = scaler.transform(x)
        x_train = np.reshape(x_scaled, (x_scaled.shape[0], 1, x_scaled.shape[1]))
        
        return x_train,x
    
class ChurnData(BaseModel):
    cid: Union[str,int]
    data: list[Union[float,int]]


@app.post("/api/predictCSV")
def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df_train,orig = preprocess(df)
        listP = []
        res = model.predict(df_train)
        res_rounded = np.round(res)
        res_r = res_rounded.astype(int).ravel()
        orig["pred"] = res_r
        orig["prob"] = res
        
    except Exception as e:
        print(e)
        return jsonable_encoder({"error":e})  
    
    result_list = orig.to_dict(orient='records')
    return jsonable_encoder(result_list)
      
@app.post("/api/getPrediction")
def get_prediction(body: ChurnData):
    cid:Union[str,int] = body.cid
    data:list[Union[float,int]] = body.data
    print(body.data)
    
    df_train = np.array([body.data])
    df_train,orig = preprocess(df_train)
    res = model.predict(df_train)
    return jsonable_encoder({"cid":cid,"prediction":np.round(res).astype(int).ravel().tolist()[0],"probablity": jsonable_encoder(float(res.tolist()[0][0])*100),"data":data})



@app.get("/api/test")
def read_root():
    df_train = pd.read_csv("dataset/pre_train.csv")
    df_train,orig = preprocess(df_train)
    res = model.predict(df_train)
    res_rounded = np.round(res)
    res_r = res_rounded.astype(int).ravel()
    return jsonable_encoder({"Result-Rounded": jsonable_encoder(res_r.tolist()),"Probability": jsonable_encoder(res.tolist())})

@app.get("/api")
def home():
    return jsonable_encoder({"message":"Welcome to Churn Prediction API"})
    