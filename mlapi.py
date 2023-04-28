from fastapi import FastAPI,UploadFile , File,HTTPException
from pydantic import BaseModel
import pickle 
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from keras.models import load_model
import matplotlib.pyplot as plt
import random
import librosa 
import math
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import pyrebase



config = {
  "apiKey": "AIzaSyCYDS_TohcDcJ-dPIvX9uP3s05PKvIkZBg",
  "authDomain": "tuneza-files.firebaseapp.com",
  "projectId": "tuneza-files",
  "storageBucket": "tuneza-files.appspot.com",
  "databaseURL": "https://tuneza-files-default-rtdb.firebaseio.com",
  "serviceAccount":"serviceAccountKey.json"
}

firebase_storage = pyrebase.initialize_app(config)

storage = firebase_storage.storage()

# storage.child('Ed_Sheeran_-_Perfect.mp3').put('Ed_Sheeran_-_Perfect.mp3')



class genre_prediction(BaseModel):
    path:str

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://192.168.43.175:8081",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genre_dict = {0:"rock",1:"blues",2:"disco",3:"raggae",4:"hiphop",5:"metal",6:"country",7:"classical",8:"pop",9:"jazz"}




@app.get("/")
async def genre_prediction():
    return{"Hello":"World"}

def process_input(audio_file, track_duration):

  SAMPLE_RATE = 22050
  NUM_MFCC = 13
  N_FTT=2048
  HOP_LENGTH=512
  TRACK_DURATION = track_duration # measured in seconds
  SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
  NUM_SEGMENTS = 10

  samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
  num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

  load_file ="assets/"+audio_file

  signal, sample_rate = librosa.load(load_file, sr=SAMPLE_RATE)
  
  for d in range(10):

    # calculate start and finish sample for current segment
    start = samples_per_segment * d
    finish = start + samples_per_segment

    # extract mfcc
    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
    mfcc = mfcc.T
    print(mfcc)
    return mfcc

@app.post("/{audio}")
# async def genre_prediction(file:UploadFile=File(...)):
async def genre_prediction(audio):

    filename = audio
    local_filename ="assets/"+filename
    storage.download(filename ,local_filename)

    # with open("pickle_model","rb") as f:
    #   model = pickle.load(f)
    # print(model)

    model = load_model('Music_Genre_10_CNN.h5')
    model.summary()
    new_input_mfcc = process_input(filename, 30)
    X_to_predict = new_input_mfcc[np.newaxis, ..., np.newaxis]
    X_to_predict.shape
    prediction = model.predict(X_to_predict)

    #get index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print(int(predicted_index))

    print("Predicted Genre:", genre_dict[int(predicted_index)])

    # return{"predicted genre is":genre_dict[int(predicted_index)]}
    
    # y,sr = librosa.load(file.filename)
    # # return{'y':y,'sr':sr}
    # print(y)
    # print(sr)
    # # return y
    print(filename)
    # return{'filename':file.filename}
    return {'genre':genre_dict[int(predicted_index)]}
    



    

   