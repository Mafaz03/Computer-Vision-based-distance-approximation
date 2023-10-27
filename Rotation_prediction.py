import cv2 as cv
import matplotlib. pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pickle
from sklearn.ensemble import RandomForestRegressor

print("""
██████╗  ██████╗ ████████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗    ███████╗██╗███╗   ███╗
██╔══██╗██╔═══██╗╚══██╔══╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║    ██╔════╝██║████╗ ████║
██████╔╝██║   ██║   ██║   ███████║   ██║   ██║██║   ██║██╔██╗ ██║    ███████╗██║██╔████╔██║
██╔══██╗██║   ██║   ██║   ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║    ╚════██║██║██║╚██╔╝██║
██║  ██║╚██████╔╝   ██║   ██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║    ███████║██║██║ ╚═╝ ██║
╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚══════╝╚═╝╚═╝     ╚═╝
""")

yaw_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Yolo on Custom Dataset
print("\n\nCutom Model Loaded Successfully\n\n")

frames_number = int(input("How many Frames : "))
# Rotational_dataset_normalized
Location = input("Enter Simulation Frame Location : ")
# Ignoring Warnings
import warnings
warnings.filterwarnings('ignore')



## Classes

# 0 - Number-plate
# 1 - Side-mirror
# 2 - Tire


classes = {2.0 : "Tire" , 1.0 : "Side-mirror" , 0.0 : "Number-plate"}

df = pd.DataFrame(columns=["Tire1" , "Tire2" , "Number-plate" , "Side-mirror"])


for frame in tqdm(range(1 , frames_number + 1)):
    result = yaw_model(cv.imread(f"{Location}/0{frame}.png"))
    classes = {2.0 : "Tire" , 1.0 : "Side-mirror" , 0.0 : "Number-plate"}
    dic = {"Tire1" : 0 , "Tire2" : 0 , "Side-mirror" : 0 , "Number-plate" : 0} # If not visible then value of the class = 0
    Tire = 0

    for predictions in range(result.xywh[0].shape[0]):    
        cordinates = result.xywh[0][predictions].numpy()
        name = classes[cordinates[-1]]
        Center_x = cordinates[0] 
        
        if name == "Tire":
            if Tire == 0:
                dic["Tire1"] = Center_x
                Tire += 1
            elif Tire == 1:
                dic["Tire2"] = Center_x
        elif name == "Side-mirror":
            dic["Side-mirror"] = Center_x
        elif name == "Number-plate":
            dic["Number-plate"] = Center_x
    df = pd.concat([df , pd.DataFrame([dic])])

print("\n\n",df)

with open('RandomForestRegresor_model', 'rb') as f:
    RFG = pickle.load(f)

print("\nRandomForestRegresor_model Loaded Successfully")

X = pd.DataFrame(df.iloc[0 : frames_number])
print(X)
prediction = RFG.predict(X)
print(f"\n\nRotation Predicted values : {prediction}")