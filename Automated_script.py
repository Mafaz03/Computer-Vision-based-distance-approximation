import pandas as pd
import cv2 as cv
import torch
import matplotlib.pyplot as plt 
import math
import numpy as np
from tqdm import tqdm

print("""
███████╗██╗███╗   ███╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║████╗ ████║██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
███████╗██║██╔████╔██║██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
╚════██║██║██║╚██╔╝██║██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
███████║██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝|
""")

#Multi_cars_5/multi_cars
file = input("Enter File name : ")
max = int(input("Enter Maximum Amount of cars to expect : "))

# Ignoring Warnings
import warnings
warnings.filterwarnings('ignore')

# Reading Lookup Table
data = pd.read_csv("Distance_data.csv")
data = data.drop(["Unnamed: 0"] , axis = 1)

# Importing Yolo
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def get_dis_Loc(link : str , look_up_table : pd.core.frame.DataFrame , 
                dis_weight : float , h_ = 1.6 , f = 1714) -> tuple[float , float , float]:
  """
  Input : Gets Image
  Output : relates it to lookup table and gets LocX and LocY

  """

  info = []

  link = link
  img = cv.imread(link)
  result = model(img)
  if result.xyxy[0].shape[0] != 0:
    for res in range(result.xyxy[0].shape[0]):
      cordinates = result.xywh[0][res].cpu().numpy()

      Center_x = cordinates[0] + (cordinates[2] / 2)
      Center_y = cordinates[1] + (cordinates[3] / 2)

      d_ = (h_ * f / cordinates[3])
      look_up_table["difference_"] = abs(look_up_table["Distance"] - d_)


      if Center_x > (1920/2):
        data_sub = look_up_table.loc[look_up_table["LocX"] >= 0]


      else:
        data_sub = look_up_table.loc[look_up_table["LocX"] < 0]

      data_sub["Distance_"] = abs(data_sub["Distance"] - d_)
      data_sub["Center_x_"] = abs(data_sub["Center_x"] - Center_x)
      data_sub["Center_y_"] = abs(data_sub["Center_y"] - Center_y)
      data_sub["Center_difference_"] = (data_sub["Center_x_"] + data_sub["Center_y_"] + dis_weight*(data_sub["Distance_"]))/3

      data_min = data_sub["Center_difference_"].idxmin()
      LocX = data.at[data_min , "LocX"]
      LocY = data.at[data_min , "LocY"]
      
      # If prediction doesnt happen then return prevous data as new (helps in keeping plotting in order)
      info.append((d_ , LocX , LocY))
  return info

def disect(result : pd.DataFrame) -> pd.DataFrame:

    """
    Disects the DataFrame, Inserts a break sort of character that makes it 
    easier later on to tell when new frame of cars have started
    """

    index = 0
    reg_index = 0

    index_list = list(result.index)

    result_disect = pd.DataFrame()

    for i in range(result.shape[0]):
        index = int(index_list[i][-1])

        if index == reg_index:
            temp_df = pd.DataFrame({index_list[i] : list(result.iloc[i])}).transpose()
            result_disect = pd.concat([result_disect ,  temp_df])
            reg_index += 1
            
        else:
            
            temp_df = pd.DataFrame({-1 : ["-" , "-"]}).transpose()
            result_disect = pd.concat([result_disect ,  temp_df])
            
            reg_index = 1
    temp_df = pd.DataFrame({-1 : ["-" , "-"]}).transpose()
    result_disect = pd.concat([result_disect ,  temp_df])
    
    result_disect = result_disect.rename({0 : "x" , 1 : "y"} , axis=1)
    return result_disect

result = pd.DataFrame()
for frame in tqdm(range(1,350 + 1)):
    link = f"{file}{frame:04d}.png"
    dashcam_results = get_dis_Loc(link = link , look_up_table = data , dis_weight = 7)
    locx = [y for x , y , z in dashcam_results]
    locy = [z for x , y , z in dashcam_results]
    
    cars = {f"Car{i}" : [locx[i] , locy[i]] for i in range(len(locx))}
    frame_df = pd.DataFrame(cars).transpose()
    result = pd.concat([result, frame_df], axis=0)

result_disect = disect(result)

# Merging the above two DataFrame,
# Fixing previous issue, i.e replacing "-" with car0
merger_df = pd.DataFrame(columns=("x" , "y"))
for i in range(result.shape[0]):
    temp_df = pd.DataFrame(result_disect.iloc[i]).transpose().rename({0 : "x" , 1 : "y"} , axis = 1)
    merger_df = pd.concat([merger_df , temp_df] , axis=0)

    if result_disect.iloc[i].name == -1:
        temp_df = pd.DataFrame(result.iloc[i]).transpose().rename({0 : "x" , 1 : "y"} , axis = 1)
        merger_df = pd.concat([merger_df , temp_df] , axis=0)
    
result = merger_df
print("Disecting and Merging values completed successfully")

# Filling Missing car
# Making it compatible with blender script
test_df = pd.DataFrame(columns=("x" , "y"))

# result.shape[0]
for i in range(result.shape[0]):
    # print(result.iloc[i].name)
    if result.iloc[i].name != -1:
        # print(f"Yes as frame : {i} , {result.iloc[i].name} , {list(result.iloc[i])}")
        car_no = int(result.iloc[i].name[-1])
        # print(car_no)
        temp_df = pd.DataFrame({result.iloc[i].name : list(result.iloc[i])}).transpose().rename({0 : "x" , 1 : "y"} , axis = 1)
        test_df = pd.concat([test_df ,  temp_df] , axis=0)


    if result.iloc[i].name == -1 and car_no != max-1:
        c = car_no
        # print(c)
        to_add_df = pd.DataFrame(columns=("x" , "y"))
        for x in range((max - 1 - c) , 0 , -1):
            temp_to_add = pd.DataFrame({f"Car{max - x}" : [None , None]}).transpose().rename({0 : "x" , 1 : "y"} , axis = 1)
            # print(f"Frame {i} Added Car{max - x}")
            test_df = pd.concat([test_df , temp_to_add], axis=0)
print("Filling values completed successfully")

# Sorting the batches of frames inside a single DataFrame 
result_final = test_df[:test_df.shape[0]-4]
result_final_2 = pd.DataFrame(columns=("x" , "y"))
for i in range(0 , result_final.shape[0] , 5):
    cropped = result_final[i:i+max].sort_values(by = "x").set_index(pd.Index([f"Car{i}" for i in range(max)]))
    result_final_2 = pd.concat([result_final_2 , cropped])
print("Sorting frame batches completed successfully")

result_final_2.to_csv(f"Simulation_{max}.csv")
print(f"Saved : Simulation_{max}.csv")
