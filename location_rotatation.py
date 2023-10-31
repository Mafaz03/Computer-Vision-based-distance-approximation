import pandas as pd
import cv2 as cv
import torch
import matplotlib.pyplot as plt 
import math
import numpy as np
from tqdm import tqdm
import pickle

print("""
███████╗██╗███╗   ███╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║████╗ ████║██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
███████╗██║██╔████╔██║██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
╚════██║██║██║╚██╔╝██║██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
███████║██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ v2
""")



# Ignoring Warnings
import warnings
warnings.filterwarnings('ignore')



# Multi_cars_5/multi_cars
file = input("Simulation Directory : ")

# 5
max = int(input("Max number of Cars to expect : "))

# 350
frames = int(input("Frames to Proccess : "))



yaw_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Yolo on Custom Dataset
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
with open('/Users/mohamedmafaz/Desktop/Distance_approximation/RandomForestRegresor_model', 'rb') as f:
    RFG = pickle.load(f)
print("\nRandomForestRegresor_model Loaded Successfully\n")



def resize_and_add_padding(original_image, final_width, final_height):
    # Get the original dimensions
    original_height, original_width, channels = original_image.shape

    # Calculate the scaling factor for resizing while maintaining the aspect ratio
    scaling_factor_width = final_width / original_width
    scaling_factor_height = final_height / original_height

    # Use the smaller scaling factor to ensure that both dimensions fit within the final image
    scaling_factor = min(scaling_factor_width, scaling_factor_height)

    # Calculate the new dimensions
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # Resize the image
    resized_image = cv.resize(original_image, (new_width, new_height))

    # Calculate the padding on all sides
    padding_top = (final_height - new_height) // 2
    padding_bottom = final_height - new_height - padding_top
    padding_left = (final_width - new_width) // 2
    padding_right = final_width - new_width - padding_left

    # Create a black background of the final dimensions
    padded_image = np.zeros((final_height, final_width, channels), dtype=np.uint8)

    # Paste the resized image in the middle
    padded_image[padding_top:padding_top + new_height, padding_left:padding_left + new_width, :] = resized_image

    return padded_image






def rotation_from_cropped(Location : str , cropped_image_cord : np.array , yaw_model , resize : int) -> int :

    """
    Input 
    Location : String Location of the image file
    cropped_image_cord : cordinates in xyxy
    model : custom model for Custom Key points

    Return
    ROtation : int

    """

    classes = {2.0 : "Tire" , 1.0 : "Side-mirror" , 0.0 : "Number-plate"}

    df = pd.DataFrame(columns=["Tire1" , "Tire2" , "Number-plate" , "Side-mirror"])
    cord = cropped_image_cord
    cropped = cv.imread(Location)[int(cord[1]) : int(cord[3]) , int(cord[0]) : int(cord[2])]
    cropped_padded = resize_and_add_padding(cropped , resize , resize)
        


    cus_result = yaw_model(cropped_padded)

    dic = {"Tire1" : 0 , "Tire2" : 0 , "Side-mirror" : 0 , "Number-plate" : 0} # If not visible then value of the class = 0
    Tire = 0

    for predictions in range(cus_result.xywh[0].shape[0]):    
        cordinates = cus_result.xywh[0][predictions].numpy()
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

    prediction = RFG.predict(df)
    return prediction[0]



def get_dis_Loc_rot(link : str , look_up_table : pd.core.frame.DataFrame , 
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
      rotation = rotation_from_cropped(Location = link , cropped_image_cord = result.xyxy[0][res][:4] , yaw_model = yaw_model , resize = 800)
      
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
      info.append((d_ , LocX , LocY , rotation))
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









# Reading Lookup Table
data = pd.read_csv("Distance_data.csv")
data = data.drop(["Unnamed: 0"] , axis = 1)



result = pd.DataFrame()
for frame in tqdm(range(1,frames + 1)):
    link = f"{file}{frame:04d}.png"
    dashcam_results = get_dis_Loc_rot(link = link , look_up_table = data , dis_weight = 7)
    locx = [y for x , y , z , rot in dashcam_results]
    locy = [z for x , y , z , rot in dashcam_results]
    rot = [i[-1] for i in dashcam_results]
    
    cars = {f"Car{i}" : [locx[i] , locy[i] , rot[i]] for i in range(len(locx))}
    frame_df = pd.DataFrame(cars).transpose()
    result = pd.concat([result, frame_df], axis=0)

print("\nLocation Rotation Extraction Complete\n")
result_disect = disect(result)
result_disect = result_disect.rename(columns = {"x" : "x" , "y" : "y" , 2 : "r"})


# Merging the above two DataFrame
# Fixing previous issue, i.e replacing "-" with car0
merger_df = pd.DataFrame(columns=("x" , "y", "r"))
for i in range(result.shape[0]):
    temp_df = pd.DataFrame(result_disect.iloc[i]).transpose().rename({0 : "x" , 1 : "y" , 2 : "r"} , axis = 1)
    merger_df = pd.concat([merger_df , temp_df] , axis=0)

    if result_disect.iloc[i].name == -1:
        temp_df = pd.DataFrame(result.iloc[i]).transpose().rename({0 : "x" , 1 : "y" , 2 : "r"} , axis = 1)
        merger_df = pd.concat([merger_df , temp_df] , axis=0)

merger_df


result = pd.DataFrame()
for frame in tqdm(range(1,frames + 1)):
    link = f"{file}{frame:04d}.png"
    dashcam_results = get_dis_Loc_rot(link = link , look_up_table = data , dis_weight = 7)
    locx = [y for x , y , z , rot in dashcam_results]
    locy = [z for x , y , z , rot in dashcam_results]
    rot = [i[-1] for i in dashcam_results]
    
    cars = {f"Car{i}" : [locx[i] , locy[i] , rot[i]] for i in range(len(locx))}
    frame_df = pd.DataFrame(cars).transpose()
    result = pd.concat([result, frame_df], axis=0)

result_disect = disect(result)
result_disect = result_disect.rename(columns = {"x" : "x" , "y" : "y" , 2 : "r"})

print("\nDisection Complete\n")

# Merging the above two DataFrame
# Fixing previous issue, i.e replacing "-" with car0
merger_df = pd.DataFrame(columns=("x" , "y", "r"))
for i in range(result.shape[0]):
    temp_df = pd.DataFrame(result_disect.iloc[i]).transpose().rename({0 : "x" , 1 : "y" , 2 : "r"} , axis = 1)
    merger_df = pd.concat([merger_df , temp_df] , axis=0)

    if result_disect.iloc[i].name == -1:
        temp_df = pd.DataFrame(result.iloc[i]).transpose().rename({0 : "x" , 1 : "y" , 2 : "r"} , axis = 1)
        merger_df = pd.concat([merger_df , temp_df] , axis=0)

merger_df


result = merger_df
print("Disecting and Merging values completed successfully\n")

# Filling Missing car
# Making it compatible with blender script
test_df = pd.DataFrame(columns=("x" , "y" , "r"))

# result.shape[0]
for i in range(result.shape[0]):
    # print(result.iloc[i].name)
    if result.iloc[i].name != -1:
        # print(f"Yes as frame : {i} , {result.iloc[i].name} , {list(result.iloc[i])}")
        car_no = int(result.iloc[i].name[-1])
        # print(car_no)
        temp_df = pd.DataFrame({result.iloc[i].name : list(result.iloc[i])}).transpose().rename({0 : "x" , 1 : "y" , 2 : "r"} , axis = 1)
        test_df = pd.concat([test_df ,  temp_df] , axis=0)


    if result.iloc[i].name == -1 and car_no != max-1:
        c = car_no
        # print(c)
        to_add_df = pd.DataFrame(columns=("x" , "y" , "r"))
        for x in range((max - 1 - c) , 0 , -1):
            temp_to_add = pd.DataFrame({f"Car{max - x}" : [None , None]}).transpose().rename({0 : "x" , 1 : "y" , 2 : "r"} , axis = 1)
            # print(f"Frame {i} Added Car{max - x}")
            test_df = pd.concat([test_df , temp_to_add], axis=0)
print("Filling values completed successfully\n")

# Sorting the batches of frames inside a single DataFrame 
result_final = test_df[:test_df.shape[0]-4]
result_final_2 = pd.DataFrame(columns=("x" , "y" , "r"))
result_final_2 = result_final_2.rename(columns = {"x" : "x" , "y" : "y" , 2 : "r"})

for i in range(0 , result_final.shape[0] , max):
    cropped = result_final[i:i+max].sort_values(by = "x") #.set_index(pd.Index([f"Car{i}" for i in range(max)]))
    result_final_2 = pd.concat([result_final_2 , cropped])
print("Sorting frame batches completed successfully\n")
result_final_2 = result_final_2[:-1]

def nearest_multiple_of_5(actual_number , multiple):
    remainder = actual_number % multiple
    nearest_multiple = actual_number - remainder
    return nearest_multiple


nearest = nearest_multiple_of_5(result_final_2.shape[0] , max)
print("NEARTEST : " , nearest)
print("ROWS : " , result_final_2.shape[0])
print("\n")

result_final_2 = result_final_2[:nearest].set_index(pd.Index([f"Car{i}" for i in range(max)] * int(nearest / max)))
result_final_2["r"] = round(result_final_2.r , 2)


rotation_list = []




for i in range(0 , result_final.shape[0] , max):
    rotation_list.append(list(result_final_2[i:i+max].r))

rotation_df = pd.DataFrame(rotation_list)
rotation_df.columns = [f"Car{i}" for i in range(max)]

r_name = f"Rotation_{file.split('/')[0]}_{max}_{frames}.csv"
rotation_df.iloc[-1] = rotation_df.iloc[-2]
rotation_df.to_csv(r_name)

print(f"Rotation DataFrame Saved at {r_name}")

l_name = f"Location_{file.split('/')[0]}_{max}_{frames}.csv"
rotation_df.to_csv(l_name)
result_final_2 = result_final_2.drop("r" , axis = 1)

result_final_2.to_csv(f"{l_name}")

print(f"Location DataFrame Saved at {l_name}")



