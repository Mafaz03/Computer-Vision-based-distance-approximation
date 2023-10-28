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


# Ignoring Warnings
import warnings
warnings.filterwarnings('ignore')

# Loaing Yolo 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
def resize_and_add_padding(original_image, final_width, final_height):
    # Get the original dimensions
    original_height, original_width, channels = original_image.shape

    # Calculate the scaling factor for resizing while maintaining the aspect ratio
    scaling_factor = final_width / original_width

    # Calculate the new height to maintain the aspect ratio
    new_height = int(original_height * scaling_factor)

    # Resize the image
    resized_image = cv.resize(original_image, (final_width, new_height))

    # Calculate the amount of padding needed at the top and bottom
    padding_top = (final_height - new_height) // 2
    padding_bottom = final_height - new_height - padding_top

    # Create a black background of the final dimensions
    padded_image = np.zeros((final_height, final_width, channels), dtype=np.uint8)

    # Paste the resized image in the middle
    padded_image[padding_top:padding_top + new_height, :, :] = resized_image

    return padded_image


# Rotation_dataset
Location = input("\n\nFile Location : ")
# 112
max_frames = int(input("Max number of frames : "))


with open('RandomForestRegresor_model', 'rb') as f:
    RFG = pickle.load(f)
print("\nRandomForestRegresor_model Loaded Successfully")

classes = {2.0 : "Tire" , 1.0 : "Side-mirror" , 0.0 : "Number-plate"}

## Classes

# 0 - Number-plate
# 1 - Side-mirror
# 2 - Tire

df = pd.DataFrame(columns=["Tire1" , "Tire2" , "Number-plate" , "Side-mirror"])

for i in range(1 , max_frames + 1):
    
    results = model(cv.imread(f"{Location}/0{i}.png"))
    if results.xyxy[0].shape[0] != 0: 
        img = cv.cvtColor(results.render()[0] , cv.COLOR_BGR2RGB)
        cord = results.xyxy[0].numpy()[0][:4].astype(int)
        cropped_img = img[int(cord[1]) : cord[3] , cord[0] : cord[2]]
        cropped_normalized_img = resize_and_add_padding(cropped_img , 800 , 800)

        result = yaw_model(cv.imread(f"{Location}/0{i}.png"))
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

        X = pd.DataFrame(df.iloc[0 : max_frames])
        # print("\n\n")
        # print(X)
        prediction = RFG.predict(X)
        print(f"\n\nRotation Predicted values : {prediction}")


# Location = input("Enter Simulation Frame Location : ")

