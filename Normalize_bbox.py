import pandas as pd
import cv2 as cv
import torch
import math
import numpy as np


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
file = input("File Location : ")
# 112
max_frames = int(input("Max number of frames : "))
# Rotational_dataset_normalized
out_file = input("Output file Location : ")

failed = 0
for i in range(1 , max_frames + 1):
    try:
        results = model(cv.imread(f"{file}/0{i}.png"))
        img = cv.cvtColor(results.render()[0] , cv.COLOR_BGR2RGB)
        cord = results.xyxy[0].numpy()[0][:4].astype(int)
        cropped_img = img[int(cord[1]) : cord[3] , cord[0] : cord[2]]
        #cropped_img = cv.cvtColor(cropped_img , cv.COLOR_BGR2RGB)
        cropped_normalized_img = resize_and_add_padding(cropped_img , 800 , 800)
        cv.imwrite(f"{out_file}/0{i}.png",cropped_normalized_img)
    except:
        print(f"Failed at frame {i}")
        failed += 1
if failed == 0:
    print("All set")
else:
    print(f"\n\n{failed} failed")