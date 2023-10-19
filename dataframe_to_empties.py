import bpy
import math
import subprocess
import pandas as pd

# Set the frame rate and total number of frames
frame_rate = 24

df = pd.read_csv("/Users/mohamedmafaz/Desktop/Distance_approximation/Reformated_Simulation_5.csv" , index_col=0)
for col in range(0 , df.shape[1] , 2):

    x_sequence = list(df.iloc[ : ,col])
    y_sequence = list(df.iloc[ : ,col + 1])
 
    num_frames = len(x_sequence)  # Assuming x_sequence and y_sequence are predefined lists

    # Create an empty object
    bpy.ops.object.empty_add(location=(0, 0, 0))
    empty = bpy.context.object
    empty.name = "MyEmpty"

    # Create a new animation data for the empty
    empty.animation_data_create()
    action = bpy.data.actions.new(name="MyAnimation")
    empty.animation_data.action = action

    # Iterate through the sequences and set keyframes
    for frame_num, (x, y) in enumerate(zip(x_sequence, y_sequence), start=1):
        frame_time = frame_num / frame_rate
        bpy.context.scene.frame_set(frame_num)
        empty.location = (x, y, 0)
        empty.keyframe_insert(data_path="location", frame=frame_num)

    # Set the timeline length to match the animation duration
    bpy.context.scene.frame_end = num_frames

    # Set the animation playback range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
