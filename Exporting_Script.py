import bpy

# Get references to the objects
object1 = bpy.data.objects.get("Object1")
object2 = bpy.data.objects.get("Tesla Model X")

if object1 and object2:
    csv_data = "Frame,Distance,Location\n"  # CSV header
    
    # Iterate over each frame
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        # Set the current frame
        bpy.context.scene.frame_set(frame)
        
        # Get the locations of the objects
        location1 = object1.location
        location2 = object2.location
        
        # Calculate the distance between the objects
        distance = (location1 - location2).length
        location = list(location2[:])
        
        # Append data to the CSV string
        csv_data += f"{frame},{distance},{location}\n"
        
        print(location)
    
    # Specify the file path
    csv_file_path = bpy.path.abspath('//data.csv')  # Save to blend file directory
    
    # Write CSV data to the file
    with open(csv_file_path, 'w') as csv_file:
        csv_file.write(csv_data)
    
    print(f"Data exported to {csv_file_path}")
else:
    print("One or both objects not found.")
