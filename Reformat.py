import pandas as pd


print("""
██████╗ ███████╗███████╗ ██████╗ ██████╗  █████╗ ███╗   ███╗████████╗██╗███╗   ██╗ ██████╗     
██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔══██╗████╗ ████║╚══██╔══╝██║████╗  ██║██╔════╝     
██████╔╝█████╗  █████╗  ██║   ██║██████╔╝███████║██╔████╔██║   ██║   ██║██╔██╗ ██║██║  ███╗    
██╔══██╗██╔══╝  ██╔══╝  ██║   ██║██╔══██╗██╔══██║██║╚██╔╝██║   ██║   ██║██║╚██╗██║██║   ██║    
██║  ██║███████╗██║     ╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║   ██║   ██║██║ ╚████║╚██████╔╝    
╚═╝  ╚═╝╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝                                                                    
""")

# Ignoring Warnings
import warnings
warnings.filterwarnings('ignore')

#Simulation_5.csv
path = input("Path to DataFrame to be reformated : ")
# Reading DataDrame to be reformated
data = pd.read_csv(path , index_col="Unnamed: 0")

## Reformating
# Caling it compatible for blender script

max = int(input("Enter Maximum Amount of cars to expect : "))
reformated_df = pd.DataFrame(columns=[["x" , "y"] * max][0]).reset_index(inplace=True, drop=True)
l_df = pd.DataFrame(columns=[["x" , "y"] * max][0]).reset_index(inplace=True, drop=True)

for j in range(0 , data.shape[0] , 5):
    l = []
    for i in range(max):
        l.extend(((data.iloc[j : j+5 ].iloc[i])[0] , list(data.iloc[j : j+5].iloc[i])[1]))
    l_df = pd.DataFrame(l).transpose()
    reformated_df = pd.concat([reformated_df , l_df])

reformated_df.columns= [["x" , "y"] * max][0]
reformated_df.to_csv(f"Reformated_Simulation_{max}.csv")
print(f"\n\nSaved : Reformated_Simulation_{max}.csv\n\nSimulation_5.csv")