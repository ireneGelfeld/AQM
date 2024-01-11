# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:15:15 2024

@author: Ireneg
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.subplots import make_subplots
import plotly.express as px
import math
import numpy as np

##############################
global dataPracentage
bins=20
Name_of_columns_4_hist={'meanC2C_Front':1,
                        'stdC2C_Front':0,
                        'percentile_95_C2C_Front':1,
                        'meanC2C_Back':1,
                        'stdC2C_Back':0,
                        'percentile_95_C2C_Back':1}

dataPracentage=95 #Extreme data to filter 95%

###############################


def load_csvs_and_plot_histogram(directory, string_in_filename):
    files = os.listdir(directory)
    
    sub_files=[]
    
    for file in files:
        if '.' in file:
            continue;
        sub_files.append(file)
    
    filesInDic=[]   
    csv_files=[] 
    for sub_file in sub_files:
        filesInDic=os.listdir(directory+'\\'+sub_file)
        csv_files =csv_files+ [directory+'\\'+sub_file+'\\'+filesCSV for filesCSV in filesInDic if string_in_filename in filesCSV]
            
        
    # Load CSV files into a DataFrame
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df_list.append(df)
    
    # Concatenate DataFrames into one
    concatenated_df = pd.concat(df_list, ignore_index=True)
    
    return  concatenated_df   

def FilterFromDataExtremVals(concatenated_df):
    
    concatenated_df_fitered=pd.DataFrame()
     
    for col in concatenated_df.columns:
        if col=='JobName':
            continue;
        
    
        percentile_x_1 = np.percentile(concatenated_df[col], dataPracentage)
        percentile_1 = np.percentile(concatenated_df[col], 100-dataPracentage)
       
        if math.isnan(percentile_1):
            continue;
        inx2copy = [i for i, x in enumerate(concatenated_df[col]) if x >= percentile_1 and x <= percentile_x_1]
    
        concatenated_df_fitered=pd.concat([concatenated_df_fitered,concatenated_df[col][inx2copy].reset_index(drop=True)],axis=1)          
    return concatenated_df_fitered                     

    
def PlotHistogram(concatenated_df,colName,bins):  
    # Plot histogram using Plotly graph_objects
    fig = px.histogram(concatenated_df, x=colName, title='Histogram '+colName,nbins=bins)
    
    
    plot(fig,auto_play=True,filename='Histogram '+colName+'.html')  
    
    # Show the plot
    fig.show()


###################################################################################
# ################################### 
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthF = filedialog.askdirectory()

directory=pthF+'/';
os.chdir(directory)

# Example usage:
# directory = r'D:\Aqm_AI'
string_in_filename = 'C2C_pressOverview_'
# load_csvs_and_plot_histogram(directory_path, string_to_search)
concatenated_df= load_csvs_and_plot_histogram(directory,string_in_filename)

concatenated_df_fitered= FilterFromDataExtremVals(concatenated_df)


for key,value in Name_of_columns_4_hist.items():
    try:
        if value:
            PlotHistogram(concatenated_df_fitered,key,bins)
    except:
        continue;



