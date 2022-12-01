# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:51:03 2022

@author: Ireneg
"""
import os


import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from datetime import datetime
import glob
from zipfile import ZipFile 
from pathlib import Path
from collections import OrderedDict
from scipy.signal import savgol_filter
from plotly.colors import n_colors
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.subplots import make_subplots




RawData=pd.read_csv('D:/CIScurve/Ariel_matlab/CIS tilt/2.bmp/Cropped images/Cropped images/RawData.csv',header = None);

MaxWaveWindow=100;

fig2 = go.Figure()


# Add traces, one for each slider step
fig2.add_trace(
    go.Scatter(x=list(RawData[0]),y=list(RawData[1]),line_color='red' , 
                name='raw Data'))
# fig.add_trace(
#     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
#                 name=ColorForDisplay+'_After'), row=2, col=1)

##### Fiter Vs Befor ####
for step in  np.arange(3, MaxWaveWindow+3, 2):
    fig2.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color='green', width=2),
            name="Window Size = " + str(step),x=list(RawData[0]),
            y=savgol_filter(RawData[1], step, 1)))



# Make 10th trace visible
fig2.data[10].visible = True






# Create and add slider
steps = []
for i in range(len(fig2.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig2.data)},
              {"title":"Slider switched to Step: " + str(i)}],  # layout attribute
    )

        
    if i+1 < len(fig2.data):
        step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"

    step["args"][0]["visible"][0] = True 

    steps.append(step)


sliders = [dict(
    active=10,
    currentvalue={"prefix": "Window Size: "},
    pad={"t": int(MaxWaveWindow)},
    steps=steps
)]

fig2.update_layout(
    sliders=sliders
)


fig2.show()

# plot(fig)  
now = datetime.now()

dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig2,filename="CIS curve raw data and filter "+ dt_string +".html") 
