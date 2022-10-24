# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:08:56 2022

@author: Ireneg
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from datetime import datetime
import glob
from zipfile import ZipFile 
from pathlib import Path
from collections import OrderedDict

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.subplots import make_subplots
import os


#########################################################################################################
MaxWaveWindow=30
FilterDegree = 1

#########################################################################################################
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthF = filedialog.askopenfilename()


# f=pthF.split('/')[len(pthF.split('/'))-1]
DirectorypathF=pthF.split('/');
os.chdir(DirectorypathF[0]+'/'+DirectorypathF[1])
# os.chdir(r'D:\waveCodeExample')


RawData=pd.read_csv(pthF,header = None);
f='RunOut'

BarDic={4:'Cyan',6:'Magenta',8:'Yellow',2:'Black',7:'Orange',5:'Green',3:'Blue'}


import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter

fig2 = go.Figure()

db=RawData

barInx={};
Bar2Clr={};

for i,itm in enumerate(db[0]):
    barInx[i]=itm;
    Bar2Clr[itm]=BarDic[i+2]

db=db.rename(index= barInx)    
db=db.drop(columns=[0])
# Add traces, one for each slider step
for value in db.index:
    fig2.add_trace(
        go.Scatter(y=list(db.loc[value,:]),line_color=Bar2Clr[value] ,line=dict(dash='dash'), 
                    name=value+'_Before'))
# fig.add_trace(
#     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
#                 name=ColorForDisplay+'_After'), row=2, col=1)

##### Fiter Vs Befor ####
for step in  np.arange(3, MaxWaveWindow+3, 2):
    for value in db.index:
        fig2.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color=Bar2Clr[value], width=2),
                name="Window Size = " + str(step),
                y=savgol_filter(list(db.loc[value,:]), step, 1)))



# Make 10th trace visible
for j in range(len(db.index)):
    fig2.data[10+j].visible = True






# Create and add slider
steps = []
for i in range(len(fig2.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig2.data)},
              {"title":f+ "Front Slider switched to Step: " + str(i)}],  # layout attribute
    )

        
    for j in range(len(db.index)):
        if i+j < len(fig2.data):
            step["args"][0]["visible"][i+j] = True  # Toggle i'th trace to "visible"

    for j in range(len(db.index)):
        step["args"][0]["visible"][j] = True 

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
plot(fig2,filename=f+" FRONT-Wave correction "+ dt_string +".html") 
