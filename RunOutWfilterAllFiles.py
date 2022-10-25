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
pthF = filedialog.askdirectory()

os.chdir(pthF)
# f=pthF.split('/')[len(pthF.split('/'))-1]
files = [f for f in glob.glob("**/*.csv", recursive=True)]
# os.chdir(r'D:\waveCodeExample')


f='RunOut'

BarDic={4:'Cyan',6:'Magenta',8:'Yellow',2:'Black',7:'Orange',5:'Green',3:'Blue'}



fig2 = go.Figure()

RawData=pd.read_csv(pthF+'/'+files[0],header = None);
db=RawData
barInx={};
Bar2Clr={};

for i,itm in enumerate(db[0]):
    barInx[i]=itm;
    Bar2Clr[itm]=BarDic[i+2]

for fl in files:
    RawData=pd.read_csv(pthF+'/'+fl,header = None);
    db=RawData

    db=db.rename(index= barInx)    
    db=db.drop(columns=[0])
    # Add traces, one for each slider step
    for value in db.index:
        fig2.add_trace(
            go.Scatter(y=list((db.loc[value,:]-1)*3.14*80000),line_color=Bar2Clr[value] , 
                        name=fl+' '+value))
# fig.add_trace(
#     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
#                 name=ColorForDisplay+'_After'), row=2, col=1)

##### Fiter Vs Befor ####


# Make 10th trace visible
fig2.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)

fig2.show()

# plot(fig)  
now = datetime.now()

dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig2,filename=f+" FRONT-Wave correction "+ dt_string +".html") 
