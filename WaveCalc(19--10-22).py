# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:59:12 2022

@author: Ireneg
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

#####################################Params #############################################################
#########################################################################################################
Panel = 6;
MaxWaveWindow=100
FilterDegree = 1
ColorForDisplay = 'Cyan'
ShowOriginalFilter=1;
#########################################################################################################
#########################################################################################################
import os


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


class CalcWave:
    def  __init__(self, pthF,side,Panel): 
        self.pthF = pthF;
        self.side = side;
        self.Panel= Panel;
        

    
    def LoadRawData(self):
        RawData=pd.read_csv(self.pthF+self.side+'/'+'CorrectionOperators/WavePrintDirectionCorrection.csv',skiprows=1,header = None);
        Hder = pd.read_csv(self.pthF+self.side+'/'+'CorrectionOperators/WavePrintDirectionCorrection.csv', index_col=0, nrows=0).columns.tolist()   

        return  RawData,Hder;
    
    
    def CreatBeforeAfterDFforSpecificPanel(self,Colors,RawData,ColorDic,BeforAfterCorrection):
        BeforAfterCorrByColor=pd.DataFrame();          
        for i,clr in enumerate(Colors):
            BeforAfterCorr=RawData[RawData[3] == BeforAfterCorrection][RawData[2] == self.Panel][RawData[1]== i+1].reset_index(drop=True);
            D = BeforAfterCorr.iloc[0,4:].reset_index(drop=True);
            BeforAfterCorrByColor=pd.concat([BeforAfterCorrByColor,D.rename(ColorDic[i+1])],axis=1)
        return BeforAfterCorrByColor;
    
    def OrgnazeDataByColorAndCorrectionState(self):
         RawData,Hder = self.LoadRawData();
         Colors=RawData.iloc[:,1].unique().tolist()
         ColorDic={1:'Cyan',2:'Magenta',3:'Yellow',4:'Black',5:'Orange',7:'Green',6:'Blue'}
         BarDic={4:'Cyan',6:'Magenta',8:'Yellow',2:'Black',7:'Orange',5:'Green',3:'Blue'}
         
         BeforCorrByColor =  self.CreatBeforeAfterDFforSpecificPanel(Colors,RawData,ColorDic,' Before')
         AfterCorrByColor =  self.CreatBeforeAfterDFforSpecificPanel(Colors,RawData,ColorDic,' After')
         CorrectionCorrByColor =  self.CreatBeforeAfterDFforSpecificPanel(Colors,RawData,ColorDic,'Correction')
         
         return ColorDic,BarDic,BeforCorrByColor,AfterCorrByColor,CorrectionCorrByColor;

#################################################################################
#################################################################################
#################################################################################

from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthF = filedialog.askdirectory()


f=pthF.split('/')[len(pthF.split('/'))-1]
DirectorypathF=pthF.replace(f,'');
os.chdir(DirectorypathF)





side='Front';


ColorDic,BarDic,BeforCorrByColorFront,AfterCorrByColorFront,CorrectionCorrByColorFront=CalcWave(pthF+'/',side,Panel).OrgnazeDataByColorAndCorrectionState();

try:
    
    side='Back';


    ColorDic,BarDic,BeforCorrByColorBack,AfterCorrByColorBack,CorrectionCorrByColorBack=CalcWave(pthF+'/',side,Panel).OrgnazeDataByColorAndCorrectionState();


except:
    1






  
# RawData=pd.read_csv(FilePath,skiprows=1,header = None);
# Hder = pd.read_csv(FilePath, index_col=0, nrows=0).columns.tolist()   
# l1=RawData.iloc[:,1].unique().tolist()





# Colors=RawData.iloc[:,1].unique().tolist()

# ColorDic={1:'Cyan',2:'Magenta',3:'Yellow',4:'Black',5:'Orange',7:'Green',6:'Blue'}
# BarDic={4:'Cyan',6:'Magenta',8:'Yellow',2:'Black',7:'Orange',5:'Green',3:'Blue'}



# # BeforCorr=RawData[RawData[3] == ' Before'][RawData[2] == 1][RawData[1]== 4].reset_index(drop=True);
# # D = BeforCorr.iloc[0,4:].reset_index(drop=True);



# def CreatBeforeAfterDFforSpecificPanel(Panel,Colors,RawData,ColorDic,BeforAfterCorrection):
#     BeforAfterCorrByColor=pd.DataFrame();
#     for i,clr in enumerate(Colors):
#         BeforAfterCorr=RawData[RawData[3] == BeforAfterCorrection][RawData[2] == Panel][RawData[1]== i+1].reset_index(drop=True);
#         D = BeforAfterCorr.iloc[0,4:].reset_index(drop=True);
#         BeforAfterCorrByColor=pd.concat([BeforAfterCorrByColor,D.rename(ColorDic[i+1])],axis=1)
#     return BeforAfterCorrByColor;
        
# BeforCorrByColor =  CreatBeforeAfterDFforSpecificPanel(1,Colors,RawData,ColorDic,' Before')
# AfterCorrByColor =  CreatBeforeAfterDFforSpecificPanel(1,Colors,RawData,ColorDic,' After')
# CorrectionCorrByColor =  CreatBeforeAfterDFforSpecificPanel(1,Colors,RawData,ColorDic,'Correction')






################################## After Correction

fig0 = go.Figure()

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
db=AfterCorrByColorFront
db1=BeforCorrByColorFront
db2=CorrectionCorrByColorFront

col=list(db.columns)

rnge=range(len(col))

for i in rnge:
# for i in rnge:
    fig0.add_trace(
    go.Scatter(y=list(db[col[i]]),line_color= ColorDic[i+1],line=dict(dash='dash'),
                name=col[i]+'_After',visible='legendonly'))
    fig0.add_trace(
    go.Scatter(y=list(db1[col[i]]),line_color= ColorDic[i+1], line=dict(dash='dot'),
                name=col[i]+'_Before',visible='legendonly'))
    fig0.add_trace(
    go.Scatter(y=list(db2[col[i]]),line_color= ColorDic[i+1],
                name=col[i]+'_Correction'))


fig0.update_layout(title=f+' Wave After Correction Front')


now = datetime.now()

dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig0,filename=f+" WaveResult_Front_"+ dt_string +".html") 


try:
    fig00 = go.Figure()

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
    db=AfterCorrByColorBack
    db1=BeforCorrByColorBack
    db2=CorrectionCorrByColorBack
    
    col=list(db.columns)
    
    rnge=range(len(col))
    
    for i in rnge:
    # for i in rnge:
        fig00.add_trace(
        go.Scatter(y=list(db[col[i]]),line_color= ColorDic[i+1],line=dict(dash='dash'),
                    name=col[i]+'_After',visible='legendonly'))
        fig00.add_trace(
        go.Scatter(y=list(db1[col[i]]),line_color= ColorDic[i+1], line=dict(dash='dot'),
                    name=col[i]+'_Before',visible='legendonly'))
        fig00.add_trace(
        go.Scatter(y=list(db2[col[i]]),line_color= ColorDic[i+1],
                    name=col[i]+'_Correction'))
    
    
    fig00.update_layout(title=f+' Wave After Correction Back')
    
    
    now = datetime.now()
    
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    plot(fig00,filename=f+" WaveResult_Back_"+ dt_string +".html") 

except:
    1



#########################  Plotly With Slider ##################################

import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter



if ShowOriginalFilter == 1:
    db=AfterCorrByColorFront
    db1=BeforCorrByColorFront
    db2=CorrectionCorrByColorFront
    
    fig = go.Figure()
    #fig_back = go.Figure()
    fig = make_subplots(rows=2, cols=1,subplot_titles=("Fliter + Before", "Filter + After"), vertical_spacing=0.1, shared_xaxes=True)
    
    
    
    
    
    # Create figure
    
    
    # Add traces, one for each slider step
    
    fig.add_trace(
        go.Scatter(y=list(db1[ColorForDisplay]),line_color=ColorForDisplay , 
                    name=ColorForDisplay+'_Before'), row=1, col=1)
    fig.add_trace(
        go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
                    name=ColorForDisplay+'_After'), row=2, col=1)
    
    ##### Fiter Vs Befor ####
    for step in  np.arange(3, MaxWaveWindow+3, 2):
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#46039f", width=2),
                name="Window Size = " + str(step),
                y=savgol_filter(db1[ColorForDisplay], step, 1)), row=1, col=1)
    
    # for step in  np.arange(3, MaxWaveWindow+3, 2):    
        fig.add_trace(
            go.Scatter(
                visible=False,
                # line=dict(color="#00CED1", width=1),
                line=dict(color="#d8576b", width=2),
                name="Window Size = " + str(step),
                y=savgol_filter(db1[ColorForDisplay], step, 1)), row=2, col=1)
    
    
    # Make 10th trace visible
    fig.data[10].visible = True
    fig.data[11].visible = True
    
    
    
    
    
    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title":f+ "Front Slider switched to Step: " + str(i)}],  # layout attribute
        )
    
            
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        if i+1 < len(fig.data):
            step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
    
        step["args"][0]["visible"][0] = True 
        step["args"][0]["visible"][1] = True 
    
        steps.append(step)
    
    
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Window Size: "},
        pad={"t": int(MaxWaveWindow)},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders
    )
    
    
    fig.show()
    
    # plot(fig)  
    now = datetime.now()
    
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    plot(fig,filename=f+" FRONT-Wave correction "+ dt_string +".html") 
    
    try:
        db=AfterCorrByColorBack
        db1=BeforCorrByColorBack
        db2=CorrectionCorrByColorBack
        
        fig11 = go.Figure()
        #fig_back = go.Figure()
        fig11 = make_subplots(rows=2, cols=1,subplot_titles=("Fliter + Before", "Filter + After"), vertical_spacing=0.1, shared_xaxes=True)
        
        
        
        
        
        # Create figure
        
        
        # Add traces, one for each slider step
        
        fig11.add_trace(
            go.Scatter(y=list(db1[ColorForDisplay]),line_color=ColorForDisplay , 
                        name=ColorForDisplay+'_Before'), row=1, col=1)
        fig11.add_trace(
            go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
                        name=ColorForDisplay+'_After'), row=2, col=1)
        
        ##### Fiter Vs Befor ####
        for step in  np.arange(3, MaxWaveWindow+3, 2):
            fig11.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color="#46039f", width=2),
                    name="Window Size = " + str(step),
                    y=savgol_filter(db1[ColorForDisplay], step, 1)), row=1, col=1)
        
        # for step in  np.arange(3, MaxWaveWindow+3, 2):    
            fig11.add_trace(
                go.Scatter(
                    visible=False,
                    # line=dict(color="#00CED1", width=1),
                    line=dict(color="#d8576b", width=2),
                    name="Window Size = " + str(step),
                    y=savgol_filter(db1[ColorForDisplay], step, 1)), row=2, col=1)
        
        
        # Make 10th trace visible
        fig11.data[10].visible = True
        fig11.data[11].visible = True
        
        
        
        
        
        # Create and add slider
        steps = []
        for i in range(len(fig11.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig11.data)},
                      {"title":f+ "Back Slider switched to Step: " + str(i)}],  # layout attribute
            )
        
                
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            if i+1 < len(fig11.data):
                step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
        
            step["args"][0]["visible"][0] = True 
            step["args"][0]["visible"][1] = True 
        
            steps.append(step)
        
        
        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Window Size: "},
            pad={"t": int(MaxWaveWindow)},
            steps=steps
        )]
        
        fig11.update_layout(
            sliders=sliders
        )
        
        
        fig11.show()
        
        # plot(fig)  
        now = datetime.now()
        
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        plot(fig11,filename=f+" Back-Wave correction "+ dt_string +".html") 
    except:
        1
#####################################################################################################
#########################  Plotly With Slider  All Colors##################################

import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter

fig2 = go.Figure()

db=AfterCorrByColorFront
db1=BeforCorrByColorFront
db2=CorrectionCorrByColorFront
# Add traces, one for each slider step
for value in db1.columns:
    fig2.add_trace(
        go.Scatter(y=list(db1[value]),line_color=value , 
                    name=value+'_Before'))
# fig.add_trace(
#     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
#                 name=ColorForDisplay+'_After'), row=2, col=1)

##### Fiter Vs Befor ####
for step in  np.arange(3, MaxWaveWindow+3, 2):
    for value in db1.columns:
        fig2.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color=value, width=2),
                name="Window Size = " + str(step),
                y=savgol_filter(db1[value], step, 1)))



# Make 10th trace visible
for j in range(len(db1.columns)):
    fig2.data[10+j].visible = True






# Create and add slider
steps = []
for i in range(len(fig2.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig2.data)},
              {"title":f+ "Front Slider switched to Step: " + str(i)}],  # layout attribute
    )

        
    for j in range(len(db1.columns)):
        if i+j < len(fig2.data):
            step["args"][0]["visible"][i+j] = True  # Toggle i'th trace to "visible"

    for j in range(len(db1.columns)):
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


try:
    fig22 = go.Figure()

    db=AfterCorrByColorBack
    db1=BeforCorrByColorBack
    db2=CorrectionCorrByColorBack
    # Add traces, one for each slider step
    for value in db1.columns:
        fig22.add_trace(
            go.Scatter(y=list(db1[value]),line_color=value , 
                        name=value+'_Before'))
    # fig.add_trace(
    #     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
    #                 name=ColorForDisplay+'_After'), row=2, col=1)
    
    ##### Fiter Vs Befor ####
    for step in  np.arange(3, MaxWaveWindow+3, 2):
        for value in db1.columns:
            fig22.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color=value, width=2),
                    name="Window Size = " + str(step),
                    y=savgol_filter(db1[value], step, 1)))
    
    
    
    # Make 10th trace visible
    for j in range(len(db1.columns)):
        fig22.data[10+j].visible = True
    
    
    
    
    
    
    # Create and add slider
    steps = []
    for i in range(len(fig22.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig22.data)},
                  {"title":f+ "Back Slider switched to Step: " + str(i)}],  # layout attribute
        )
    
            
        for j in range(len(db1.columns)):
            if i+j < len(fig22.data):
                step["args"][0]["visible"][i+j] = True  # Toggle i'th trace to "visible"
    
        for j in range(len(db1.columns)):
            step["args"][0]["visible"][j] = True 
    
        steps.append(step)
    
    
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Window Size: "},
        pad={"t": int(MaxWaveWindow)},
        steps=steps
    )]
    
    fig22.update_layout(
        sliders=sliders
    )
    
    
    fig22.show()
    
    # plot(fig)  
    now = datetime.now()
    
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    plot(fig22,filename=f+" BACK-Wave correction "+ dt_string +".html") 
except:
    1
#####################################################################################################





