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


class CalcWaveFromRawData:
    def  __init__(self, pthF,side,Panel,ColorForDisplay): 
        self.pthF = pthF;
        self.side = side;
        self.Panel= Panel;
        self.ColorForDisplay=ColorForDisplay;
        

    
    def LoadRawData(self):
        RawData=pd.read_csv(self.pthF+self.side+'/'+'RawResults/WavePrintDirection.csv');

        return  RawData;
    
    def FilterRawData(self):
        RawData= self.LoadRawData();
        
        DataSec=RawData[RawData['Overall Status']=='Success']

        DataSecPrintDirc=DataSec[DataSec['Direction Type ']=='Print Direction']
        
        DataSecPrintDircPanel=DataSecPrintDirc[DataSecPrintDirc['Panel Id']==Panel]
        
        DataSecPrintDircPanelColor=DataSecPrintDircPanel[DataSecPrintDircPanel[' Seperation']==self.ColorForDisplay].reset_index(drop=True)
        
        col=list(DataSecPrintDircPanelColor.columns)
        
        cutCols=col[12:396]
        
        DataSecPrintDircPanelColorCUT=DataSecPrintDircPanelColor[cutCols];
        
        return DataSecPrintDircPanelColorCUT,cutCols;
    
    
    def ArrangeRawDataForAnalize(self):
       
        DataSecPrintDircPanelColorCUT,cutCols=self.FilterRawData();
        WaveRaw=pd.DataFrame();

        for i in range(len(DataSecPrintDircPanelColorCUT[cutCols[0]])):
            l=list(DataSecPrintDircPanelColorCUT.loc[i,:])
            tmpDF=pd.DataFrame();
            while (1):
                tmp=next((j for j, x in enumerate(l) if not isinstance(x, float)), 'DONE');
                if  tmp == 'DONE':
                    break;
                else:
                   if  l[tmp].replace('.', '', 1).isdigit():
                       l[tmp]=float(l[tmp]);
                   else: 
                       if l[tmp] == 'NotFound':
                           break;
            if not tmp == 'DONE':
               WaveRaw=pd.concat([WaveRaw,pd.DataFrame(l[0:tmp-1])],axis=1).rename(columns={0:i+1}) 
        return  WaveRaw;      
            

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


WaveRaw= CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();


          
     
    
            
    


#########################################
#########################################
#########################################
fig0 = go.Figure()

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
db=RawDataSecPrintDircPclr


col=list(db.columns)

rnge=range(len(col))

for i in rnge:
# for i in rnge:
    fig0.add_trace(
    go.Scatter(y=list(db.loc[1,:]),line_color= ColorDic[i+1],line=dict(dash='dash'),
                name=col[i]+'_After',visible='legendonly'))
    fig0.add_trace(
    go.Scatter(y=list(db1[col[i]]),line_color= ColorDic[i+1], line=dict(dash='dot'),
                name=col[i]+'_Before',visible='legendonly'))
    fig0.add_trace(
    go.Scatter(y=list(db2[col[i]]),line_color= ColorDic[i+1],
                name=col[i]+'_Correction'))













  
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
                name=col[i]+'_After'))
    fig0.add_trace(
    go.Scatter(y=list(db1[col[i]]),line_color= ColorDic[i+1], line=dict(dash='dot'),
                name=col[i]+'_Before'))
    fig0.add_trace(
    go.Scatter(y=list(db2[col[i]]),line_color= ColorDic[i+1],
                name=col[i]+'_Correction'))


# fig.update_layout(title='ImagePlacement_Right')
fig0.update_layout(title=f+' Wave After Correction')

# fig.update_layout(title='ImagePlacement_Left-back')

# fig.update_layout(
#     legend=dict(x= 1.1,y=1.1,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),
#     width=1000,
#     height=600,
#     autosize=False,
#     template="plotly_white",
#     # side ='left'
# )

# fig.show()        

# plot(fig)  
now = datetime.now()

dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig0,filename=f+" WaveResult_"+ dt_string +".html") 




#################################################################################
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
# from matplotlib.widgets import Slider

# import numpy as np
# # Generating the noisy signal 
# # x = np.linspace(0, 2*np.pi, 100)
# y = db1[col[1]]

# # Savitzky-Golay filter
# y_filtered = savgol_filter(y, 99, 1)



# # Plotting
# fig = plt.figure()
# ax = fig.subplots()
# # p = ax.plot(y, '-*')
# p = ax.plot(y)

# p, = ax.plot(y_filtered, 'g')
# plt.subplots_adjust(bottom=0.25)

# # Defining the Slider button
# ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03]) #xposition, yposition, width and height
# # Properties of the slider
# win_size = Slider(ax_slide, 'Window size', valmin=5, valmax=99, valinit=99, valstep=2)

# # Updating the plot
# def update(val):
#     current_v = int(win_size.val)
#     new_y = savgol_filter(y, current_v, 3)
#     p.set_ydata(new_y)
#     fig.canvas.draw() #redraw the figure

# # calling the function "update" when the value of the slider is changed
# win_size.on_changed(update)
# plt.show()

############################ Original Code 

# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
# from matplotlib.widgets import Slider

# import numpy as np
# # Generating the noisy signal 
# x = np.linspace(0, 2*np.pi, 100)
# y = np.sin(x) + np.cos(x) + np.random.random(100)

# # Savitzky-Golay filter
# y_filtered = savgol_filter(y, 99, 3)



# # Plotting
# fig = plt.figure()
# ax = fig.subplots()
# p = ax.plot(x, y, '-*')
# p, = ax.plot(x, y_filtered, 'g')
# plt.subplots_adjust(bottom=0.25)

# # Defining the Slider button
# ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03]) #xposition, yposition, width and height
# # Properties of the slider
# win_size = Slider(ax_slide, 'Window size', valmin=5, valmax=99, valinit=99, valstep=2)

# # Updating the plot
# def update(val):
#     current_v = int(win_size.val)
#     new_y = savgol_filter(y, current_v, 3)
#     p.set_ydata(new_y)
#     fig.canvas.draw() #redraw the figure

# # calling the function "update" when the value of the slider is changed
# win_size.on_changed(update)
# plt.show()

#########################  Plotly With Slider ##################################

import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter

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
              {"title":f+ " Slider switched to Window Size: " + str(i)}],  # layout attribute
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
#####################################################################################################
#########################  Plotly With Slider  All Colors##################################

import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter

fig = go.Figure()
#fig_back = go.Figure()
# fig = make_subplots(rows=2, cols=1,subplot_titles=("Fliter + Before", "Filter + After"), vertical_spacing=0.1, shared_xaxes=True)





# Create figure


# Add traces, one for each slider step
for value in db1.columns:
    fig.add_trace(
        go.Scatter(y=list(db1[value]),line_color=value , 
                    name=value+'_Before'))
# fig.add_trace(
#     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
#                 name=ColorForDisplay+'_After'), row=2, col=1)

##### Fiter Vs Befor ####
for step in  np.arange(3, MaxWaveWindow+3, 2):
    for value in db1.columns:
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color=value, width=2),
                name="Window Size = " + str(step),
                y=savgol_filter(db1[value], step, 1)))

# for step in  np.arange(3, MaxWaveWindow+3, 2):    
    # fig.add_trace(
    #     go.Scatter(
    #         visible=False,
    #         # line=dict(color="#00CED1", width=1),
    #         line=dict(color="#d8576b", width=2),
    #         name="Window Size = " + str(step),
    #         y=savgol_filter(db1[ColorForDisplay], step, 1)), row=2, col=1)


# Make 10th trace visible
for j in range(len(db1.columns)):
    fig.data[10+j].visible = True






# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title":f+ " Slider switched to Window Size: " + str(i)}],  # layout attribute
    )

        
    for j in range(len(db1.columns)):
        if i+j < len(fig.data):
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

fig.update_layout(
    sliders=sliders
)


fig.show()

# plot(fig)  
now = datetime.now()

dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig,filename=f+" FRONT-Wave correction "+ dt_string +".html") 
#####################################################################################################



#########################  Original code- Plotly With Slider ##################################

# import plotly.graph_objects as go
# import numpy as np

# # Create figure
# fig = go.Figure()

# # Add traces, one for each slider step
# for step in np.arange(0, 5, 0.1):
#     fig.add_trace(
#         go.Scatter(
#             visible=False,
#             line=dict(color="#00CED1", width=6),
#             name="𝜈 = " + str(step),
#             x=np.arange(0, 10, 0.01),
#             y=np.sin(step * np.arange(0, 10, 0.01))))

# # Make 10th trace visible
# fig.data[10].visible = True

# # Create and add slider
# steps = []
# for i in range(len(fig.data)):
#     step = dict(
#         method="update",
#         args=[{"visible": [False] * len(fig.data)},
#               {"title": "Slider switched to step: " + str(i)}],  # layout attribute
#     )
#     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
#     steps.append(step)

# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "Frequency: "},
#     pad={"t": 50},
#     steps=steps
# )]

# fig.update_layout(
#     sliders=sliders
# )

# fig.show()

# plot(fig)  