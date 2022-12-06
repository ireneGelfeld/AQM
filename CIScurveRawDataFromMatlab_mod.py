# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:51:03 2022

@author: Ireneg

"""
#######################################################
MaxWaveWindow=100;
limitDataCount=0.18;
BarNum=20
CISsavgolWindow=25

PixelSize_um=84.6666
#######################################################

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


from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthF = filedialog.askdirectory()


f=pthF.split('/')[len(pthF.split('/'))-1]
DirectorypathF=pthF.replace(f,'');
os.chdir(pthF)

RawData=pd.read_csv(pthF+'/RawData.csv',header = None);




z=np.polyfit(RawData[0], RawData[1], 1)
tlt=(z[0]*(RawData[0])+z[1])


RawData_Tilt=RawData[1]-tlt

l=len(RawData_Tilt)

# plt.Figure()
m=plt.hist(abs(RawData_Tilt),BarNum)
# plt.show()

DataCount=m[0]
DataRange=m[1]



NumberOfvalidData=int((1-limitDataCount)*l);

DataSum=0
for i,dt in enumerate(DataCount):
    DataSum=DataSum+dt;
    if DataSum>NumberOfvalidData:
        break;
inx2delete=[]
fixedRawData=[]
for j,dt in enumerate(RawData_Tilt):
    if abs(dt)<  DataRange[i]:
        fixedRawData.append(RawData[1][j])
    else:
       inx2delete.append(j)
       fixedRawData.append(tlt[j]) 
        
# RawData[1]=fixedRawData

RawDataCopy=RawData.copy()

RawDataCopy.drop(index=inx2delete,inplace=True)
RawDataCopy=RawDataCopy.reset_index(drop=True)




plt.figure()
plt.plot(RawDataCopy[0],RawDataCopy[1],'o')
plt.plot(RawData[0],RawData[1],'x')

# plt.plot(RawData[0],tlt)
# plt.plot(RawData[0],fixedRawData)

# DistBtwP=int((RawDataCopy[0][len(RawDataCopy[0])-1])/385)
# Px=RawDataCopy[0][0]
Data385=pd.DataFrame();
# XvalueMean=[];
# xinx=[]
# YvalueMean=[];

DistBtwPFULL=int((RawData[0][len(RawData[0])-1])/385)
XvalueMeanFULL=[]
xinxFULL=[]
PxFull=RawData[0][0]
for i in range(385):
    XvalueMeanFULL.append(PxFull);
    st= np.where(RawData[0] == PxFull)
    xinxFULL.append(st)
    PxFull=PxFull+DistBtwPFULL;
    if PxFull>RawData[0][len(RawData[0])-1]:
        break;
stLoc=[]
enLoc=[]
YvalueMeanFULL=[]

for i in range(len(XvalueMeanFULL)-1):
    st= np.where(RawDataCopy[0] == XvalueMeanFULL[i])
    en= np.where(RawDataCopy[0] == XvalueMeanFULL[i+1])
    if not (len(st[0])==0) and not len(en[0])==0:
        stLoc.append(st[0][0]);
        enLoc.append(en[0][0]);
    if not len(enLoc) == 0:
        YvalueMeanFULL.append(np.mean(RawDataCopy[1][stLoc[len(stLoc)-1]:enLoc[len(enLoc)-1]])) 
        
YvalueMeanFULL=YvalueMeanFULL[0:2]+YvalueMeanFULL
plt.figure()
plt.plot(RawDataCopy[0],RawDataCopy[1],'-x')
plt.plot(XvalueMeanFULL[1:],YvalueMeanFULL,'-o')



    # en= np.where(RawData[0] == Px);
    # YvalueMean.append(np.mean(RawData[1][st[0][0]:en[0][0]]))
# DistBtwP=25   
# for i in range(385):    
#     st= np.where(RawDataCopy[0] == Px)
#     while len(st[0])==0: 
#         Px=Px+1
#         st= np.where(RawDataCopy[0] == Px)
#     XvalueMean.append(Px); 
#     xinx.append(st[0][0])
#     Px=Px+DistBtwP;
#     if Px>RawDataCopy[0][len(RawDataCopy[0])-1]:
#         break;
 
# for i in range(len(XvalueMean)-1):
#     YvalueMean.append(np.mean(RawDataCopy[1][xinx[i]:xinx[i+1]]))

# plt.figure();
# plt.plot(np.diff(RawData[0]),'-o')

 
# plt.figure();
# plt.plot(RawDataCopy[0],RawDataCopy[1],'-o')
# plt.figure();
# plt.plot(XvalueMean[1:],YvalueMean,'-o')



# plt.figure();
# plt.plot(np.diff(RawData[0]),'-o')

   
# plt.figure();
# plt.plot(RawDataCopy[0][1:],np.diff(RawDataCopy[0]),'-o')
    
    
# # plt.figure();
# plt.plot(XvalueMean[1:],np.diff(XvalueMean),'-o')

Data385[0]=XvalueMeanFULL[1:]
Data385[1]=YvalueMeanFULL
Data385[2]=(Data385[1][0]-Data385[1])*PixelSize_um
Data385[3]=(Data385[1]-Data385[1][0])

# Data385[1]=Data385[1]-Data385[1][0]


z=np.polyfit(Data385[0], Data385[3], 1)

tlt=(z[0]*(Data385[0])+z[1])


z1=np.polyfit(Data385[0], Data385[2], 1)

tlt1=(z1[0]*(Data385[0])+z1[1])

CIScurve=pd.DataFrame()

y=savgol_filter(Data385[2], CISsavgolWindow, 1)

# plt.figure()
# plt.plot(y)


for i,yy in enumerate(y):
    CIScurve[i]=[yy]

CIScurve.to_csv('CIScurev.csv',index=False,header=False);

# ####### Plot


# fig2 = go.Figure()


# # Add traces, one for each slider step
# fig2.add_trace(
#     go.Scatter(x=list(RawData[0]),y=list(RawData[1]),line_color='red' , 
#                 name='raw Data'))

# fig2.add_trace(
#     go.Scatter(x=list(RawData[0]),y=tlt,line_color='blue' , 
#                 name='Tilt '+'Slop(x1000)='+"{0:.3f}".format(z[0]*1000)))
# # fig.add_trace(
# #     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
# #                 name=ColorForDisplay+'_After'), row=2, col=1)

# ##### Fiter Vs Befor ####
# for step in  np.arange(3, MaxWaveWindow+3, 2):
#     fig2.add_trace(
#         go.Scatter(
#             visible=False,
#             line=dict(color='green', width=2),
#             name="Window Size = " + str(step),x=list(RawData[0]),
#             y=savgol_filter(RawData[1], step, 1)))



# # Make 10th trace visible
# fig2.data[10].visible = True






# # Create and add slider
# steps = []
# for i in range(len(fig2.data)):
#     step = dict(
#         method="update",
#         args=[{"visible": [False] * len(fig2.data)},
#               {"title":"Slider switched to Step: " + str(i)}],  # layout attribute
#     )

        
#     if i+1 < len(fig2.data):
#         step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"

#     step["args"][0]["visible"][0] = True 
#     step["args"][0]["visible"][1] = True

#     steps.append(step)


# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "Window Size: "},
#     pad={"t": int(MaxWaveWindow)},
#     steps=steps
# )]

# fig2.update_layout(
#     sliders=sliders
# )


# fig2.show()

# # plot(fig)  
# now = datetime.now()

# dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# plot(fig2,filename="CIS curve raw data and filter "+ dt_string +".html") 


################Plot 385-compare

figCompare = go.Figure()


# Add traces, one for each slider step
figCompare.add_trace(
    go.Scatter(x=list(Data385[0]),y=list(Data385[3]),line_color='red' , 
                name='raw Data'))

figCompare.add_trace(
    go.Scatter(x=list(Data385[0]),y=tlt,line_color='blue' , 
                name='Tilt '+'Slope(x1000)='+"{0:.3f}".format(z[0]*1000)))
# fig.add_trace(
#     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
#                 name=ColorForDisplay+'_After'), row=2, col=1)

##### Fiter Vs Befor ####
for step in  np.arange(3, MaxWaveWindow+3, 2):
    figCompare.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color='green', width=2),
            name="Window Size = " + str(step),x=list(Data385[0]),
            y=savgol_filter(Data385[3], step, 1)))



# Make 10th trace visible
figCompare.data[10].visible = True






# Create and add slider
steps = []
for i in range(len(figCompare.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(figCompare.data)},
              {"title":"385 points - For compare Slider switched to Step: " + str(i)}],  # layout attribute
    )

        
    if i+1 < len(figCompare.data):
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

figCompare.update_layout(
    sliders=sliders
)


figCompare.show()

# plot(fig)  
now = datetime.now()

dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(figCompare,filename="CIS curve raw data and filter 385 compre"+ dt_string +".html") 



################Plot 385-calculation

figCIScalc = go.Figure()


# Add traces, one for each slider step
figCIScalc.add_trace(
    go.Scatter(x=list(Data385[0]),y=list(Data385[2]),line_color='red' , 
                name='raw Data'))

figCIScalc.add_trace(
    go.Scatter(x=list(Data385[0]),y=tlt1,line_color='blue' , 
                name='Tilt '+'Slope(x1000)='+"{0:.3f}".format(z1[0]*1000)))
# fig.add_trace(
#     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
#                 name=ColorForDisplay+'_After'), row=2, col=1)

##### Fiter Vs Befor ####
for step in  np.arange(3, MaxWaveWindow+3, 2):
    figCIScalc.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color='green', width=2),
            name="Window Size = " + str(step),x=list(Data385[0]),
            y=savgol_filter(Data385[2], step, 1)))



# Make 10th trace visible
figCIScalc.data[10].visible = True






# Create and add slider
steps = []
for i in range(len(figCIScalc.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(figCIScalc.data)},
              {"title":"385 points - For CIS calc Slider switched to Step: " + str(i)+ ' Tilt in um=' +"{0:.3f}".format(tlt1[0]-tlt1[len(tlt1)-1])}],  # layout attribute
    )

        
    if i+1 < len(figCIScalc.data):
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

figCIScalc.update_layout(
    sliders=sliders
)


figCIScalc.show()

# plot(fig)  
now = datetime.now()

dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(figCIScalc,filename="CIS curve raw data and filter 385 CIS calc"+ dt_string +".html") 
