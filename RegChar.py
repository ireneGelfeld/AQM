# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:35:59 2022

@author: Ireneg
"""
import os

import plotly.express as px

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


# f=pthF.split('/')[len(pthF.split('/'))-1]
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthF = filedialog.askopenfilename()


# f=pthF.split('/')[len(pthF.split('/'))-1]
DirectorypathF=pthF.split('/');
os.chdir(DirectorypathF[0]+'/'+DirectorypathF[1]+'/'+DirectorypathF[2])
# os.chdir(r'D:\waveCodeExample')


RawData=pd.read_csv(pthF);

RawData=RawData.dropna(axis=0)
RawData=RawData.reset_index(drop=True)
AbsMax=min(RawData['delta_x'])

maxI=max(RawData['i'])
maxJ=max(RawData['j'])



Side=pthF.split('_')[len(pthF.split('_'))-1].split('.')[0]

PageNumberDF=pd.DataFrame();
FirstRawPgNumberDF=pd.DataFrame();
ListofListdelta_x=[]
FirstRaw=[];
tmp=[]

maxL=[]
maxL=[abs(AbsMax)] * (maxJ)
# RawDataFile=RawData[RawData['file #']==1][RawData['i']==0][RawData['j']==0]['delta_x']
for k in range(1,RawData['file #'][len(RawData['file #'])-1]+1):
    
    for i in range(maxI): 
       for j in range(maxJ):
           # if i == 0:
           #     FirstRaw.append(int(RawData[RawData['file #']==k][RawData['i']==i][RawData['j']==j]['delta_x']))
           # else:
               tmp.append(abs(int(RawData[RawData['file #']==k][RawData['i']==i][RawData['j']==j]['delta_x'])))
        
       ListofListdelta_x.append(tmp)
       tmp=[]
       
    ListofListdelta_x.append(maxL)       
    PageNumberDF=pd.concat([PageNumberDF,pd.Series(ListofListdelta_x)],axis=1).rename(columns={0:k})  
    # FirstRawPgNumberDF=pd.concat([FirstRawPgNumberDF,pd.Series(FirstRaw)],axis=1).rename(columns={0:k})  
    ListofListdelta_x=[]


figHeatMap = px.imshow(list(PageNumberDF[1]),color_continuous_scale='OrRd', text_auto=True, aspect="auto")


steps= [{'args': [{'z':[list(PageNumberDF[k])]
                  }], 
         'method': 'update'} for k in range(1,RawData['file #'][len(RawData['file #'])-1]+1)] 


figHeatMap.update_layout(sliders=[dict(active = 0,
                                minorticklen = 0,
                                steps = steps)]);


figHeatMap.update_layout(title='Registration Charecterization absolute values '+Side)
figHeatMap.show()
plot(figHeatMap,filename="Registration Charecterization_"+Side+".html")


















































##### Create list of lists
# PageNumberDF=pd.DataFrame();
# FirstRawPgNumberDF=pd.DataFrame();
# ListofListdelta_x=[]
# FirstRaw=[];
# tmp=[]
# # RawDataFile=RawData[RawData['file #']==1][RawData['i']==0][RawData['j']==0]['delta_x']
# for k in range(1,RawData['file #'][len(RawData['file #'])-1]+1):
    
#     for j in range(9): 
#        for i in range(8):
#            # if i == 0:
#            #     FirstRaw.append(int(RawData[RawData['file #']==k][RawData['i']==i][RawData['j']==j]['delta_x']))
#            # else:
#                tmp.append(int(RawData[RawData['file #']==k][RawData['i']==i][RawData['j']==j]['delta_x']))
        
#        ListofListdelta_x.append(tmp)
#        tmp=[]
       
           
#     PageNumberDF=pd.concat([PageNumberDF,pd.Series(ListofListdelta_x)],axis=1).rename(columns={0:k})  
#     FirstRawPgNumberDF=pd.concat([FirstRawPgNumberDF,pd.Series(FirstRaw)],axis=1).rename(columns={0:k})  

#     ListofListdelta_x=[]
    # FirstRaw=[];

# side='Front'
# headerTilt=[]
# ListofListTiltFRONT=[]
# ListofListTiltBACK=[]
# # PHname=[]

# # for i in range(24):
# #     PHname.append(i)


# for col in ColorList:
#     headerTilt.append(col+' Tilt')
#     # header.append(col+' Tilt')
#     ListofListTiltFRONT.append(PHtiltPerHFRONT[col])
# ColorLevelsTilt=50
# DivideByNumTilt=50
# CellHight=80
# backGroundCLR='rgb(200, 200, 200)'
# colors = n_colors(backGroundCLR, 'rgb(200, 0, 0)', ColorLevelsTilt, colortype='rgb')
# figTableRegChar=go.Figure()
# k=2;
# fillcolorList=[]
# formatList=[]
# formatList.append("")
# fillcolorListFirstRaw=[]
# for i in range(len(PageNumberDF[1])):
#     fillcolorList.append(np.array(colors)[(abs(np.asarray(PageNumberDF[k][i]))/DivideByNumTilt).astype(int)])
# fillcolorList=[list(np.array(colors)[(abs(np.asarray(PageNumberDF[k][i]))/DivideByNumTilt).astype(int)]) for i in range(len(PageNumberDF[1]))]
# fillcolorListFirstRaw=list(np.array(colors)[(abs(np.asarray(FirstRawPgNumberDF[k]))/DivideByNumTilt).astype(int)])
    
# ####FRONT Tilt
# figTableRegChar= go.Figure(data=[go.Table(header=dict(values=list(FirstRawPgNumberDF[k]),fill_color=list(np.array(colors)[(abs(np.asarray(FirstRawPgNumberDF[k]))/DivideByNumTilt).astype(int)]),font=dict(color='black', size=15),height=CellHight,align=['center', 'center']),
#                   cells=dict(values=list(PageNumberDF[k]),fill_color=[list(np.array(colors)[(abs(np.asarray(PageNumberDF[k][i]))/DivideByNumTilt).astype(int)]) for i in range(len(PageNumberDF[k]))],font=dict(color='black', size=15),height=CellHight,align=['center', 'center']))
#                       ])


# steps= [{'args': [{'header.values': [list(FirstRawPgNumberDF[k])],
#                     'header.fill_color':[list(np.array(colors)[(abs(np.asarray(FirstRawPgNumberDF[k]))/DivideByNumTilt).astype(int)])],
#                     'header.font':dict(color='black', size=15),
#                     'header.height':CellHight,
#                     'header.align':['center', 'center'],
#                     'cells.values': [list(PageNumberDF[k])],
#                     'cells.fill_color':[[list(np.array(colors)[(abs(np.asarray(PageNumberDF[k][i]))/DivideByNumTilt).astype(int)]) for i in range(len(PageNumberDF[k]))]],
#                     'cells.height':CellHight,
#                     'cells.align':['center', 'center']
#                   }], 
#           'method': 'update'} for k in range(1,RawData['file #'][len(RawData['file #'])-1]+1)] 

# # sliders = [dict(
# #     active=10,
# #     currentvalue={"prefix": "Window Size: "},
# #     pad={"t": int(1)},
# #     steps=steps
# # )]

# # figTableRegChar.update_layout(
# #     sliders=sliders
# # )
# figTableRegChar.update_layout(sliders=[dict(active = 0,
#                                 #minorticklen = 0,
#                                 steps = steps)]);

# figTableRegChar.update_layout(title='Registration Charecterization')
# # figTableRegChar.update_layout(width=1800, height=1000)

# figTableRegChar.show()





# plot(figTableRegChar,filename="Registration Charecterization.html") 