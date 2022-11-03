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
ColorForDisplay = 'Cyan'
CycleNumber =4

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
    
    def getColors(self):
        RawData= self.LoadRawData();
        ColorList=RawData.iloc[:,7].unique().tolist();
        return ColorList
    
    def getNumberOfFlats(self):
        RawData= self.LoadRawData();
        FlatList=RawData.iloc[:,1].unique().tolist();
        return FlatList
    
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

ColorList= CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).getColors();

# FlatList= CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).getNumberOfFlats();


# WaveRaw= CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();


          
     
# RawData=pd.read_csv(r'D:\waveCodeExample\wave\QCS WaveCalibration_500 Archive 18-09-2022 14-15-38 (1)\Front\RawResults\WavePrintDirection.csv');
# c=list(RawData.columns)    
            
    


#########################################
#########################################
#########################################
fig0 = go.Figure()

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
for ColorForDisplay in ColorList:
    db=CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();
    
    
    col=list(db.columns)
    
    rnge=range(len(col))
    
    for i in rnge:
    # for i in rnge:
        fig0.add_trace(
        go.Scatter(y=list(db[i+1]),line_color= ColorForDisplay,
                    name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))



fig0.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
fig0.update_layout(title=f+' WAVE RAW DATA')

now = datetime.now()


dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig0,filename=f+" WaveResult_RawDataPerPanel"+ dt_string +".html") 


#########################################
#########################################
#########################################
fig1 = go.Figure()

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
for Panel in range(1,12):
    # print(Panel)
    for ColorForDisplay in ColorList:
        db=CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();
        
        
        col=list(db.columns)
        
        # rnge=range(len(col))
        
        # for i in rnge:
        # for i in rnge:
        fig1.add_trace(
        go.Scatter(y=list(db[CycleNumber]),line_color= ColorForDisplay,
                    name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))



fig1.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
fig1.update_layout(title=f+' WAVE RAW DATA PER CYCLE')

now = datetime.now()


dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig1,filename=f+" WaveResult_RawDataPerCycle"+ dt_string +".html") 