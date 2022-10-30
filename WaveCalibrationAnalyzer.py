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
startCycle =1

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
db=WaveRaw


col=list(db.columns)

rnge=range(startCycle,len(col))

for i in rnge:
# for i in rnge:
    fig0.add_trace(
    go.Scatter(y=list(db[i]),line_color= ColorForDisplay,
                name='Wave Raw Data cycle '+str(i)))




fig0.update_layout(title=f+' Wave After Correction')

now = datetime.now()


dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig0,filename=f+" WaveResult_"+ dt_string +".html") 



