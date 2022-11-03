# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:59:12 2022

@author: Ireneg
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

#####################################Params #############################################################
#########################################################################################################
Panel = 11;
ColorForDisplay = 'Cyan'
CycleNumber =4
SideOffset='LeftSide';


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
fig00 = go.Figure()
fig001 = go.Figure()
fig002 = go.Figure()

# fig00 = make_subplots(rows=3, cols=1,subplot_titles=("LeftSide","Middle", "RightSide"), vertical_spacing=0.1, shared_xaxes=True)

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
# for ColorForDisplay in ColorList:
for ColorForDisplay in ColorList:    
    db=CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();
    
    
    col=list(db.columns)
    
    rnge=range(len(col))
    
    for i in rnge:
    # for i in rnge:
        # if SideOffset=='LeftSide':
        #     offSet=db[i+1][0];
        # else:
        #     if SideOffset=='RightSide':    
        #         offSet=db[i+1][length(len(db[i+1]))]
        #     else:
        #         if  SideOffset=='Middle':
        #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
        #         else:
        #             offSet=0;
        offSet=db[i+1][0];
        fig00.add_trace(
        go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
                    name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
        
        
        offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
        fig001.add_trace(
        go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
                    name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
        
        offSet=db[i+1][(len(db[i+1]))-1]  
        fig002.add_trace(
        go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
                    name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))



fig00.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
fig001.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
fig002.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
fig00.update_layout(title=f+' Left Side Offset WAVE RAW DATA '+side)
fig001.update_layout(title=f+'Middle Side Offset WAVE RAW DATA '+side)
fig002.update_layout(title=f+'Right Side Offset WAVE RAW DATA '+side)

now = datetime.now()


dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# plot(fig00)
plot(fig00,filename=f+" Left Side WaveResult_RawDataPerPanel "+side+".html") 
plot(fig001,filename=f+" Middle Side WaveResult_RawDataPerPanel "+side+".html") 
plot(fig002,filename=f+" Right Side WaveResult_RawDataPerPanel "+side+".html") 


########## BACK ########
try:
    fig000 = go.Figure()
    fig011 = go.Figure()
    fig022 = go.Figure()
    
    # fig00 = make_subplots(rows=3, cols=1,subplot_titles=("LeftSide","Middle", "RightSide"), vertical_spacing=0.1, shared_xaxes=True)
    
    # rnge=[3,6,7]
    
    # db=ImagePlacement_Rightpp
    # db=ImagePlacement_pp
    # for ColorForDisplay in ColorList:
    for ColorForDisplay in ColorList:    
        db=CalcWaveFromRawData(pthF+'/','Back',Panel,ColorForDisplay).ArrangeRawDataForAnalize();
        
        
        col=list(db.columns)
        
        rnge=range(len(col))
        
        for i in rnge:
        # for i in rnge:
            # if SideOffset=='LeftSide':
            #     offSet=db[i+1][0];
            # else:
            #     if SideOffset=='RightSide':    
            #         offSet=db[i+1][length(len(db[i+1]))]
            #     else:
            #         if  SideOffset=='Middle':
            #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
            #         else:
            #             offSet=0;
            offSet=db[i+1][0];
            fig000.add_trace(
            go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
                        name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
            
            
            offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
            fig011.add_trace(
            go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
                        name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
            
            offSet=db[i+1][(len(db[i+1]))-1]  
            fig022.add_trace(
            go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
                        name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
    
    
    
    fig000.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    fig011.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    fig022.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    fig000.update_layout(title=f+' Left Side Offset WAVE RAW DATA-Back')
    fig011.update_layout(title=f+'Middle Side Offset WAVE RAW DATA-Back')
    fig022.update_layout(title=f+'Right Side Offset WAVE RAW DATA-Back')
    
    now = datetime.now()
    
    
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # plot(fig00)
    plot(fig000,filename=f+" Left Side WaveResult_RawDataPerPanel-Back.html") 
    plot(fig011,filename=f+" Middle Side WaveResult_RawDataPerPanel-Back.html") 
    plot(fig022,filename=f+" Right Side WaveResult_RawDataPerPanel-Back.html") 
except:
    1    
    
#########################################
#########################################
#########################################
fig01 = go.Figure()

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
# for ColorForDisplay in ColorList:
for ColorForDisplay in ColorList:
    db=CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();
    
    
    col=list(db.columns)
    
    rnge=range(len(col))
    middledb=pd.DataFrame()
    Rightdb=pd.DataFrame()
    Leftdb=pd.DataFrame()
    
    middleSTD=[]
    RightSTD=[]
    LeftSTD=[]
    for i in rnge:
    # for i in rnge:
        # if SideOffset=='LeftSide':
        #     offSet=db[i+1][0];
        # else:
        #     if SideOffset=='RightSide':    
        #         offSet=db[i+1][length(len(db[i+1]))]
        #     else:
        #         if  SideOffset=='Middle':
        #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
        #         else:
        #             offSet=0;
        
        offSet1=db[i+1][0];
        
        Leftdb=pd.concat([Leftdb,db[i+1]-offSet1],axis=1);
        
        offSet2=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
        middledb=pd.concat([middledb,db[i+1]-offSet2],axis=1);
        
        offSet3=db[i+1][(len(db[i+1]))-1] 
        Rightdb=pd.concat([Rightdb,db[i+1]-offSet3],axis=1);
    
    
    for i in Leftdb.index:
        LeftSTD.append(np.std(Leftdb.loc[i,:]))
        middleSTD.append(np.std(middledb.loc[i,:]))
        RightSTD.append(np.std(Rightdb.loc[i,:]))
       
    
    fig01.add_trace(
    go.Scatter(y=LeftSTD,line_color= ColorForDisplay,
                name='Panel '+str(Panel)+' ' +ColorForDisplay+' LeftSide'))
    
    
    fig01.add_trace(
    go.Scatter(y=middleSTD,line_color= ColorForDisplay,
                name='Panel '+str(Panel)+' ' +ColorForDisplay+' Middle'))
    
    fig01.add_trace(
    go.Scatter(y=RightSTD,line_color= ColorForDisplay,
                name='Panel '+str(Panel)+' ' +ColorForDisplay+' RightSide'))



fig01.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
fig01.update_layout(title=f+'STD Side Offset WAVE RAW DATA-'+side )

now = datetime.now()


dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# plot(fig00)
plot(fig01,filename=f+"STD SideOffset_ WaveResult_RawDataPerPanel "+side+".html") 

################# Back  ########################
try:
    fig010 = go.Figure()
    side='Back';
# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
# for ColorForDisplay in ColorList:
    for ColorForDisplay in ColorList:
        db=CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();
        
        
        col=list(db.columns)
        
        rnge=range(len(col))
        middledb=pd.DataFrame()
        Rightdb=pd.DataFrame()
        Leftdb=pd.DataFrame()
        
        middleSTD=[]
        RightSTD=[]
        LeftSTD=[]
        for i in rnge:
        # for i in rnge:
            # if SideOffset=='LeftSide':
            #     offSet=db[i+1][0];
            # else:
            #     if SideOffset=='RightSide':    
            #         offSet=db[i+1][length(len(db[i+1]))]
            #     else:
            #         if  SideOffset=='Middle':
            #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
            #         else:
            #             offSet=0;
            
            offSet1=db[i+1][0];
            
            Leftdb=pd.concat([Leftdb,db[i+1]-offSet1],axis=1);
            
            offSet2=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
            middledb=pd.concat([middledb,db[i+1]-offSet2],axis=1);
            
            offSet3=db[i+1][(len(db[i+1]))-1] 
            Rightdb=pd.concat([Rightdb,db[i+1]-offSet3],axis=1);
        
        
        for i in Leftdb.index:
            LeftSTD.append(np.std(Leftdb.loc[i,:]))
            middleSTD.append(np.std(middledb.loc[i,:]))
            RightSTD.append(np.std(Rightdb.loc[i,:]))
           
        
        fig010.add_trace(
        go.Scatter(y=LeftSTD,line_color= ColorForDisplay,
                    name='Panel '+str(Panel)+' ' +ColorForDisplay+' LeftSide'))
        
        
        fig010.add_trace(
        go.Scatter(y=middleSTD,line_color= ColorForDisplay,
                    name='Panel '+str(Panel)+' ' +ColorForDisplay+' Middle'))
        
        fig010.add_trace(
        go.Scatter(y=RightSTD,line_color= ColorForDisplay,
                    name='Panel '+str(Panel)+' ' +ColorForDisplay+' RightSide'))
    
    
    
    fig010.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    fig010.update_layout(title=f+'STD Side Offset WAVE RAW DATA-'+side)
    
    now = datetime.now()
    
    
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # plot(fig00)
    plot(fig010,filename=f+"STD SideOffset_ WaveResult_RawDataPerPanel "+side+".html") 
except:
    1    
#########################################
#########################################
#########################################
