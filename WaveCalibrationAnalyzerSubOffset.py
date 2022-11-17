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
CycleNumber =3
StartCycle=2
StartCycle4Avr = 2;
MainColor = "Black"
LeftSide=1;
Middle=1;
RightSide=1;
CIScurve=1;
registrationBetweenWavePrints=0;
presentAllColors=0;
MaxWaveWindow=51;
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
from scipy.signal import savgol_filter

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
        LocatorIndex= DataSec['Locator Index'][0];
        
        return LocatorIndex,DataSecPrintDircPanelColorCUT,cutCols;
    
    
    def ArrangeRawDataForAnalize(self):
       
        LocatorIndex,DataSecPrintDircPanelColorCUT,cutCols=self.FilterRawData();
        WaveRaw=pd.DataFrame();

        for i in range(len(DataSecPrintDircPanelColorCUT[cutCols[0]])):
            l=list(DataSecPrintDircPanelColorCUT.loc[i,:])
            tmpDF=pd.DataFrame();
            while (1):
                tmp=next((j for j, x in enumerate(l) if not isinstance(x, float)), 'DONE');
                if  tmp == 'DONE':
                    break;
                else:
                   if  l[tmp].replace('.', '', 1).replace('-', '').isdigit():
                       l[tmp]=float(l[tmp]);
                   else: 
                       if l[tmp] == 'NotFound':
                           break;
            if not tmp == 'DONE':
               WaveRaw=pd.concat([WaveRaw,pd.DataFrame(l[0:tmp-1])],axis=1).rename(columns={0:i+1}) 
        return  WaveRaw;  

    def GetLocatorIndex(self):
        LocatorIndex,DataSecPrintDircPanelColorCUT,cutCols=self.FilterRawData();
        return LocatorIndex;


class CalcRegistrationFromWaveData:
    def  __init__(self, pthF,side,Panel,ColorList,MainColor,StartCycle): 
        self.pthF = pthF;
        self.side = side;
        self.Panel= Panel;
        self.ColorList=ColorList;
        self.MainColor=MainColor;
        self.StartCycle=StartCycle;
        

   
    
    def DeltaForCycleAndColor(self):
        
        DeltaPerCycleFromRef=pd.DataFrame();
        DeltaPerCycle=pd.DataFrame();
        
        mainColorRef=CalcWaveFromRawData(self.pthF+'/',self.side,self.Panel,self.MainColor).ArrangeRawDataForAnalize();
        DFdicPerClr={}
        for clr in self.ColorList:
            if clr == self.MainColor:
                continue;
            DeltaPerCycleFromRef=pd.DataFrame();
            DeltaPerCycle=pd.DataFrame();    
            ColorWavePerCycle=CalcWaveFromRawData(self.pthF+'/',self.side,self.Panel,clr).ArrangeRawDataForAnalize();
            DeltaPerCycleFromRef= mainColorRef.loc[:,self.StartCycle:]-ColorWavePerCycle.loc[:,self.StartCycle:];
            for col in DeltaPerCycleFromRef.loc[:,StartCycle:].columns:
                DeltaPerCycle=pd.concat([DeltaPerCycle,DeltaPerCycleFromRef[col]-DeltaPerCycleFromRef[self.StartCycle]],axis=1);
                DeltaPerCycle=DeltaPerCycle.rename(columns={0:col})
                DFdicPerClr[clr]=  DeltaPerCycle;       
                    
       
        
        return DFdicPerClr;
    
    




           
class CIScurveFromRawData:
    def  __init__(self, pthF): 
        self.pthF = pthF;
 

    
    def LoadRawData(self):
        lines=[];
        LogFile= self.pthF+'Data/JobData.csv';
        with open(LogFile, mode='r') as f:
           while 1:
               line = f.readline()
               if len(line) >1:   
                  lines.append(line)
               else:
                  if len(line)==0:   
                      break;
        return  lines;
    
    def GetCIScurveOldVersion(self):
        jobData=self.LoadRawData()
        sub='CisCurvatureDataBasedOnWaveFormat=';
        res = list(filter(lambda x: sub in x, jobData));
        cisFRONT=[]
        cisBACK=[]

        if len(res)>0:
            for i,rs in enumerate(res):
                if len(rs)> 1000:
                    tmp=rs.split(',')
                    tmp.pop(0);
                    for c in tmp:
                        if c.replace('.', '', 1).replace('-', '').isdigit():
                            if i==0:
                                cisFRONT.append(float(c))
                            else:
                                cisBACK.append(float(c))

        
        return cisBACK,cisFRONT;    
    
    def GetCIScurveNewVersion(self):
        jobData=self.LoadRawData()
        sub='ShouldUseCISCurvaturePerPixel=Value:True';
        res = list(filter(lambda x: sub in x, jobData));
        cisFRONT=[]
        cisBACK=[]
        
        if len(res)>0:
            sub='CISTilt=Value';
            res = list(filter(lambda x: sub in x, jobData));
            for i,rs in enumerate(res):
                ind=jobData.index(rs)
                if len(jobData[ind+1])> 1000:
                    tmp=jobData[ind+1].split(',')
                    # tmp=list(map(float, jobData[ind+1].split(',')))
                    for j,c in enumerate(tmp):                            
                        if c.replace('.', '', 1).replace('-', '').isdigit():
                            if i==0:
                                cisFRONT.append(float(c))
                            else:
                                cisBACK.append(float(c)) 
                     

        
        return cisBACK,cisFRONT;    
            
                    

                


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

LocatorIndexFront= CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).GetLocatorIndex();
# FlatList= CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).getNumberOfFlats();

cisBACK,cisFRONT=CIScurveFromRawData(pthF+'/').GetCIScurveOldVersion()

if len(cisFRONT) == 0:
    cisBACK,cisFRONT=CIScurveFromRawData(pthF+'/').GetCIScurveNewVersion()
    
    

if registrationBetweenWavePrints:
    DFdicPerClr =  CalcRegistrationFromWaveData(pthF+'/',side,Panel,ColorList,MainColor,StartCycle).DeltaForCycleAndColor()    

WaveRawDataDic={};
for ColorForDisplay in ColorList:  
    tmp=CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();
    tmp=pd.concat([tmp,tmp.loc[:,StartCycle4Avr:].mean(axis=1)],axis=1).rename(columns={0:'Mean'})
    WaveRawDataDic[ColorForDisplay]=tmp;



WaveDataWithMaxFilterDic={};

for ColorForDisplay in ColorList: 
    tmp=pd.DataFrame();
    for col in WaveRawDataDic[ColorForDisplay].columns:
        tmp=pd.concat([tmp,pd.Series(savgol_filter(WaveRawDataDic[ColorForDisplay][col], MaxWaveWindow, 1))],axis=1)
        tmp=tmp.rename(columns={0:col})
    WaveDataWithMaxFilterDic[ColorForDisplay]=tmp






### Calc PH location

PHloc=[]
PHloc.append(LocatorIndexFront-1)
numForward=LocatorIndexFront-1
numBack=LocatorIndexFront-1

for i in range(len(tmp['Mean'])):
    numForward=numForward+16;
    numBack=numBack-16;
    if numBack>0:
        PHloc.append(numBack);
    if numForward<len(tmp['Mean']):
        PHloc.append(numForward);

PHloc.sort()

#########################################
#########################################
#########################################
fig00 = go.Figure()
fig001 = go.Figure()
fig002 = go.Figure()
side='Front';
# fig00 = make_subplots(rows=3, cols=1,subplot_titles=("LeftSide","Middle", "RightSide"), vertical_spacing=0.1, shared_xaxes=True)

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
# for ColorForDisplay in ColorList:
for ColorForDisplay in ColorList:    
    db=CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();
    
    if ColorForDisplay=='Yellow':
        ColorForDisplay='gold'; 
        
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
if LeftSide:
    plot(fig00,filename=f+" Left Side WaveResult_RawDataPerPanel "+side+".html") 
if Middle:    
    plot(fig001,filename=f+" Middle Side WaveResult_RawDataPerPanel "+side+".html") 
if RightSide:
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
        
        if ColorForDisplay=='Yellow':
            ColorForDisplay='gold';
            
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
    
    if LeftSide:
        plot(fig000,filename=f+" Left Side WaveResult_RawDataPerPanel-Back.html") 
    if Middle:
        plot(fig011,filename=f+" Middle Side WaveResult_RawDataPerPanel-Back.html") 
    if RightSide:
        plot(fig022,filename=f+" Right Side WaveResult_RawDataPerPanel-Back.html") 
except:
    1    
    
#########################################
#########################################
#########################################
fig01 = go.Figure()
side='Front';
# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
# for ColorForDisplay in ColorList:
for ColorForDisplay in ColorList:
    db=CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();

    if ColorForDisplay=='Yellow':
        ColorForDisplay='gold';    
    
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
    
    
    for j,i in enumerate(Leftdb.index):
        if j == 0:
            continue;
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
fig01.update_layout(title=f+' STD Side Offset WAVE RAW DATA-'+side )

now = datetime.now()


dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# plot(fig00)
plot(fig01,filename=f+" STD SideOffset_ WaveResult_RawDataPerPanel "+side+".html") 

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
        
        if ColorForDisplay=='Yellow':
            ColorForDisplay='gold';
            
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
        
        
        for j,i in enumerate(Leftdb.index):
            if j == 0:
                continue;
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

#################PANEL##################
fig100 = go.Figure()
fig101 = go.Figure()
fig102 = go.Figure()
side='Front';
# fig00 = make_subplots(rows=3, cols=1,subplot_titles=("LeftSide","Middle", "RightSide"), vertical_spacing=0.1, shared_xaxes=True)

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
# for ColorForDisplay in ColorList:
for Panel in  range(1,12):  
    for ColorForDisplay in ColorList:
        db=CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();
        
        
        col=list(db.columns)
        
        rnge=range(len(col))
        
        if ColorForDisplay=='Yellow':
            ColorForDisplay='gold';
        
        # for i in rnge:
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
        offSet=db[CycleNumber][0];
        fig100.add_trace(
        go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
                    name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
        
        
        offSet=np.min(db[CycleNumber][int(len(db[CycleNumber])/2)-50:int(len(db[CycleNumber])/2)+50])
        fig101.add_trace(
        go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
                    name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
        
        offSet=db[CycleNumber][(len(db[CycleNumber]))-1]  
        fig102.add_trace(
        go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
                    name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))



fig100.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
fig101.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
fig102.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
fig100.update_layout(title=f+' Left Side Offset WAVE RAW DATA '+side)
fig101.update_layout(title=f+'Middle Side Offset WAVE RAW DATA '+side)
fig102.update_layout(title=f+'Right Side Offset WAVE RAW DATA '+side)

now = datetime.now()


dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# plot(fig00)
if LeftSide:
    plot(fig100,filename=f+" Left Side WaveResult_RawDataPerPanel "+side+".html") 
if Middle:    
    plot(fig101,filename=f+" Middle Side WaveResult_RawDataPerPanel "+side+".html") 
if RightSide:
    plot(fig102,filename=f+" Right Side WaveResult_RawDataPerPanel "+side+".html") 


########## BACK ########
try:
    fig110 = go.Figure()
    fig111 = go.Figure()
    fig122 = go.Figure()
    side = 'Back'
    # fig00 = make_subplots(rows=3, cols=1,subplot_titles=("LeftSide","Middle", "RightSide"), vertical_spacing=0.1, shared_xaxes=True)
    
    # rnge=[3,6,7]
    
    # db=ImagePlacement_Rightpp
    # db=ImagePlacement_pp
    # for ColorForDisplay in ColorList:
    for Panel in  range(1,12):  
        for ColorForDisplay in ColorList:
            db=CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).ArrangeRawDataForAnalize();
            
            
            col=list(db.columns)
            
            rnge=range(len(col))
            
            # for i in rnge:
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
    
            if ColorForDisplay=='Yellow':
                ColorForDisplay='gold';    
                
            offSet=db[CycleNumber][0];
            fig110.add_trace(
            go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
                        name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
            
            
            offSet=np.min(db[CycleNumber][int(len(db[CycleNumber])/2)-50:int(len(db[CycleNumber])/2)+50])
            fig111.add_trace(
            go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
                        name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
            
            offSet=db[CycleNumber][(len(db[CycleNumber]))-1]  
            fig112.add_trace(
            go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
                        name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))



    fig110.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    fig111.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    fig112.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    fig110.update_layout(title=f+' Left Side Offset WAVE RAW DATA '+side)
    fig111.update_layout(title=f+'Middle Side Offset WAVE RAW DATA '+side)
    fig112.update_layout(title=f+'Right Side Offset WAVE RAW DATA '+side)
    
    now = datetime.now()
    
    
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # plot(fig00)
    if LeftSide:
        plot(fig110,filename=f+" Left Side WaveResult_RawDataPerPanel "+side+".html") 
    if Middle:    
        plot(fig111,filename=f+" Middle Side WaveResult_RawDataPerPanel "+side+".html") 
    if RightSide:
        plot(fig112,filename=f+" Right Side WaveResult_RawDataPerPanel "+side+".html") 
 
except:
    1    
    
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
######CIS############
if CIScurve:
    try:
        fig012 = go.Figure()
    
                
        fig012.add_trace(
        go.Scatter(y=cisFRONT,
                    name='CIS FRONT curve'))
                
      
    
    
        fig012.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        fig012.update_layout(title=f+'CIS curve FRONT' )
        
        now = datetime.now()
        
        
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # plot(fig00)
        plot(fig012,filename=f+"CIScurveFRONT.html") 
    
    except:
     1
    if len(cisFRONT)<1:
        print('************************************************************************************')
        print(f+' Has No CIS FRONT curve information')
        print('************************************************************************************')
      
    ##### BACK
    try:
        fig013 = go.Figure()
    
                
        fig013.add_trace(
        go.Scatter(y=cisBACK,
                    name='CIS BACK curve'))
                
      
    
    
        fig013.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        fig013.update_layout(title=f+'CIS curve BACK' )
        
        now = datetime.now()
        
        
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # plot(fig00)
        plot(fig013,filename=f+"CIScurveBACK.html") 
    
    except:
     1
    if len(cisBACK)<1:
        print('************************************************************************************')
        print(f+' Has No CIS BACK curve information')
        print('************************************************************************************')    

 ##################################################################################################       
 ##################################################################################################       
 ##################################################################################################       
 ##################################################################################################       
if registrationBetweenWavePrints:
    for clr in ColorList:
        if clr == MainColor:
            continue;
        figClr = go.Figure()
        
        
        
        for col in DFdicPerClr[clr].columns:
            
            figClr.add_trace(
            go.Scatter(y=DFdicPerClr[clr][col],line_color= clr,
                        name='Registration for cycle '+str(col)+' color '+clr))
    
    
        figClr.update_layout(
                hoverlabel=dict(
                    namelength=-1
                )
            )
        figClr.update_layout(title=f+'Registration for '+clr )
        
        now = datetime.now()
        
        
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # plot(fig00)
        plot(figClr,filename=f+'Registration for '+clr+".html") 
        
##################################################################################################       
 ##################################################################################################       
 ##################################################################################################       
 ##################################################################################################          
if presentAllColors:
    for clr in ColorList:
        figPH = make_subplots(specs=[[{"secondary_y": True}]])
        for col in WaveRawDataDic[clr].columns:
            
            figPH.add_trace(
            go.Scatter(y=WaveRawDataDic[clr][col],line_color= clr,
                        name='WaveData Raw '+str(col)+' color '+clr), secondary_y=False)
            
            figPH.add_trace(
            go.Scatter(y=WaveDataWithMaxFilterDic[clr][col],line_color= clr,
                        name='WaveData with Filter '+str(col)+' color '+clr), secondary_y=False)
            
            figPH.add_trace(
            go.Scatter(y=WaveRawDataDic[clr][col]-WaveDataWithMaxFilterDic[clr][col],line_color= clr,
                        name='Fiter - Raw '+str(col)+' color '+clr), secondary_y=True)
        
        
        figPH.update_layout(
                hoverlabel=dict(
                    namelength=-1
                )
            )
        figPH.update_layout(title=f+'Wave Data - Filtered color '+clr+' Max Filter = '+ str(MaxWaveWindow))
        
        now = datetime.now()
        
        
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # plot(fig00)
        plot(figPH,filename=f+'Wave Data - Filtered color '+clr+' Max Filter_'+ str(MaxWaveWindow)+".html") 
 
##################################################################################################       
 ##################################################################################################       
 ##################################################################################################       
 ##################################################################################################          


figPH = make_subplots(specs=[[{"secondary_y": True}]])
col='Mean';
for clr in ColorList:     
    lineColor=clr;
    
    if lineColor=='Yellow':
        lineColor='gold';
    
    figPH.add_trace(
    go.Scatter(y=WaveRawDataDic[clr][col],line_color= lineColor,
                name='WaveData Raw '+str(col)+' color '+clr), secondary_y=False)
    
    figPH.add_trace(
    go.Scatter(y=WaveDataWithMaxFilterDic[clr][col],line_color= lineColor,
                name='WaveData with Filter '+str(col)+' color '+clr), secondary_y=False)
    
    figPH.add_trace(
    go.Scatter(y=WaveRawDataDic[clr][col]-WaveDataWithMaxFilterDic[clr][col],line_color= lineColor,
                name='Fiter - Raw '+str(col)+' color '+clr), secondary_y=True)
    
    
    for PHlocMem in PHloc:
        figPH.add_vline(x=PHlocMem, line_width=2, line_dash="dash", line_color="green")
    
    figPH.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
    figPH.update_layout(title=f+'Wave Data - Filtered color '+clr+' Max Filter = '+ str(MaxWaveWindow)+' LocatorIndexFront = '+str(LocatorIndexFront))
    
    now = datetime.now()
    
    
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # plot(fig00)
plot(figPH,filename=f+'Wave Data - Filtered color '+clr+' Max Filter_'+ str(MaxWaveWindow)+".html") 
 
 
 
  
 
 
 
 
 
 