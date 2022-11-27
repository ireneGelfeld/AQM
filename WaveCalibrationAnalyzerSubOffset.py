# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:59:12 2022

@author: Ireneg
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

#####################################Params #############################################################
#########################################################################################################
global StartCycle,StartCycle4Avr,PHpoitToIgnor,MaxWaveWindow,DistanceBtWPointMM

StartCycle=3
rgistBtwPntStartCycle=4
rgistBtwPntEndCycle=5

CycleNumber =3
StartCycle4Avr = 2;
PHpoitToIgnor=2;
MaxWaveWindow=51;
ColorLevels= 5;
DivideByNum= 20;

ColorLevelsTilt=3;
DivideByNumTilt=1;

DistanceBtWPointMM=2.734

Panel = 6;
ColorForDisplay = 'Cyan'
MainColor = "Black"

LeftSide=0;
Middle=0;
RightSide=0;
CIScurve=1;
DisplayOffSet=1;
DisplayTilt=1;
registrationBetweenWavePrints=0;
presentAllColors=0


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
from plotly.colors import n_colors
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.subplots import make_subplots


class CalcWaveFromRawData:
    def  __init__(self, pthF,side,Panel): 
        self.pthF = pthF;
        self.side = side;
        self.Panel= Panel;
        

    
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
    
    def FilterRawData(self,ColorForDisplay):
        RawData= self.LoadRawData();
        
        DataSec=RawData[RawData['Overall Status']=='Success'].reset_index(drop=True);

        DataSecPrintDirc=DataSec[DataSec['Direction Type ']=='Print Direction']
        
        DataSecPrintDircPanel=DataSecPrintDirc[DataSecPrintDirc['Panel Id']==Panel]
        
        DataSecPrintDircPanelColor=DataSecPrintDircPanel[DataSecPrintDircPanel[' Seperation']==ColorForDisplay].reset_index(drop=True)
        
        col=list(DataSecPrintDircPanelColor.columns)
        
        cutCols=col[12:396]
        
        DataSecPrintDircPanelColorCUT=DataSecPrintDircPanelColor[cutCols];
        LocatorIndex= DataSec['Locator Index'][0];
        
        return LocatorIndex,DataSecPrintDircPanelColorCUT,cutCols;
    
    
    def ArrangeRawDataForAnalize(self,ColorForDisplay):
       
        LocatorIndex,DataSecPrintDircPanelColorCUT,cutCols=self.FilterRawData(ColorForDisplay);
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

    def GetLocatorIndex(self,ColorForDisplay):
        LocatorIndex,DataSecPrintDircPanelColorCUT,cutCols=self.FilterRawData(ColorForDisplay);
        return LocatorIndex;
    
    def CreateDicOfWaveRawData(self):
        ColorList=self.getColors();
        WaveRawDataDic={};
        for ColorForDisplay in ColorList:  
            tmp=self.ArrangeRawDataForAnalize(ColorForDisplay);
            tmp=pd.concat([tmp,tmp.loc[:,StartCycle4Avr:].mean(axis=1)],axis=1).rename(columns={0:'Mean'})
            WaveRawDataDic[ColorForDisplay]=tmp;
        return WaveRawDataDic;
    
    def FilterWaveDataDic(self):
        ColorList=self.getColors();
        WaveRawDataDic=self.CreateDicOfWaveRawData();
        WaveDataWithMaxFilterDic={};

        for ColorForDisplay in ColorList: 
            tmp=pd.DataFrame();
            for col in WaveRawDataDic[ColorForDisplay].columns:
                tmp=pd.concat([tmp,pd.Series(savgol_filter(WaveRawDataDic[ColorForDisplay][col], MaxWaveWindow, 1))],axis=1)
                tmp=tmp.rename(columns={0:col})
            WaveDataWithMaxFilterDic[ColorForDisplay]=tmp
        return WaveDataWithMaxFilterDic;

    def CalcPHlocation(self,ColorForDisplay):
        
        LocatorIndex= self.GetLocatorIndex(ColorForDisplay)
        WaveRawDataDic=self.CreateDicOfWaveRawData();
        PHloc=[]
        PHloc.append(LocatorIndex-1)
        numForward=LocatorIndex-1
        numBack=LocatorIndex-1

        for i in range(len(WaveRawDataDic[ColorForDisplay]['Mean'])):
            numForward=numForward+16;
            numBack=numBack-16;
            if numBack>0:
                PHloc.append(numBack);
            if numForward<len(WaveRawDataDic[ColorForDisplay]['Mean']):
                PHloc.append(numForward);
        
        PHloc.sort()
        return PHloc;
    
    
    
    

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
        
        mainColorRef=CalcWaveFromRawData(self.pthF+'/',self.side,self.Panel).ArrangeRawDataForAnalize(self.MainColor);
        DFdicPerClr={}
        for clr in self.ColorList:
            if clr == self.MainColor:
                continue;
            DeltaPerCycleFromRef=pd.DataFrame();
            DeltaPerCycle=pd.DataFrame();    
            ColorWavePerCycle=CalcWaveFromRawData(self.pthF+'/',self.side,self.Panel).ArrangeRawDataForAnalize(clr);
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
            
def CalcMeanAndTilt(WaveRawDataDic,WaveDataWithMaxFilterDic,PHloc):
    PHoffSet={}
    PHtilt={}
    
    PHoffsetPerH={}
    PHtiltPerH={}
    
    for ColorForDisplay in ColorList: 
        y=WaveRawDataDic[ColorForDisplay]['Mean']-WaveDataWithMaxFilterDic[ColorForDisplay]['Mean'];
        t=list(y);
        tlt=t.copy();
        PHoffsetPerHList=[]
        PHtiltPerHList=[]
        # x=[]
        # tlt1=[]
        for i in range(len(PHloc)+1):
        # for i in range(9):

            if i==0:
                PHrangeForCalc=slice(PHloc[0]-PHpoitToIgnor);
                indexSlice=slice(PHloc[0])
                PHrange=abs(PHloc[0]);
            else:
                if i== len(PHloc):
                    PHrangeForCalc=slice(PHloc[i-1]+PHpoitToIgnor,len(y));
                    indexSlice=slice(PHloc[i-1],len(y))
                    PHrange=abs(len(y)-PHloc[i-1]);
                    # break;

                else:
                
                    PHrangeForCalc=slice(PHloc[i-1]+PHpoitToIgnor,PHloc[i]-PHpoitToIgnor+1);
                    indexSlice=slice(PHloc[i-1],PHloc[i])
                    PHrange=abs(PHloc[i]-PHloc[i-1]); 
            
            Points=y[PHrangeForCalc].index*DistanceBtWPointMM
            PHoffsetPerHList.append(int(np.mean(y[PHrangeForCalc])))
            z=np.polyfit(list(Points), list(y[PHrangeForCalc]), 1)
            tlt[PHrangeForCalc]=list(z[0]*(Points)+z[1])
            t[indexSlice]=[np.mean(y[PHrangeForCalc])]*PHrange;
            PHtiltPerHList.append((z[0]))
            # x=x+list(y[PHrangeForCalc].index)
            # tlt1=tlt1+tlt[PHrangeForCalc]
    
                
        PHoffSet[ColorForDisplay]=t
        PHtilt[ColorForDisplay]=tlt
        ## For Table plot ##
        PHoffsetPerH[ColorForDisplay]=PHoffsetPerHList
        PHtiltPerH[ColorForDisplay]=PHtiltPerHList
        
    return PHoffSet,PHtilt,PHoffsetPerH,PHtiltPerH                    

# plt.figure()
# plt.plot(y)
# plt.plot(tlt)
# plt.plot(x,tlt1,'o')


                
# WaveRawDataDic=WaveRawDataDicFRONT;
# WaveDataWithMaxFilterDic=WaveDataWithMaxFilterDicFRONT;
# PHloc=PHlocFRONT;

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

ColorList= CalcWaveFromRawData(pthF+'/',side,Panel).getColors();

LocatorIndex= CalcWaveFromRawData(pthF+'/',side,Panel).GetLocatorIndex(ColorForDisplay);
# FlatList= CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).getNumberOfFlats();

cisBACK,cisFRONT=CIScurveFromRawData(pthF+'/').GetCIScurveOldVersion()

if len(cisFRONT) == 0:
    cisBACK,cisFRONT=CIScurveFromRawData(pthF+'/').GetCIScurveNewVersion()
    
    

if registrationBetweenWavePrints:
    DFdicPerClrFRONT =  CalcRegistrationFromWaveData(pthF+'/',side,Panel,ColorList,MainColor,StartCycle).DeltaForCycleAndColor() 
    try:
        DFdicPerClrBACK =  CalcRegistrationFromWaveData(pthF+'/','Back',Panel,ColorList,MainColor,StartCycle).DeltaForCycleAndColor() 
    except:
        1


WaveRawDataDicFRONT=CalcWaveFromRawData(pthF+'/',side,Panel).CreateDicOfWaveRawData();
WaveDataWithMaxFilterDicFRONT=CalcWaveFromRawData(pthF+'/',side,Panel).FilterWaveDataDic()
PHlocFRONT= CalcWaveFromRawData(pthF+'/',side,Panel).CalcPHlocation(ColorForDisplay)
try:
    WaveRawDataDicBACK=CalcWaveFromRawData(pthF+'/','Back',Panel).CreateDicOfWaveRawData();
    WaveDataWithMaxFilterDicBACK=CalcWaveFromRawData(pthF+'/','Back',Panel).FilterWaveDataDic()
    PHlocBACK= CalcWaveFromRawData(pthF+'/','Back',Panel).CalcPHlocation(ColorForDisplay)

except:
    1





################ Calc offset and tilt

PHoffSetFRONT,PHtiltFRONT,PHoffsetPerHFRONT,PHtiltPerHFRONT=CalcMeanAndTilt(WaveRawDataDicFRONT,WaveDataWithMaxFilterDicFRONT,PHlocFRONT)

try:
   PHoffSetBACK,PHtiltBACK,PHoffsetPerHBACK,PHtiltPerHBACK=CalcMeanAndTilt(WaveRawDataDicBACK,WaveDataWithMaxFilterDicBACK,PHlocBACK)
except:
    1
 

# x=range(12)  
# y1=y[PHloc[i-1]+2:PHloc[i]-2]      
# z = np.polyfit(x, y1, 3)

# yy=z[0]*x+z[1]
# plt.figure();
# plt.plot(x,y1)
# plt.plot(x,yy)
# plt.show()

# y=WaveRawDataDicFRONT[clr][col]-WaveDataWithMaxFilterDicFRONT[clr][col];
# t=list(y);
# tlt=t.copy();
# for i in range(1,len(PHloc)):
#     for j in range(PHloc[i-1],PHloc[i]):
#         t[j]=np.mean(y[PHloc[i-1]+2:PHloc[i]-2])
#         tlt[PHloc[i-1]+2:PHloc[i]-2]=savgol_filter(y[PHloc[i-1]+2:PHloc[i]-2], 11, 1)


# plt.figure();
# plt.plot(t)
# plt.show()

# plt.figure();
# plt.plot(tlt)
# plt.show()

#########################################
#########################################
#########################################
if LeftSide+Middle+RightSide:

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
        db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        
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
    fig00.update_layout(title=side+'- Left Side Offset WAVE RAW DATA --->'+f)
    fig001.update_layout(title=side+'- Middle Side Offset WAVE RAW DATA --->'+f)
    fig002.update_layout(title=side+'- Right Side Offset WAVE RAW DATA --->'+f)
    
    now = datetime.now()
    
    
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # plot(fig00)
    if LeftSide:
        plot(fig00,filename=f+" Left Side WaveResult_RawDataPerCycle "+side+".html") 
    if Middle:    
        plot(fig001,filename=f+" Middle Side WaveResult_RawDataPerCycle "+side+".html") 
    if RightSide:
        plot(fig002,filename=f+" Right Side WaveResult_RawDataPerCycle "+side+".html") 
    
    
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
            db=CalcWaveFromRawData(pthF+'/','Back',Panel).ArrangeRawDataForAnalize(ColorForDisplay);
            
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
        
        fig000.update_layout(title=side+'- Left Side Offset WAVE RAW DATA --->'+f)
        fig011.update_layout(title=side+'- Middle Side Offset WAVE RAW DATA --->'+f)
        fig022.update_layout(title=side+'- Right Side Offset WAVE RAW DATA --->'+f)
    
        
        now = datetime.now()
        
        
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        
        if LeftSide:
            plot(fig000,filename=f+' Left Side WaveResult_RawDataPerCycle'+side+".html") 
        if Middle:
            plot(fig011,filename=f+' Middle Side WaveResult_RawDataPerCycle'+side+".html") 
        if RightSide:
            plot(fig022,filename=f+' Right Side WaveResult_RawDataPerCycle'+side+".html") 
    except:
        1    
    
#########################################
#########################################
#########################################

if LeftSide+Middle+RightSide:

    fig01 = go.Figure()
    side='Front';
    # rnge=[3,6,7]
    
    # db=ImagePlacement_Rightpp
    # db=ImagePlacement_pp
    # for ColorForDisplay in ColorList:
    for ColorForDisplay in ColorList:
        db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    
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
    fig01.update_layout(title=side+'- <b>STD </b> Side Offset WAVE RAW DATA --->'+f )
    
    now = datetime.now()
    
    
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    if LeftSide or Middle or RightSide:
    # plot(fig00)
        plot(fig01,filename=f+" STD SideOffset_ WaveResult_RawDataPerColor "+side+".html") 
    
    ################# Back  ########################
    try:
        fig010 = go.Figure()
        side='Back';
    # rnge=[3,6,7]
    
    # db=ImagePlacement_Rightpp
    # db=ImagePlacement_pp
    # for ColorForDisplay in ColorList:
        for ColorForDisplay in ColorList:
            db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
            
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
        fig010.update_layout(title=side+'- <b>STD </b> Side Offset WAVE RAW DATA --->'+f)
        
        now = datetime.now()
        
        
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # plot(fig00)
        if LeftSide or Middle or RightSide:
    
            plot(fig010,filename=f+"STD SideOffset_ WaveResult_RawDataPerColor "+side+".html") 
    except:
        1    
#########################################
#########################################
#########################################

#################PANEL##################
if LeftSide+Middle+RightSide:

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
            db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
            
            
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
    fig100.update_layout(title=side+'- Left Side Offset WAVE RAW DATA (For one Cycle)--->'+f)
    fig101.update_layout(title=side+'- Middle Offset WAVE RAW DATA (For one Cycle)--->'+f)
    fig102.update_layout(title=side+'- Right Side Offset WAVE RAW DATA (For one Cycle)--->'+f)
    
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
                db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
                
                
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
        fig110.update_layout(title=side+'- Left Side Offset WAVE RAW DATA (For one Cycle)--->'+f)
        fig111.update_layout(title=side+'- Middle Offset WAVE RAW DATA (For one Cycle)--->'+f)
        fig112.update_layout(title=side+'- Right Side Offset WAVE RAW DATA (For one Cycle)--->'+f)
        
 
        
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
        fig012.update_layout(title='FRONT CIS curve --->'+f )
        
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
        fig013.update_layout(title='BACK CIS curve --->'+f)
        
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
try:
    if registrationBetweenWavePrints:
        
        figClrFRONT = go.Figure()
            
        side='Front'    
        for clr in ColorList:
            if clr == MainColor:
                continue;
            
            for col in range(rgistBtwPntStartCycle,rgistBtwPntEndCycle+1):        
            # for col in DFdicPerClr[clr].columns:
                
                figClrFRONT.add_trace(
                go.Scatter(y=DFdicPerClrFRONT[clr][col],line_color= clr,
                            name='Registration for cycle '+str(col)+' color '+clr))
        
        
            figClrFRONT.update_layout(
                    hoverlabel=dict(
                        namelength=-1
                    )
                )
            figClrFRONT.update_layout(title=side+' wave registration normalized to '+MainColor+' for Cycle Start ='+str(rgistBtwPntStartCycle)+' Cycle End='+str(rgistBtwPntEndCycle)+' ---> '+f)
            
        now = datetime.now()
        
        
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            # plot(fig00)
        plot(figClrFRONT,filename=f+'Registration for cycle '+str(rgistBtwPntStartCycle)+'_'+str(rgistBtwPntEndCycle)+'_'+side+".html") 
        
        
        ####BACK
        
        figClrBACK = go.Figure()
            
        side='Back'    
        for clr in ColorList:
            if clr == MainColor:
                continue;
            
            for col in range(rgistBtwPntStartCycle,rgistBtwPntEndCycle+1):        
            # for col in DFdicPerClr[clr].columns:
                
                figClrBACK.add_trace(
                go.Scatter(y=DFdicPerClrBACK[clr][col],line_color= clr,
                            name='Registration for cycle '+str(col)+' color '+clr))
        
        
            figClrBACK.update_layout(
                    hoverlabel=dict(
                        namelength=-1
                    )
                )
            figClrBACK.update_layout(title=side+' wave registration normalized to '+MainColor+' for Cycle Start ='+str(rgistBtwPntStartCycle)+' Cycle End='+str(rgistBtwPntEndCycle)+' ---> '+f)
            
        now = datetime.now()
        
        
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            # plot(fig00)
        plot(figClrBACK,filename=f+'Registration for cycle '+str(rgistBtwPntStartCycle)+'_'+str(rgistBtwPntEndCycle)+'_'+side+".html") 
except:
    1      
   
##################################################################################################       
 ##################################################################################################       
 ##################################################################################################       
 ##################################################################################################          
if presentAllColors:
    for clr in ColorList:
        figPH = make_subplots(specs=[[{"secondary_y": True}]])
        for col in WaveRawDataDicFRONT[clr].columns:
            
            figPH.add_trace(
            go.Scatter(y=WaveRawDataDicFRONT[clr][col],line_color= clr,
                        name='WaveData Raw '+str(col)+' color '+clr), secondary_y=False)
            
            figPH.add_trace(
            go.Scatter(y=WaveDataWithMaxFilterDicFRONT[clr][col],line_color= clr,
                        name='WaveData with Filter '+str(col)+' color '+clr), secondary_y=False)
            
            figPH.add_trace(
            go.Scatter(y=WaveRawDataDicFRONT[clr][col]-WaveDataWithMaxFilterDicFRONT[clr][col],line_color= clr,
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
side='Front'
for clr in ColorList:     
    lineColor=clr;
  
    
    if lineColor=='Yellow':
        lineColor='gold';
    
    figPH.add_trace(
    go.Scatter(y=WaveRawDataDicFRONT[clr][col],line_color= lineColor,
                name='WaveData Raw '+str(col)+' color '+clr), secondary_y=False)
    
    figPH.add_trace(
    go.Scatter(y=WaveDataWithMaxFilterDicFRONT[clr][col],line_color= lineColor,
                name='WaveData with Filter '+str(col)+' color '+clr), secondary_y=False)
    
    figPH.add_trace(
    go.Scatter(y=WaveRawDataDicFRONT[clr][col]-WaveDataWithMaxFilterDicFRONT[clr][col],line_color= lineColor,
                name='Fiter - Raw '+str(col)+' color '+clr), secondary_y=True)
    
    ymax=max(WaveRawDataDicFRONT[ColorList[0]][col]-WaveDataWithMaxFilterDicFRONT[ColorList[0]][col])
    
    for i,PHlocMem in enumerate(PHlocFRONT):
        figPH.add_trace(go.Scatter(x=[PHlocMem], y=[ymax],
                                marker=dict(color="green", size=6),
                                mode="markers",
                                text='PH #'+str(i),
                                # font_size=18,
                                hoverinfo='text'),secondary_y=True)
        figPH.data[len(figPH.data)-1].showlegend = False

        figPH.add_vline(x=PHlocMem, line_width=2, line_dash="dash", line_color="green")
    
    
    if DisplayOffSet:
        figPH.add_trace(
        go.Scatter(y=PHoffSetFRONT[clr],line_color= lineColor,
                    name='Average(Fiter - Raw) '+str(col)+' color '+clr), secondary_y=True)
        
    
    if DisplayTilt:
        figPH.add_trace(
        go.Scatter(y=PHtiltFRONT[clr],line_color= lineColor,line=dict(dash='dot'),
                    name='Tilt(Fiter - Raw) '+str(col)+' color '+clr), secondary_y=True)
    
    
    figPH.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
    figPH.update_layout(title=side+' Wave Data S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
    
    now = datetime.now()
    
    
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # plot(fig00)
plot(figPH,filename=f+' '+side+' Wave Data S.Golay _'+ str(MaxWaveWindow)+".html") 
 
## Back ##
try:
    figPHBACK = make_subplots(specs=[[{"secondary_y": True}]])
    col='Mean';
    side='Back'
    for clr in ColorList:     
        lineColor=clr;
        
        if lineColor=='Yellow':
            lineColor='gold';
        
        figPHBACK.add_trace(
        go.Scatter(y=WaveRawDataDicBACK[clr][col],line_color= lineColor,
                    name='WaveData Raw '+str(col)+' color '+clr), secondary_y=False)
        
        figPHBACK.add_trace(
        go.Scatter(y=WaveDataWithMaxFilterDicBACK[clr][col],line_color= lineColor,
                    name='WaveData with Filter '+str(col)+' color '+clr), secondary_y=False)
        
        figPHBACK.add_trace(
        go.Scatter(y=WaveRawDataDicBACK[clr][col]-WaveDataWithMaxFilterDicBACK[clr][col],line_color= lineColor,
                    name='Fiter - Raw '+str(col)+' color '+clr), secondary_y=True)
        
        
        for PHlocMem in PHlocBACK:
            figPHBACK.add_vline(x=PHlocMem, line_width=2, line_dash="dash", line_color="green")
        
        
        if DisplayOffSet:
            figPHBACK.add_trace(
            go.Scatter(y=PHoffSetBACK[clr],line_color= lineColor,
                        name='Average(Fiter - Raw) '+str(col)+' color '+clr), secondary_y=True)
        
        if DisplayTilt:
            figPHBACK.add_trace(
            go.Scatter(y=PHtiltBACK[clr],line_color= lineColor,line=dict(dash='dot'),
                        name='Tilt(Fiter - Raw) '+str(col)+' color '+clr), secondary_y=True)
        
        
        figPHBACK.update_layout(
                hoverlabel=dict(
                    namelength=-1
                )
            )
        figPHBACK.update_layout(title=side+' Wave Data S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
        
        now = datetime.now()
        
        
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # plot(fig00)
    plot(figPHBACK,filename=f+' '+side+' Wave Data S.Golay _'+ str(MaxWaveWindow)+".html") 
except:
    1    
 
 
##################################################################################################       
##################################################################################################       
##################################################################################################       
##################################################################################################      

PHname=[]
header=[]
ListofListFRONT=[]
ListofListBACK=[]

headerTilt=[]
ListofListTiltFRONT=[]
ListofListTiltBACK=[]

side='Front'
for i in range(24):
    PHname.append('PH NUMBER# '+str(i))

for col in ColorList:
    header.append(col+' Offset')
    # header.append(col+' Tilt')
    new_list = [-number for number in PHoffsetPerHFRONT[col]]
    ListofListFRONT.append(new_list)
    # ListofList.append(PHtiltPerH[col])
####FRONT 
figTableFRONT = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
                 cells=dict(values=[PHname]+ListofListFRONT,font=dict(color='black', size=15)))
                     ])

figTableFRONT.update_layout(title=side+' offset (Correction-For simplex) table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)

plot(figTableFRONT,filename=f+" Offset Table FRONT.html") 
####BACK
 
try:
    side='Back'
    new_list=[]
    for col in ColorList:
        new_list = [-number for number in PHoffsetPerHBACK[col]]

        ListofListBACK.append(new_list)
        
    figTableBACK = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
                 cells=dict(values=[PHname]+ListofListBACK,font=dict(color='black', size=15)))
                     ])
    figTableBACK.update_layout(title=side+' offset table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)


    plot(figTableBACK,filename=f+" Offset Table BACK.html") 
except:
    1


### Tilt
side='Front'
headerTilt=[]
ListofListTiltFRONT=[]
ListofListTiltBACK=[]

for col in ColorList:
    headerTilt.append(col+' Tilt')
    # header.append(col+' Tilt')
    ListofListTiltFRONT.append(PHtiltPerHFRONT[col])

backGroundCLR='rgb(200, 200, 200)'
colors = n_colors(backGroundCLR, 'rgb(200, 0, 0)', ColorLevelsTilt, colortype='rgb')
fillcolorList=[]
for i in range(len(ListofListTiltFRONT)):
    fillcolorList.append(np.array(colors)[(abs(np.asarray(ListofListTiltFRONT[i]))/DivideByNumTilt).astype(int)])
    

####FRONT Tilt
figTableTiltFRONT = go.Figure(data=[go.Table(header=dict(values=['PH#']+headerTilt),
                 cells=dict(values=[PHname]+ListofListTiltFRONT,fill_color=[backGroundCLR]+fillcolorList,font=dict(color='black', size=15),format=["0.2f"]))
                     ])

figTableTiltFRONT.update_layout(title=side+' Tilt table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)

plot(figTableTiltFRONT,filename=f+" Tilt Table FRONT.html") 

####BACK Tilt


try:
    side='Back'
    # headerTilt=[]
    for col in ColorList:
        # headerTilt.append(col+' Tilt')
        # header.append(col+' Tilt')
        ListofListTiltBACK.append(PHtiltPerHBACK[col])
        
    fillcolorList=[]
    for i in range(len(ListofListTiltBACK)):
        fillcolorList.append(np.array(colors)[(abs(np.asarray(ListofListTiltBACK[i]))/DivideByNumTilt).astype(int)]) 
        
    figTableTiltBACK = go.Figure(data=[go.Table(header=dict(values=['PH#']+headerTilt),
                 cells=dict(values=[PHname]+ListofListTiltBACK,fill_color=[backGroundCLR]+fillcolorList,font=dict(color='black', size=15),format=["0.2f"]))
                     ])
    
    figTableTiltBACK.update_layout(title=side+' Tilt table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
    
    plot(figTableTiltBACK,filename=f+" Tilt Table BACK.html")  
except:
    1    



 
#### FRONT -BACK delta
 

try:
    ListofListDelta=[]    
    header=[]
    fillcolorList=[]  
    backGroundCLR='rgb(200, 200, 200)'
    colors = n_colors(backGroundCLR, 'rgb(200, 0, 0)', ColorLevels, colortype='rgb')

    for col in ColorList:
        header.append(col+'Delta(Front-Back) Offset')
    for col in ColorList:
        ListofListDelta.append(list(np.asarray(PHoffsetPerHFRONT[col])-np.asarray(PHoffsetPerHBACK[col])))
        
    for i in range(len(ListofListDelta)):
        # x2 = 30 * np.ones(len(ListofListDelta[i]))
        fillcolorList.append(np.array(colors)[(abs(np.asarray(ListofListDelta[i]))/DivideByNum).astype(int)])
    
    
        
    figTableDelta = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
                 cells=dict(values=[PHname]+ListofListDelta,fill_color=[backGroundCLR]+fillcolorList,font=dict(color='black', size=15)))
                     ])
    figTableDelta.update_layout(title='Delta offset table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)


    plot(figTableDelta,filename=f+" Delta Offset Table.html") 
except:
    1



#### FRONT -BACK Average
 

try:
    ListofListAverage=[]    
    header=[]
    fillcolorList=[]  
    

    for col in ColorList:
        header.append(col+'Average(Front&Back) Offset')
    for col in ColorList:
        ListofListAverage.append(list(-(np.asarray(PHoffsetPerHFRONT[col])+np.asarray(PHoffsetPerHBACK[col]))/2))
        
  
    
    
        
    figTableDelta = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
                 cells=dict(values=[PHname]+ListofListAverage,font=dict(color='black', size=15)))
                     ])
    figTableDelta.update_layout(title='Correction table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)


    plot(figTableDelta,filename=f+" Correction Table.html") 
except:
    1


# cells=dict(
#     values=[a, b, c],
#     line_color=[np.array(colors)[a],np.array(colors)[b], np.array(colors)[c]],
#     fill_color=[np.array(colors)[a],np.array(colors)[b], np.array(colors)[c]],
#     align='center', font=dict(color='white', size=11)
#     ))



 