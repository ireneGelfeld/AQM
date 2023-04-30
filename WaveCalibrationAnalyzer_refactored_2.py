# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:59:12 2022

@author: Ireneg
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import plotly.express as px

#####################################Params #############################################################
#########################################################################################################
global StartCycle,StartCycle4Avr,PHpoitToIgnor,MaxWaveWindow,DistanceBtWPointMM,Panel,Cycle2Display,Panel2Disply,colorPNL,MaxMinPcnt



## for plot per panel and plot per cycle and WaveData SUB Average_PerPanel_PerCycle
plotPerPanel=0;# On/OFF plot
plotPerCycle=0;## On/OFF plot
WaveDataSUBAverage_PerPanel_PerCycle=1 # On/OFF plot (Avi method)
CycleNumber =3 # cycle view in => plot Per Panel
StartCycle4Avr = 2; # Start averaging for all plots defult = 2
Panel = 6;          #view panel for plot Per cycle
ColorForDisplay = 'Cyan' # Not in use
Cycle2Display = 4 # defult visible cycle in plot WaveDataSUBAverage_PerPanel_PerCycle
Panel2Disply= [11,6]
MaxMinPcnt=90 # %
colorPNL=px.colors.sequential.Reds[2:]+px.colors.sequential.Viridis;



## for plot CIScurve
CIScurve=1;# On/OFF plot

## for plot registration estimation in Wave Prints (yuval)
registrationBetweenWavePrints=0; # On/OFF plot ERROR
StartCycle=3
rgistBtwPntStartCycle=StartCycle # (it is not a parameter)
rgistBtwPntEndCycle=StartCycle+1 # for long print can change to larger number
MainColor = "Black" #Referance Color

##  Wave plot ( before and after correction)
BeforAndAfterCorr=1# On/OFF plot


## DX plot - delta between wave and starvitzky filer (residue) 
WaveFilterResidue_dxPlot=0 # On/OFF plot
PHpoitToIgnor=2; # Ponits of Print head to ignar (16 point in total) in each side
MaxWaveWindow=51;# S.gol filter window
S_g_Degree=1;# S.gol filter degree
DistanceBtWPointMM=2.734
NieghborColorsFor7colrs=4# parameter for distortion correction (number of nighboring colors)


###for Tables
PlotTables=0# On/OFF table
ColorLevels= 5; # Heat Map for offset- number of levels of colors from white to hot red
DivideByNum= 20; # Correction for offset Haet map- if occurs error try to increase this number
ColorLevelsTilt=7; #Heat Map for tilt- number of levels of colors from white to hot red
DivideByNumTilt=1;# Correction for tilt Haet map- if occurs error try to increase this number

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
import zipfile
import csv
from io import BytesIO

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.subplots import make_subplots
import math 


class CalcWaveFromRawData:
    def  __init__(self, pthF,side,Panel): 
        self.pthF = pthF;
        self.side = side;
        self.Panel= Panel;
        
    def LoadRawDataOLD(self):
        RawData=pd.read_csv(self.pthF+self.side+'/'+'RawResults/WavePrintDirection.csv');

        return  RawData;
    
    
    def GetFileFromZip(self,zip_file_path,subdir_name_in_zip,file_name_in_zip):
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_path_in_zip = subdir_name_in_zip + "/" + file_name_in_zip
            with zip_ref.open(file_path_in_zip) as file:
                # read the contents of the file into memory
                file_content = file.read()
                
                # convert the file content to a pandas dataframe
                df = pd.read_csv(BytesIO(file_content))
        return  df;     
    
    
    def LoadRawData(self):
        
        zip_file_path=self.pthF
        subdir_name_in_zip=self.side+'/'+'RawResults'; 
        file_name_in_zip='WavePrintDirection.csv'
        
        # RawData=pd.read_csv(self.pthF+self.side+'/'+'RawResults/WavePrintDirection.csv');
        RawData=self.GetFileFromZip(zip_file_path, subdir_name_in_zip, file_name_in_zip)
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
        
        # DataSec=RawData[RawData['Overall Status']=='Success'].reset_index(drop=True);
        
        DataSec=RawData;


        DataSecPrintDirc=DataSec[DataSec['Direction Type ']=='Print Direction']
        
        DataSecPrintDircPanel=DataSecPrintDirc[DataSecPrintDirc['Panel Id']==self.Panel]
        
        DataSecPrintDircPanelColor=DataSecPrintDircPanel[DataSecPrintDircPanel[' Seperation']==ColorForDisplay].reset_index(drop=True)
        
        col=list(DataSecPrintDircPanelColor.columns)
        
        cutCols=col[12:396]
        
        DataSecPrintDircPanelColorCUT=DataSecPrintDircPanelColor[cutCols];
        LocatorIndex= DataSec['Locator Index'][int(len(DataSec['Locator Index'])/2)];
        
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
                           if isinstance(l[tmp+1], float):
                               l[tmp]=l[tmp+1]
                           else:
                               if l[tmp+1].replace('.', '', 1).replace('-', '').isdigit():
                                  l[tmp]=float(l[tmp+1])
                               else:
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
                tmp=pd.concat([tmp,pd.Series(savgol_filter(WaveRawDataDic[ColorForDisplay][col], MaxWaveWindow, S_g_Degree))],axis=1)
                tmp=tmp.rename(columns={0:col})
            WaveDataWithMaxFilterDic[ColorForDisplay]=tmp
        return WaveDataWithMaxFilterDic;

    def FilterWaveDataDicTEST(self,WaveRawDataDic):
        ColorList=self.getColors();
        # WaveRawDataDic=self.CreateDicOfWaveRawData();
        WaveDataWithMaxFilterDic={};

        for ColorForDisplay in ColorList: 
            tmp=pd.DataFrame();
            for col in WaveRawDataDic[ColorForDisplay].columns:
                WaveRawDataDic[ColorForDisplay][col]=WaveRawDataDic[ColorForDisplay][col].fillna(method='ffill')
                tmp=pd.concat([tmp,pd.Series(savgol_filter(WaveRawDataDic[ColorForDisplay][col], MaxWaveWindow, S_g_Degree))],axis=1)
                tmp=tmp.rename(columns={0:col})
            WaveDataWithMaxFilterDic[ColorForDisplay]=tmp
        return WaveDataWithMaxFilterDic;

    def CalcPHlocation(self,ColorForDisplay):
        
        LocatorIndex= self.GetLocatorIndex(ColorForDisplay)
        WaveRawDataDic=self.CreateDicOfWaveRawData();
        PHloc=[]
        PHloc.append(LocatorIndex)
        numForward=LocatorIndex
        numBack=LocatorIndex

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
        
        mainColorRef=CalcWaveFromRawData(self.pthF,self.side,self.Panel).ArrangeRawDataForAnalize(self.MainColor);
        DFdicPerClr={}
        for clr in self.ColorList:
            if clr == self.MainColor:
                continue;
            DeltaPerCycleFromRef=pd.DataFrame();
            DeltaPerCycle=pd.DataFrame();    
            ColorWavePerCycle=CalcWaveFromRawData(self.pthF,self.side,self.Panel).ArrangeRawDataForAnalize(clr);
            DeltaPerCycleFromRef= mainColorRef.loc[:,self.StartCycle:]-ColorWavePerCycle.loc[:,self.StartCycle:];
            for col in DeltaPerCycleFromRef.loc[:,StartCycle:].columns:
                DeltaPerCycle=pd.concat([DeltaPerCycle,DeltaPerCycleFromRef[col]-DeltaPerCycleFromRef[self.StartCycle]],axis=1);
                DeltaPerCycle=DeltaPerCycle.rename(columns={0:col})
                DFdicPerClr[clr]=  DeltaPerCycle;       
                    
       
        
        return DFdicPerClr;
    
    




           
class CIScurveFromRawData:
    def  __init__(self, pthF): 
        self.pthF = pthF;
 

    
    def LoadRawDataOLD(self):
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
 
    def LoadRawData(self):
        
        
        zip_file_path=self.pthF
        subdir_name_in_zip='Data'
        file_name_in_zip='JobData.csv'
        
        lines = self.GetFileFromZip(zip_file_path, subdir_name_in_zip, file_name_in_zip)
       
        return  lines;   
    
    def GetFileFromZip(self,zip_file_path,subdir_name_in_zip,file_name_in_zip):
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_path_in_zip = subdir_name_in_zip + "/" + file_name_in_zip
            with zip_ref.open(file_path_in_zip) as file:
                # read the contents of the file into memory
                lines = [row for row in csv.reader(file.read().decode("utf-8").splitlines())]
                
                # convert the file content to a pandas dataframe
                # df = pd.read_csv(BytesIO(file_content))
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
    
    
    def GetCIScurveOldVersion_SecondTry(self):
         jobData=self.LoadRawData()
         sub='CisCurvatureDataBasedOnWaveFormat=';
         indices = []
         
         for line_num, line in enumerate(jobData):
             if len(line)>1:
                 if sub in line[0]:
                     indices.append(line_num) 
         cisFRONT=[]      
         try:
             cisFRONT = list(map(float, jobData[indices[0]][1:]))
         except:
             1
         cisBACK=[]
         if len(indices)>1:
            cisBACK = list(map(float, jobData[indices[1]][1:]))

         flag = True  # Set flag to True by default

         if len(cisFRONT) == len(cisBACK):  # Check if the lists have the same length
            for i in range(len(cisFRONT)):
                if cisFRONT[i] != cisBACK[i]:  # Check if the corresponding elements are different
                    flag = False  # Set flag to False if a difference is found
                    break
         else:
            flag = False
            
         if flag:
             cisBACK=[]
         
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
    
    def GetCIScurveNewVersion_secondTry(self):
        jobData=self.LoadRawData()
        sub='CISTilt=Value';
        
        indices = []
        for line_num, line in enumerate(jobData):
            try:
                if sub in line[0]:
                    indices.append(line_num) 
            except:
                continue
        cisFRONT=[] 
        cisBACK=[]   
        for s in jobData[indices[0]+1]:
            try:
              cisFRONT.append(float(s))  
            except:
                continue
                  
        if len(indices)>1:
            for s in jobData[indices[1]+1]:
                try:
                  cisBACK.append(float(s))  
                except:
                    continue
                             
        return cisBACK,cisFRONT; 

class RepareDistortions:
     def  __init__(self, WaveRawDataDic,WaveDataWithMaxFilterDic,ColorList): 
        self.WaveRawDataDic = WaveRawDataDic;
        self.WaveDataWithMaxFilterDic = WaveDataWithMaxFilterDic;
        self.ColorList= ColorList;
        
     def CalcWaveAfterFilterSubstraction(self): 
        col='Mean'
        WaveFilter_RawData={}
        
        for clr in self.ColorList:
            WaveFilter_RawData[clr]=list(self.WaveRawDataDic[clr][col]-self.WaveDataWithMaxFilterDic[clr][col])
        return   WaveFilter_RawData;
    
     def CalcCorrectionArrayOLD(self):
        
        WaveFilter_RawData=self.CalcWaveAfterFilterSubstraction();
        
        minDistpC=pd.DataFrame();
        CorrectionArr=[]
        for i in range(len(WaveFilter_RawData['Black'])):
            minDistpC=pd.DataFrame();
            for clrD in self.ColorList:
                difList={}
                for clr in ColorList:
                    if clr == clrD:
                        continue;
                    difList[(abs(WaveFilter_RawData[clrD][i]-WaveFilter_RawData[clr][i]))]=clr;
                tmpList=list(abs(np.array(list(difList.keys()))));
                
                minVal=min(tmpList);
                if  len(self.ColorList) > 4:   
                    
                    
                    tmpList.remove(min(tmpList))  
                    DistanceVal=math.sqrt(math.pow(minVal,2)+math.pow(min(tmpList),2));
                    minDistpC=pd.concat([minDistpC,pd.DataFrame([[DistanceVal,difList[minVal],difList[min(tmpList)]]])],axis=0).rename(index={0:clrD})
                else:

                    DistanceVal=math.sqrt(math.pow(minVal,2)+math.pow(min(tmpList),2));
                    minDistpC=pd.concat([minDistpC,pd.DataFrame([[DistanceVal,difList[minVal],difList[min(tmpList)]]])],axis=0).rename(index={0:clrD})
                
            clrName=pd.Series();    
            clrName=minDistpC[[0]].idxmin()
            ColssetCols=[]
            ColssetCols.append(WaveFilter_RawData[clrName[0]][i])
            for k in range(1,len(minDistpC.columns)):
                ColssetCols.append(WaveFilter_RawData[minDistpC[k][clrName[0]]][i])
                
            CorrectionArr.append(np.mean(ColssetCols))       
         
        return CorrectionArr
    
     def CalcCorrectionArray(self):
        WaveFilter_RawData=self.CalcWaveAfterFilterSubstraction();
        
        minDistpC=pd.DataFrame();
        CorrectionArr=[]
        for i in range(len(WaveFilter_RawData['Black'])):
            minDistpC=pd.DataFrame();
            ##Build dic of distance for each color
            count=0
            for clrD in self.ColorList:
                difList={}
                for clr in self.ColorList:
                    if clr == clrD:
                        continue;
                    difList[abs(WaveFilter_RawData[clrD][i]-WaveFilter_RawData[clr][i])]=clr;
                    count=count+1;
                tmpList=list((np.array(list(difList.keys()))));
                tmpList.sort()
                DistanceVal=0;
                listToAdd=[]
                # minVal=min(tmpList);
                if  len(self.ColorList) > 4:   
                    NieghborColors = NieghborColorsFor7colrs-1;
                else:
                    NieghborColors = 1;    
                for nbr in  range(NieghborColors):   
                    DistanceVal=DistanceVal+math.pow(tmpList[nbr],2);
                    listToAdd.append(difList[tmpList[nbr]])
                DistanceVal=math.sqrt(DistanceVal);
                listToAdd= [DistanceVal]+listToAdd
                minDistpC=pd.concat([minDistpC,pd.DataFrame([listToAdd])],axis=0).rename(index={0:clrD})
                # else:
        
            clrName=pd.Series();    
            clrName=minDistpC[[0]].idxmin()
            ColssetCols=[]
            ColssetCols.append(WaveFilter_RawData[clrName[0]][i])
            for k in range(1,len(minDistpC.columns)):
                ColssetCols.append(WaveFilter_RawData[minDistpC[k][clrName[0]]][i])
                
            CorrectionArr.append(np.mean(ColssetCols))
        
        return CorrectionArr
    
     def correctWaveRawData(self):
      
         CorrectionArr=self.CalcCorrectionArray();
         WaveRawDataDicAfterCorr={};
         WaveDataWithMaxFilterDicAfterCorr={};
         for clr in self.ColorList:
             WaveRawDataDicAfterCorr[clr]=self.WaveRawDataDic[clr]['Mean']-CorrectionArr;
             WaveDataWithMaxFilterDicAfterCorr[clr]=pd.Series(savgol_filter(WaveRawDataDicAfterCorr[clr], MaxWaveWindow, S_g_Degree))
             
         return  WaveRawDataDicAfterCorr,WaveDataWithMaxFilterDicAfterCorr,CorrectionArr 
     
 

            
def CalcMeanAndTilt(WaveRawDataDic,WaveDataWithMaxFilterDic,PHloc):
    PHoffSet={}
    PHtilt={}
    
    PHoffsetPerH={}
    PHtiltPerH={}
    
    for ColorForDisplay in ColorList: 
        try:
            y=WaveRawDataDic[ColorForDisplay]['Mean']-WaveDataWithMaxFilterDic[ColorForDisplay]['Mean'];
        except:
            y=WaveRawDataDic[ColorForDisplay]-WaveDataWithMaxFilterDic[ColorForDisplay];
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




        
class PlotGraphPlotly(CalcWaveFromRawData):
     def  __init__(self ,pthF,side,Panel,ColorList):
        super().__init__(pthF,side,Panel)
        self.ColorList = ColorList;
        
   


     def ShowWaveRawData_SubOffset_PerCycle(self,PlotTitle,offSetType,fileName,pnl):
         fig= go.Figure()
         for ColorForDisplay in self.ColorList:    
            db=self.ArrangeRawDataForAnalize(ColorForDisplay);
            
            if ColorForDisplay=='Yellow':
                ColorForDisplay='gold'; 
                
            col=list(db.columns)           
            rnge=range(len(col))
            
          
            for i in rnge:
                if offSetType == 'Left Side':
                    offSet=db[i+1][0];
                if offSetType == 'Right Side':
                    offSet=db[i+1][(len(db[i+1]))-1]  
                if offSetType == 'Middle':
                    offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
                if offSetType == 'Average All':
                    offSet=np.mean(db[i+1])
                if offSetType == 'Average Left Right':
                    offSet=np.mean([db[i+1][0],db[i+1][(len(db[i+1]))-1]])    
                fig.add_trace(
                go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
                            name='Cycle '+str(i+1)+' '+'Panel '+str(pnl)+' ' +ColorForDisplay))
       
         fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
          )

         fig.update_layout(title=self.side+' '+PlotTitle)
        
        
         plot(fig,filename=fileName+' '+str(pnl)+' '+self.side+".html") 
         
         return fig
     
     def CalcSTDbyOffset(self,offSet,db):
         
        col=list(db.columns)
        rnge=range(len(col))
        dbOffset=pd.DataFrame()
        STDOffset=[]
        for i in rnge:
 
            dbOffset=pd.concat([dbOffset,db[i+1]-offSet[i]],axis=1);
        for j,i in enumerate(dbOffset.index):
              if j == 0:
                  continue;
              STDOffset.append(np.std(dbOffset.loc[i,:]))
         
        return STDOffset;
   
     def ShowSTDforRawWaveWithOffset(self,PlotTitle,fileName,pnl):
         
        fig= go.Figure()
        for ColorForDisplay in self.ColorList:
            db=self.ArrangeRawDataForAnalize(ColorForDisplay);
        
            if ColorForDisplay=='Yellow':
                ColorForDisplay='gold';    
            
            offSet1=[]
            offSet2=[]
            offSet3=[]
            offSetAvgAll=[]
            offSetLRavrg=[]
            
            rnge=range(len(db.columns))
            for i in rnge:
                offSet1.append(db[i+1][0])
                offSet2.append(np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50]))
                offSet3.append(db[i+1][(len(db[i+1]))-1])
                
                offSetAvgAll.append(np.mean(db[i+1]))
                offSetLRavrg.append(np.mean([db[i+1][0],db[i+1][(len(db[i+1]))-1]]))
                
            middleSTD=self.CalcSTDbyOffset(offSet2,db)
            RightSTD=self.CalcSTDbyOffset(offSet3,db)
            LeftSTD=self.CalcSTDbyOffset(offSet1,db)
               
            AvgAllSTD=self.CalcSTDbyOffset(offSetAvgAll,db)
            LRavrgSTD=self.CalcSTDbyOffset(offSetLRavrg,db)

            
            fig.add_trace(
            go.Scatter(y=LeftSTD,line_color= ColorForDisplay,
                        name='Panel '+str(pnl)+' ' +ColorForDisplay+' LeftSide'))
            
            fig.data[len(fig.data)-1].visible = 'legendonly';
            
            fig.add_trace(
            go.Scatter(y=middleSTD,line_color= ColorForDisplay,
                        name='Panel '+str(pnl)+' ' +ColorForDisplay+' Middle'))
            fig.data[len(fig.data)-1].visible = 'legendonly';
            
            fig.add_trace(
            go.Scatter(y=RightSTD,line_color= ColorForDisplay,
                        name='Panel '+str(pnl)+' ' +ColorForDisplay+' RightSide'))
            fig.data[len(fig.data)-1].visible = 'legendonly';
            
            fig.add_trace(
            go.Scatter(y=AvgAllSTD,line_color= ColorForDisplay,
                        name='Panel '+str(pnl)+' ' +ColorForDisplay+' Average All'))
            
            fig.add_trace(
            go.Scatter(y=LRavrgSTD,line_color= ColorForDisplay,
                        name='Panel '+str(pnl)+' ' +ColorForDisplay+' Left Right Average'))
        
        
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )

        fig.update_layout(title=self.side+' '+PlotTitle)

        # plot(fig00)
        plot(fig,filename=fileName+' '+str(pnl)+' '+self.side+".html") 
      
        return fig

      
        
     def ShowWaveRawData_SubOffset_PerPanel(self,PlotTitle,offSetType,fileName,CcleNmber):
        
        fig = go.Figure()

        for Pnl in  range(1,12):  
            for ColorForDisplay in self.ColorList:
                db=CalcWaveFromRawData(self.pthF,self.side,Pnl).ArrangeRawDataForAnalize(ColorForDisplay);
                if ColorForDisplay=='Yellow':
                    ColorForDisplay='gold';
                

                if offSetType == 'Left Side':
                    offSet=db[CcleNmber][0];
                
                if offSetType == 'Right Side':
                    offSet=db[CcleNmber][(len(db[CcleNmber]))-1]  
                if offSetType == 'Middle':
                    offSet=np.min(db[CcleNmber][int(len(db[CcleNmber])/2)-50:int(len(db[CcleNmber])/2)+50])                
                
                if offSetType == 'Average All':
                    offSet=np.mean(db[CcleNmber])
                    
                if offSetType == 'Average Left Right':
                    offSet=np.mean([db[CcleNmber][0],db[CcleNmber][(len(db[CcleNmber]))-1]])  

                fig.add_trace(
                go.Scatter(y=list(db[CcleNmber]-offSet),line_color= ColorForDisplay,
                            name='Cycle '+str(CcleNmber)+' '+'Panel '+str(Pnl)+' ' +ColorForDisplay))
                
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        fig.update_layout(title=self.side+' '+PlotTitle)

        # plot(fig00)
        plot(fig,filename=fileName+' '+str(CcleNmber)+' '+self.side+".html")  
        
        return fig

     def PlotCIScurve(self,cisCurve,PlotTitle,fileName):      
        
     
        fig = go.Figure()
    
                
        fig.add_trace(
        go.Scatter(y=cisCurve,
                    name='CIS '+self.side+' curve'))
                
      
    
    
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        fig.update_layout(title=self.side+' '+PlotTitle)
        
       
        plot(fig,filename=fileName+"CIScurve"+self.side+".html") 
    
   
        if len(cisCurve)<1:
            print('************************************************************************************')
            print(fileName+' Has No CIS '+self.side+' curve information')
            print('************************************************************************************')
      
        return fig
    
     def PlotRegistrationBetweenWavePrints(self,DFdicPerClr,MainColor,rgistBtwPntStartCycle,rgistBtwPntEndCycle,fileName):
        
        fig = go.Figure()
            
        for clr in self.ColorList:
            if clr == MainColor:
                continue;
            
            for col in range(rgistBtwPntStartCycle,rgistBtwPntEndCycle+1):        
            # for col in DFdicPerClr[clr].columns:
                
                fig.add_trace(
                go.Scatter(y=DFdicPerClr[clr][col],line_color= clr,
                            name='Registration for cycle '+str(col)+' color '+clr))
        
        
            fig.update_layout(
                    hoverlabel=dict(
                        namelength=-1
                    )
                )
            fig.update_layout(title=self.side+' wave registration normalized to '+MainColor+' for Cycle Start ='+str(rgistBtwPntStartCycle)+' Cycle End='+str(rgistBtwPntEndCycle)+' ---> '+f)
            
       
            # plot(fig00)
        plot(fig,filename=fileName+' '+str(rgistBtwPntStartCycle)+'_'+str(rgistBtwPntEndCycle)+'_'+self.side+".html") 
        
        return fig

     def PlotWaveDataAfterApliedAVRGCorrection(self,WaveRawDataDic,WaveRawDataDicAfterCorr,CorrectionArr,PlotTitle,fileName):
        
            fig = make_subplots(specs=[[{"secondary_y": True}]])            
            for clr in self.ColorList:
                
                OffsetBefore= WaveRawDataDic[clr]['Mean'][0] 
        
            
                fig.add_trace(
                        go.Scatter(y=list(WaveRawDataDic[clr]['Mean']-OffsetBefore),line_color= clr,line=dict(dash='dot'),
                                    name='Wave Raw Data Before Corr '+ clr),secondary_y=False)
                fig.add_trace(
                        go.Scatter(y=list(WaveRawDataDicAfterCorr[clr]-OffsetBefore),line_color= clr,
                                    name='Wave Raw Data After Corr '+ clr),secondary_y=False)
            fig.add_trace(
                        go.Scatter(y=list(CorrectionArr),line=dict(color="#d8576b", width=3),
                        name='Average Correction'),secondary_y=True)
            fig.update_layout(title=self.side+' '+PlotTitle)
                    
          
                # plot(fig00)
            plot(fig,filename=fileName+' ' +self.side+".html")
            
            return fig
     def PlotDesidueBeforAfterAndAverageCorr(self,WaveRawDataDic,WaveRawDataDicAfterCorr,WaveDataWithMaxFilterDic,WaveDataWithMaxFilterDicAfterCorr,CorrectionArr,PlotTitle,fileName):
        
            fig = go.Figure()
            
            for clr in self.ColorList:
                
                ResidueBEFORE=(WaveRawDataDic[clr]['Mean']-WaveDataWithMaxFilterDic[clr]['Mean']);
                ResidueAFTER=(WaveRawDataDicAfterCorr[clr]-WaveDataWithMaxFilterDicAfterCorr[clr]);
                # fig.add_trace(
                #         go.Scatter(y=list(WaveRawDataDic[clr]['Mean']-WaveDataWithMaxFilterDic[clr]['Mean']),line_color= clr,line=dict(dash='dot'),
                #                     name='Wave Raw Data Before Corr '+ clr))
                fig.add_trace(
                        go.Scatter(y=ResidueBEFORE,line_color= clr,line=dict(dash='dot'),
                                    name='Wave Raw Data Before Corr '+ clr))
                fig.add_trace(
                        go.Scatter(y=ResidueAFTER,line_color= clr,
                                    name='Wave Raw Data After Corr '+ clr))
                
                fig.add_trace(
                        go.Scatter(y=WaveRawDataDic[clr]['Mean']-WaveRawDataDicAfterCorr[clr],line_color= clr,line=dict(dash='dash'),
                                    name='WaveRawData-WaveRawDataDicAfter '+ clr))
            fig.add_trace(
            go.Scatter(y=list(CorrectionArr),line=dict(color="#d8576b", width=3),
                        name='Average Correction'))
            fig.update_layout(title=self.side+' '+PlotTitle)
                    
          
                # plot(fig00)
            plot(fig,filename=fileName+' ' +self.side+".html")
            
            return fig   
        
     def PlotWaveDataResidue(self,WaveRawDataDic,WaveDataWithMaxFilterDic,PHloc,PHoffSet,PHtilt,PlotTitle,fileName):
         
        fig = make_subplots(specs=[[{"secondary_y": True}]])
 
        for clr in self.ColorList:     
            lineColor=clr;
          
            
            if lineColor=='Yellow':
                lineColor='gold';
            
            fig.add_trace(
            go.Scatter(y=WaveRawDataDic[clr],line_color= lineColor,
                        name='WaveData Raw '+str('Mean')+' color '+clr), secondary_y=False)
            
            fig.add_trace(
            go.Scatter(y=WaveDataWithMaxFilterDic[clr],line_color= lineColor,
                        name='WaveData with Filter color '+clr), secondary_y=False)
            
            fig.add_trace(
            go.Scatter(y=WaveRawDataDic[clr]-WaveDataWithMaxFilterDic[clr],line_color= lineColor,
                        name='Fiter - Raw color '+clr), secondary_y=True)
            
            # ymax=max(WaveRawDataDic[ColorList[0]]-WaveDataWithMaxFilterDic[self.ColorList[0]])
            ymax=20
            
            for i,PHlocMem in enumerate(PHloc):
                fig.add_trace(go.Scatter(x=[PHlocMem], y=[ymax],
                                        marker=dict(color="green", size=6),
                                        mode="markers",
                                        text='PH #'+str(i),
                                        # font_size=18,
                                        hoverinfo='text'),secondary_y=True)
                fig.data[len(fig.data)-1].showlegend = False
        
                fig.add_vline(x=PHlocMem, line_width=2, line_dash="dash", line_color="green")
            
            
            
            fig.add_trace(
            go.Scatter(y=PHoffSet[clr],line_color= lineColor,
                        name='Average(Fiter - Raw) color '+clr), secondary_y=True)
                
            
            
            fig.add_trace(
            go.Scatter(y=PHtilt[clr],line_color= lineColor,line=dict(dash='dot'),
                        name='Tilt(Fiter - Raw)  color '+clr), secondary_y=True)
            
            
            fig.update_layout(
                    hoverlabel=dict(
                        namelength=-1
                    )
                )
            fig.update_layout(title=self.side+' '+PlotTitle)
            
     
        plot(fig,filename=self.side+' '+fileName+".html") 
        
        return fig
    
    
     def PlotWaveDataSUBAveragePerPanelPerCycle_withMAX_MIN_diff(self,WaveRawDataDic,offSetType,PlotTitle,fileName):
        
        fig = go.Figure()
        max_vals=pd.DataFrame()
        min_vals=pd.DataFrame()
        max_valsPerPanel=pd.DataFrame()
        min_valsPerPanel=pd.DataFrame()
        for pnl in range(1,12):
            WaveRawDataDic=CalcWaveFromRawData(pthF,side,pnl).CreateDicOfWaveRawData();
        
            WaveRawDataDic_mean_offset=WaveRawDataDic;
            max_valsCLR=pd.DataFrame()
            min_valsCLR=pd.DataFrame()
            for clr in self.ColorList:     
                lineColor=clr;
              
                
                if lineColor=='Yellow':
                    lineColor='gold';
                for col in WaveRawDataDic[clr].columns:
                    if col == 'Mean':
                        WaveRawDataDic_mean_offset[clr].drop(col, axis=1, inplace=True)
        
                        break;
                    if col< StartCycle4Avr:
                        WaveRawDataDic_mean_offset[clr].drop(col, axis=1, inplace=True)
        
                        continue;
                    
                    if offSetType == 'Average All':
                        offset=np.mean(WaveRawDataDic[clr][col]-WaveRawDataDic[clr]['Mean']);
                        
                    if offSetType == 'Average Left Right':
                        WaveSUBmean=WaveRawDataDic[clr][col]-WaveRawDataDic[clr]['Mean'];
                        offset=np.mean([WaveSUBmean[0],WaveSUBmean[len(WaveSUBmean)-1]]);
        
                    WaveRawDataDic_mean_offset[clr][col]=WaveRawDataDic[clr][col]-WaveRawDataDic[clr]['Mean']-offset
        
                    fig.add_trace(
                    go.Scatter(y=WaveRawDataDic_mean_offset[clr][col],line_color= lineColor,
                                        name='WaveData Raw cycle '+str(col)+' - Mean'+' color '+clr+' Panel '+str(pnl)))    
                    
                    if not col == Cycle2Display:
                        fig.data[len(fig.data)-1].visible = 'legendonly';
                    if not pnl in Panel2Disply:
                        fig.data[len(fig.data)-1].visible = 'legendonly';
                 
                max_valsCLR=pd.concat([max_valsCLR,WaveRawDataDic_mean_offset[clr].max(axis=1)],axis=1).rename(columns={0: clr})
                min_valsCLR=pd.concat([min_valsCLR,WaveRawDataDic_mean_offset[clr].min(axis=1)],axis=1).rename(columns={0: clr})        
          
            max_valsPerPanel =pd.concat([max_valsPerPanel, max_valsCLR.max(axis=1)],axis=1).rename(columns={0: pnl}).dropna()
            min_valsPerPanel =pd.concat([min_valsPerPanel, min_valsCLR.min(axis=1)],axis=1).rename(columns={0: pnl}).dropna()
        
        max_vals=max_valsPerPanel.max(axis=1)
        min_vals=min_valsPerPanel.min(axis=1)
        
            
        fig.add_trace(
        go.Scatter(y=list( max_vals),line=dict(color=colorPNL[3], width=3),
                    name='Max  value')) 
        
        fig.add_trace(
        go.Scatter(y=list( min_vals),line=dict(color=colorPNL[3], width=3),
                    name='Min  value')) 
        
        # if not pnl in Panel2Disply:
        #     fig.data[len(fig.data)-1].visible = 'legendonly';
            
        fig.update_layout(
                hoverlabel=dict(
                    namelength=-1
                )
            )
        # fig.update_layout(title=PlotGraphPlotly(pthF,side,Panel,ColorList).side+' '+PlotTitle ,subtitle='Max-Min Mean ='+"{:.3f}".format(np.mean(max_vals-min_vals))+' Max-Min STD ='+"{:.3f}".format(np.std(max_vals-min_vals)))
            
           
        
        fig.update_layout(
            title=self.side+' '+PlotTitle+"<br><span style='font-size: 14px'><b>" +'Max-Min Mean ='+"{:.3f}".format(np.mean(max_vals-min_vals))+' Max-Min '+str(MaxMinPcnt)+'% ='+"{:.3f}".format(np.percentile(max_vals-min_vals, MaxMinPcnt))+"</b></span>",
            title_font=dict(size=16)
        )
        
        
        plot(fig,filename=self.side+' '+fileName+".html") 
       
        return fig    
    
     def PlotWaveDataSUBAveragePerPanelPerCycle(self,WaveRawDataDic,offSetType,PlotTitle,fileName):
         
        fig = go.Figure()
        
        for pnl in range(1,12):
            WaveRawDataDic=CalcWaveFromRawData(pthF,side,pnl).CreateDicOfWaveRawData();
    
     
            for clr in self.ColorList:     
                lineColor=clr;
              
                
                if lineColor=='Yellow':
                    lineColor='gold';
                for col in WaveRawDataDic[clr].columns:
                    if col == 'Mean':
                        break;
                    if col< StartCycle4Avr:
                        continue;
                    
                    if offSetType == 'Average All':
                        offset=np.mean(WaveRawDataDic[clr][col]-WaveRawDataDic[clr]['Mean']);
                    if offSetType == 'Average Left Right':
                        WaveSUBmean=WaveRawDataDic[clr][col]-WaveRawDataDic[clr]['Mean'];
                        offset=np.mean([WaveSUBmean[0],WaveSUBmean[len(WaveSUBmean)-1]]);

                    fig.add_trace(
                    go.Scatter(y=WaveRawDataDic[clr][col]-WaveRawDataDic[clr]['Mean']-offset,line_color= lineColor,
                                        name='WaveData Raw cycle '+str(col)+' - Mean'+' color '+clr+' Panel '+str(pnl)))    
                    
                    if not col == Cycle2Display:
                        fig.data[len(fig.data)-1].visible = 'legendonly';
                    if not pnl in Panel2Disply:
                        fig.data[len(fig.data)-1].visible = 'legendonly';

            
            
            
        fig.update_layout(
                hoverlabel=dict(
                    namelength=-1
                )
            )
        fig.update_layout(title=self.side+' '+PlotTitle)
            
     
        plot(fig,filename=self.side+' '+fileName+".html") 
        
        return fig
        
     def  PlotOffsetTabel(self,PHoffsetPerH,PlotTitle,fileName):
         
        PHname=[]
        header=[]
        ListofList=[]
 
        
        for i in range(24):
            PHname.append('PH NUMBER# '+str(i))
        
        for col in self.ColorList:
            header.append(col+' Offset')
            # header.append(col+' Tilt')
            new_list = [-number for number in PHoffsetPerH[col]]
            ListofList.append(new_list)
            # ListofList.append(PHtiltPerH[col])
        ####FRONT 
        figTable = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
                         cells=dict(values=[PHname]+ListofList,font=dict(color='black', size=15)))
                             ])
        
        figTable.update_layout(title=self.side+' '+PlotTitle)
        
        plot(figTable,filename=self.side+' '+fileName+".html") 
        
        return figTable
    
     def PlotTiltTable(self,PHtiltPerH,ColorLevelsTilt,DivideByNumTilt,PlotTitle,fileName):
        
        PHname=[]
        for i in range(24):
            PHname.append('PH NUMBER# '+str(i)) 
        headerTilt=[]
        ListofListTilt=[]
  
        for col in self.ColorList:
            headerTilt.append(col+' Tilt')
            # header.append(col+' Tilt')
            ListofListTilt.append(PHtiltPerH[col])
        
        backGroundCLR='rgb(200, 200, 200)'
        colors = n_colors(backGroundCLR, 'rgb(200, 0, 0)', ColorLevelsTilt, colortype='rgb')
        fillcolorList=[]
        formatList=[]
        formatList.append("")
        for i in range(len(ListofListTilt)):
            fillcolorList.append(np.array(colors)[(abs(np.asarray(ListofListTilt[i]))/DivideByNumTilt).astype(int)])
            formatList.append("0.2f")
            
        
        ####FRONT Tilt
        figTableTilt = go.Figure(data=[go.Table(header=dict(values=['PH#']+headerTilt),
                         cells=dict(values=[PHname]+ListofListTilt,fill_color=[backGroundCLR]+fillcolorList,font=dict(color='black', size=15),format=formatList))
                             ])
        
        figTableTilt.update_layout(title=self.side+' '+PlotTitle)     
        plot(figTableTilt,filename=self.side+' '+fileName+".html")  
        
        return figTableTilt
    
    
     def PlotFRONT_BACKDeltaTable(self,PHoffsetPerHFRONT,PHoffsetPerHBACK,DivideByNum,ColorLevels,PlotTitle,fileName):
         
        PHname=[]
        for i in range(24):
            PHname.append('PH NUMBER# '+str(i)) 
        ListofListDelta=[]    
        header=[]
        fillcolorList=[]  
        backGroundCLR='rgb(200, 200, 200)'
        colors = n_colors(backGroundCLR, 'rgb(200, 0, 0)', ColorLevels, colortype='rgb')
    
        for col in ColorList:
            header.append(col+'Delta(Front-Back) Offset')
        for col in ColorList:
            ListofListDelta.append(list(np.asarray(PHoffsetPerHFRONT[col])-np.asarray(PHoffsetPerHBACK[col])))
        formatList=[]
        formatList.append("")    
        for i in range(len(ListofListDelta)):
            # x2 = 30 * np.ones(len(ListofListDelta[i]))
            fillcolorList.append(np.array(colors)[(abs(np.asarray(ListofListDelta[i]))/DivideByNum).astype(int)])
            formatList.append("0.2f")

        
            
        figTableDelta = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
                     cells=dict(values=[PHname]+ListofListDelta,fill_color=[backGroundCLR]+fillcolorList,font=dict(color='black', size=15),format=formatList))
                         ])
        figTableDelta.update_layout(title=PlotTitle)  
        
        plot(figTableDelta,filename=fileName+"_.html")   
        
        return figTableDelta;
    
    
     def  PlotFRONT_BACKAverageTable(self,PHoffsetPerHFRONT,PHoffsetPerHBACK,PlotTitle,fileName):
         
        PHname=[]
        for i in range(24):
            PHname.append('PH NUMBER# '+str(i)) 
        ListofListAverage=[]    
        header=[]
        fillcolorList=[]  
        
    
        for col in ColorList:
            header.append(col+'Average(Front&Back) Offset')
        for col in ColorList:
            ListofListAverage.append(list(-(np.asarray(PHoffsetPerHFRONT[col])+np.asarray(PHoffsetPerHBACK[col]))/2))
            
      
        
        
            
        figTableAverage = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
                     cells=dict(values=[PHname]+ListofListAverage,font=dict(color='black', size=15)))
                         ])
        figTableAverage.update_layout(title=PlotTitle)  
        
        plot(figTableAverage,filename=fileName+"_.html")   
        
        return figTableAverage;
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
# pthF = filedialog.askdirectory()


pthF = filedialog.askopenfilename()

f=pthF.split('/')[len(pthF.split('/'))-1]

DirectorypathF=pthF.replace(f,"")[:-1]



# f=pthF.split('/')[len(pthF.split('/'))-1]
# DirectorypathF=pthF.replace(f,'');
os.chdir(DirectorypathF)

side='Front';

ColorList= CalcWaveFromRawData(pthF,side,Panel).getColors();

LocatorIndex= CalcWaveFromRawData(pthF,side,Panel).GetLocatorIndex(ColorForDisplay);
# FlatList= CalcWaveFromRawData(pthF,side,Panel,ColorForDisplay).getNumberOfFlats();
if CIScurve:
    cisBACKold,cisFRONTold=CIScurveFromRawData(pthF).GetCIScurveOldVersion()
    
    
    if len(cisFRONTold) == 0:
        cisBACKold,cisFRONTold=CIScurveFromRawData(pthF).GetCIScurveOldVersion_SecondTry()

    cisBACKnew,cisFRONTnew=CIScurveFromRawData(pthF).GetCIScurveNewVersion()

    if len(cisFRONTnew) == 0:
        cisBACKnew,cisFRONTnew=CIScurveFromRawData(pthF).GetCIScurveNewVersion_secondTry()
       
        
    
if registrationBetweenWavePrints:
    DFdicPerClrFRONT =  CalcRegistrationFromWaveData(pthF,side,Panel,ColorList,MainColor,StartCycle).DeltaForCycleAndColor() 
    try:
        DFdicPerClrBACK =  CalcRegistrationFromWaveData(pthF,'Back',Panel,ColorList,MainColor,StartCycle).DeltaForCycleAndColor() 
    except:
        1


WaveRawDataDicFRONT=CalcWaveFromRawData(pthF,side,Panel).CreateDicOfWaveRawData();
# WaveDataWithMaxFilterDicFRONT=CalcWaveFromRawData(pthF,side,Panel).FilterWaveDataDic()
WaveDataWithMaxFilterDicFRONT=CalcWaveFromRawData(pthF,side,Panel).FilterWaveDataDicTEST(WaveRawDataDicFRONT)

PHlocFRONT= CalcWaveFromRawData(pthF,side,Panel).CalcPHlocation(ColorForDisplay)
try:
    WaveRawDataDicBACK=CalcWaveFromRawData(pthF,'Back',Panel).CreateDicOfWaveRawData();
    WaveDataWithMaxFilterDicBACK=CalcWaveFromRawData(pthF,'Back',Panel).FilterWaveDataDic()
    PHlocBACK= CalcWaveFromRawData(pthF,'Back',Panel).CalcPHlocation(ColorForDisplay)

except:
    1



################ Calc offset and tilt

PHoffSetFRONT,PHtiltFRONT,PHoffsetPerHFRONT,PHtiltPerHFRONT=CalcMeanAndTilt(WaveRawDataDicFRONT,WaveDataWithMaxFilterDicFRONT,PHlocFRONT)

try:
   PHoffSetBACK,PHtiltBACK,PHoffsetPerHBACK,PHtiltPerHBACK=CalcMeanAndTilt(WaveRawDataDicBACK,WaveDataWithMaxFilterDicBACK,PHlocBACK)
except:
    1
############

#################### Calc curev, filetr, offset, tilt after correction

WaveRawDataDicAfterCorrFRONT,WaveDataWithMaxFilterDicAfterCorrFRONT,CorrectionArrFRONT=RepareDistortions(WaveRawDataDicFRONT,WaveDataWithMaxFilterDicFRONT,ColorList).correctWaveRawData();
try:
    WaveRawDataDicAfterCorrBACK,WaveDataWithMaxFilterDicAfterCorrBACK,CorrectionArrBACK=RepareDistortions(WaveRawDataDicBACK,WaveDataWithMaxFilterDicBACK,ColorList).correctWaveRawData();
except:
    1

PHoffSetFRONTAfterCorr,PHtiltFRONTAfterCorr,PHoffsetPerHFRONTAfterCorr,PHtiltPerHFRONTAfterCorr=CalcMeanAndTilt(WaveRawDataDicAfterCorrFRONT,WaveDataWithMaxFilterDicAfterCorrFRONT,PHlocFRONT)

try:
   PHoffSetBACKAfterCorr,PHtiltBACKAfterCorr,PHoffsetPerHBACKAfterCorr,PHtiltPerHBACKAfterCorr=CalcMeanAndTilt(WaveRawDataDicAfterCorrBACK,WaveDataWithMaxFilterDicAfterCorrBACK,PHlocBACK)
except:
    1















#######################################################################################
#######################################################################################
#############################PLOT############################################
#############################Wave RawData Per Cycle###################################
##### Front
if plotPerCycle:
    offSetType='Average All'
    PlotTitle='- Left Side Offset WAVE RAW DATA --->'+f +' offSetType='+offSetType; # Can modify Plot title
    fileName=f+" Left Side WaveResult_RawDataPerCycle Panel Number "; # Can modify File nmae
    side='Front'
    # db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    figLeftsideFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerCycle(PlotTitle,offSetType,fileName,Panel)
    
    
    offSetType='Average Left Right' 
    PlotTitle='- Right Side Offset WAVE RAW DATA --->'+f+' offSetType='+offSetType;# Can modify Plot title
    fileName=f+" Right Side WaveResult_RawDataPerCycle Panel Number ";# Can modify File nmae
    side='Front'
    db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    figRightsideFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerCycle(PlotTitle,offSetType,fileName,Panel)
    
   
    
    PlotTitle='- <b>STD </b> Side Offset WAVE RAW DATA --->'+f# Can modify Plot title
    fileName=f+" STD SideOffset_ WaveResult_RawDataPerColor Panel Number "# Can modify File nmae
    side='Front'
    figSTDFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowSTDforRawWaveWithOffset(PlotTitle,fileName,Panel)
    
    #### Back
    try:
        offSetType='Average All'
        PlotTitle='- Left Side Offset WAVE RAW DATA --->'+f+' offSetType='+offSetType;# Can modify Plot title
        fileName=f+" Left Side WaveResult_RawDataPerCycle Panel Number ";# Can modify File nmae
        side='Back'
        # db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        figLeftsideBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerCycle(PlotTitle,offSetType,fileName,Panel)
        
        
        offSetType='Average Left Right' 
        PlotTitle='- Right Side Offset WAVE RAW DATA --->'+f+' offSetType='+offSetType;# Can modify Plot title
        fileName=f+" Right Side WaveResult_RawDataPerCycle Panel Number ";# Can modify File nmae
        side='Back'
        # db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        figRightsideBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerCycle(PlotTitle,offSetType,fileName,Panel)
        
        # offSetType='Middle'
        # PlotTitle='- Middle Offset WAVE RAW DATA --->'+f;# Can modify Plot title
        # fileName=f+" Middle WaveResult_RawDataPerCycle Panel Number ";# Can modify File nmae
        # side='Back'
        # # db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        # figMiddlesideBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerCycle(PlotTitle,offSetType,fileName,Panel)
        
        PlotTitle='- <b>STD </b> Side Offset WAVE RAW DATA --->'+f# Can modify Plot title
        fileName=f+" STD SideOffset_ WaveResult_RawDataPerColor Panel Number "# Can modify File nmae
        side='Back'
        figSTDBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowSTDforRawWaveWithOffset(PlotTitle,fileName,Panel)
    except:
        1

########################################################################################
#############################Wave RawData Per Panel###################################
##### Front
if plotPerPanel:
    offSetType='Average All'
    PlotTitle='- offSetType='+offSetType+' WAVE RAW DATA (For one Cycle)--->'+f;
    fileName=f+'- offSetType='+offSetType+" WaveResult_RawDataPerPanel ";
    side='Front'
    # db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    figLeftsideFRONTperPanel=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
    
    
    
    offSetType='Average Left Right' 
    PlotTitle='- offSetType='+offSetType+' WAVE RAW DATA (For one Cycle)--->'+f;
    fileName=f+'- offSetType='+offSetType+" WaveResult_RawDataPerPanel ";
    side='Front'
    # db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    figRightsideFRONTperPanel=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
    
    # offSetType='Middle'
    # PlotTitle='- Middle Offset WAVE RAW DATA (For one Cycle)--->'+f;
    # fileName=f+" Right Side WaveResult_RawDataPerPanel ";
    # side='Front'
    # # db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    # figMiddlesideFRONTperPanel=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
    
    
    #### Back
    try:
        offSetType='Average All'
        PlotTitle='- offSetType='+offSetType+' WAVE RAW DATA (For one Cycle)--->'+f;
        fileName=f+'- offSetType='+offSetType+" WaveResult_RawDataPerPanel ";
        side='Back'
        # db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        figLeftsideBACKperPanel=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
        
        
        
        offSetType='Average Left Right' 
        PlotTitle='- offSetType='+offSetType+' WAVE RAW DATA (For one Cycle)--->'+f;
        fileName=f+'- offSetType='+offSetType+" WaveResult_RawDataPerPanel ";
        side='Back'
        # db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        figRightsideBACKperPanel=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
        
        # offSetType='Middle'
        # PlotTitle='- Middle Offset WAVE RAW DATA (For one Cycle)--->'+f;
        # fileName=f+" Right Side WaveResult_RawDataPerPanel ";
        # side='Back'
        # # db=CalcWaveFromRawData(pthF,side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        # figMiddlesideBACKperPanel=PlotGraphPlotly(pthF,side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
    
    except:
        1

########################################################################################
#############################Plot Wave Data SUB Average Per Panel Per Cycle###################################
##### Front
if WaveDataSUBAverage_PerPanel_PerCycle:
    offSetType='Average All' ;
    PlotTitle='- Wave Behavior- Avi  Method --->'+f+' offSetType='+offSetType;
    fileName=f+" Wave Behavior- Avi  Method Offsettype_"+offSetType;
    side='Front'
    figWaveDataSubAveragePerPanet=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotWaveDataSUBAveragePerPanelPerCycle_withMAX_MIN_diff(WaveRawDataDicFRONT,offSetType,PlotTitle,fileName);

    offSetType='Average Left Right' ;
    PlotTitle='- Wave Behavior- Avi  Method --->'+f+' offSetType='+offSetType;
    fileName=f+" Wave Behavior- Avi  Method Offsettype_"+offSetType;
    side='Front'
    figWaveDataSubAveragePerPanet=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotWaveDataSUBAveragePerPanelPerCycle_withMAX_MIN_diff(WaveRawDataDicFRONT,offSetType,PlotTitle,fileName);
        
    #####Back
    try:
        offSetType='Average All' #
        PlotTitle='- Wave Behavior- Avi  Method --->'+f+' offSetType='+offSetType;
        fileName=f+" Wave Behavior- Avi  Method Offsettype_"+offSetType;
        
        side='Back'
        figWaveDataSubAveragePerPanet=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotWaveDataSUBAveragePerPanelPerCycle_withMAX_MIN_diff(WaveRawDataDicBACK,offSetType,PlotTitle,fileName);
        
        offSetType='Average Left Right' #    
        PlotTitle='- Wave Behavior- Avi  Method --->'+f+' offSetType='+offSetType;
        fileName=f+" Wave Behavior- Avi  Method Offsettype_"+offSetType;
        side='Back'
        figWaveDataSubAveragePerPanet=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotWaveDataSUBAveragePerPanelPerCycle_withMAX_MIN_diff(WaveRawDataDicBACK,offSetType,PlotTitle,fileName);

    except:
        1

##################################################################################
################################CIS Curve 
######FRONT
if CIScurve:

    try:
        PlotTitle='FRONT CIS curve (old version)--->'+f;
        fileName=f+' old ';
        side='Front';
        cisCurve=cisFRONTold
        figCISFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotCIScurve(cisCurve,PlotTitle,fileName);
    except:
        1
        
    #######BACK
    try:
        PlotTitle='BACK CIS curve (old version)--->'+f;
        fileName=f+' old ';
        side='Back';
        cisCurve=cisBACKold
        figCISBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotCIScurve(cisCurve,PlotTitle,fileName); 
    except:
        1

    try:
        PlotTitle='FRONT CIS curve (new version)--->'+f;
        fileName=f+' new ';
        side='Front';
        cisCurve=cisFRONTnew
        figCISFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotCIScurve(cisCurve,PlotTitle,fileName);
    except:
        1
        
    #######BACK
    try:
        PlotTitle='BACK CIS curve (new version)--->'+f;
        fileName=f+' new ';
        side='Back';
        cisCurve=cisBACKnew
        figCISBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotCIScurve(cisCurve,PlotTitle,fileName); 
    except:
        1

##################################################################################
################################Registration Between Wave Prints
######FRONT  & BACK
if registrationBetweenWavePrints: ##Yuval method
    try:
        fileName=f
        side='Front'
        figRegistrationBetweenWavePrintsFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotRegistrationBetweenWavePrints(DFdicPerClrFRONT,MainColor,rgistBtwPntStartCycle,rgistBtwPntEndCycle,fileName)
    
        side='Back'
        figRegistrationBetweenWavePrintsBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotRegistrationBetweenWavePrints(DFdicPerClrBACK,MainColor,rgistBtwPntStartCycle,rgistBtwPntEndCycle,fileName)
    except:
        1
##################################################################################
################################ Residue (Wave- S.Go filter) Before and  After Aplied AVRG Correction
######FRONT 
if BeforAndAfterCorr:
    PlotTitle=' Residue (Wave- S.Go filter) before and after correction ---> '+f;
    fileName=f+'Residue _Wave sub S.Go filter_ before and after correction_';
    side='Front';
    figCorrAvrBeforeAndAfterFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotDesidueBeforAfterAndAverageCorr(WaveRawDataDicFRONT,WaveRawDataDicAfterCorrFRONT,WaveDataWithMaxFilterDicFRONT,WaveDataWithMaxFilterDicAfterCorrFRONT,CorrectionArrFRONT,PlotTitle,fileName)
    
    ######BACK
    try:
        
        PlotTitle=' Residue (Wave- S.Go filter) before and after correction ---> '+f;
        fileName=f+'Residue _Wave sub S.Go filter_ before and after correction_';
        side='Back';
        figCorrAvrBeforeAndAfterBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotDesidueBeforAfterAndAverageCorr(WaveRawDataDicBACK,WaveRawDataDicAfterCorrBACK,WaveDataWithMaxFilterDicBACK,WaveDataWithMaxFilterDicAfterCorrBACK,CorrectionArrBACK,PlotTitle,fileName)
    except:
        1

##################################################################################
################################Wave Data Before and  After Aplied AVRG Correction
######FRONT 
if BeforAndAfterCorr:
    PlotTitle=' wave raw data before and after correction ---> '+f;
    fileName=f+'wave raw data before and after correction_';
    side='Front';
    figClrBeforeAndAfterFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotWaveDataAfterApliedAVRGCorrection(WaveRawDataDicFRONT,WaveRawDataDicAfterCorrFRONT,CorrectionArrFRONT,PlotTitle,fileName)
    
    ######BACK
    try:
        
        PlotTitle=' wave raw data before and after correction ---> '+f;
        fileName=f+'wave raw data before and after correction_';
        side='Back';
        figClrBeforeAndAfterBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotWaveDataAfterApliedAVRGCorrection(WaveRawDataDicBACK,WaveRawDataDicAfterCorrBACK,CorrectionArrBACK,PlotTitle,fileName)
    except:
        1

#################################################################################
##################################Plot DX Wave DataResidue After correction
########FRONT
if WaveFilterResidue_dxPlot:
    PlotTitle=' After Correction Wave Data S.Golay = '+ str(MaxWaveWindow)+'---> '+f
    fileName=f+'  After Correction Wave Data S.Golay _'+ str(MaxWaveWindow)
    side='Front';
    figWaveResidueAfterCorrFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotWaveDataResidue(WaveRawDataDicAfterCorrFRONT,WaveDataWithMaxFilterDicAfterCorrFRONT,PHlocFRONT,PHoffSetFRONTAfterCorr,PHtiltFRONTAfterCorr,PlotTitle,fileName)
    ########BACK
    try:
        PlotTitle=' After Correction Wave Data S.Golay = '+ str(MaxWaveWindow)+'---> '+f
        fileName=f+'  After Correction Wave Data S.Golay _'+ str(MaxWaveWindow)
        side='Back';
        figWaveResidueAfterCorrBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotWaveDataResidue(WaveRawDataDicAfterCorrBACK,WaveDataWithMaxFilterDicAfterCorrBACK,PHlocBACK,PHoffSetBACKAfterCorr,PHtiltBACKAfterCorr,PlotTitle,fileName)
    except:
         1     


#################################################################################
####################### Table: Offset Table -  After correction
#############Front
if PlotTables:
    PlotTitle=' offset (Correction-For simplex) table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
    fileName=f+" Offset Table"
    side='Front';
    TableOffsetAfterCorrFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotOffsetTabel(PHoffsetPerHFRONTAfterCorr,PlotTitle,fileName)
    
    
    #####Back
    
    try:
        PlotTitle=' offset (Correction-For simplex) table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
        fileName=f+" Offset Table"
        side='Back';
        TableOffsetAfterCorrBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotOffsetTabel(PHoffsetPerHBACKAfterCorrAfterCorr,PlotTitle,fileName)
    except:
     1     
    
    #################################################################################
    ####################### Table: Tilt Table -  After correction
    #############Front
    
    PlotTitle=' Tilt table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
    fileName=f+" Tilt Table"
    side='Front';
    TableTiltAfterCorrFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotTiltTable(PHtiltPerHFRONTAfterCorr,ColorLevelsTilt,DivideByNumTilt,PlotTitle,fileName)
    
    #####Back
    try:
        PlotTitle=' Tilt table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
        fileName=f+" Tilt Table"
        side='Back';
        TableTiltAfterCorrBACK=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotTiltTable(PHtiltPerHBACKAfterCorr,ColorLevelsTilt,DivideByNumTilt,PlotTitle,fileName)
    except:
        1
    
    #################################################################################
    ####################### Table: FRONT -BACK Delta - After correction
    
    try:
        PlotTitle='Delta offset table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
        fileName=f+" Delta Offset Table"
        side='Front';
        TableFRONT_BACK_AverageAfterCorrFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotFRONT_BACKDeltaTable(PHoffsetPerHFRONTAfterCorr,PHoffsetPerHBACKAfterCorr,DivideByNum,ColorLevels,PlotTitle,fileName);
    except:
        1
        
    #################################################################################
    ####################### Table: FRONT -BACK Average -  After correction
    try:
        PlotTitle='Correction table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
        fileName=f+" FRONT -BACK Average Table"
        side='Front';
        TableFRONT_BACK_AverageAfterCorrFRONT=PlotGraphPlotly(pthF,side,Panel,ColorList).PlotFRONT_BACKAverageTable(PHoffsetPerHFRONTAfterCorr,PHoffsetPerHBACKAfterCorr,PlotTitle,fileName);
    except:
        1





#########################################################################################

# jobData=CIScurveFromRawData(pthF).LoadRawData()
# sub='CisCurvatureDataBasedOnWaveFormat=';
# indices = []

# for line_num, line in enumerate(jobData):
#     if len(line)>1:
#         if sub in line[0]:
#             indices.append(line_num) 
   
# cisFRONT = list(map(float, jobData[indices[0]][1:]))
# if len(indices)>1:
#     cisBACK = list(map(float, jobData[indices[1]][1:]))
