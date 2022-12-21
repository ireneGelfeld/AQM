# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:59:12 2022

@author: Ireneg
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

#####################################Params #############################################################
#########################################################################################################
global StartCycle,StartCycle4Avr,PHpoitToIgnor,MaxWaveWindow,DistanceBtWPointMM,Panel,Cycle2Display



## for plot per panel and plot per cycle and WaveData SUB Average_PerPanel_PerCycle
plotPerPanel=1;# On/OFF plot
plotPerCycle=1;## On/OFF plot
WaveDataSUBAverage_PerPanel_PerCycle=1 # On/OFF plot
CycleNumber =3 # cycle view in => plot Per Panel
StartCycle4Avr = 2; # Start averaging for all plots defult = 2
Panel = 6;          #view panel for plot Per cycle
ColorForDisplay = 'Cyan' # Not in use
Cycle2Display = 2 # defult visible cycle in plot WaveDataSUBAverage_PerPanel_PerCycle

## for plot CIScurve
CIScurve=1;# On/OFF plot

## for plot registration estimation in Wave Prints (yuval)
registrationBetweenWavePrints=1; # On/OFF plot
StartCycle=3
rgistBtwPntStartCycle=StartCycle # (it is not a parameter)
rgistBtwPntEndCycle=StartCycle+1 # for long print can change to larger number
MainColor = "Black" #Referance Color

##  Wave plot ( before and after correction)
BeforAndAfterCorr=1# On/OFF plot


## DX plot - delta between wave and starvitzky filer (residue) 
WaveFilterResidue_dxPlot=1 # On/OFF plot
PHpoitToIgnor=2; # Ponits of Print head to ignar (16 point in total) in each side
MaxWaveWindow=51;# S.gol filter window
DistanceBtWPointMM=2.734


###for Tables
PlotTables=1 # On/OFF table
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
        
        # DataSec=RawData[RawData['Overall Status']=='Success'].reset_index(drop=True);
        
        DataSec=RawData;


        DataSecPrintDirc=DataSec[DataSec['Direction Type ']=='Print Direction']
        
        DataSecPrintDircPanel=DataSecPrintDirc[DataSecPrintDirc['Panel Id']==self.Panel]
        
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
    
     def CalcCorrectionArray(self):
        
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
    
     def correctWaveRawData(self):
      
         CorrectionArr=self.CalcCorrectionArray();
         WaveRawDataDicAfterCorr={};
         WaveDataWithMaxFilterDicAfterCorr={};
         for clr in self.ColorList:
             WaveRawDataDicAfterCorr[clr]=self.WaveRawDataDic[clr]['Mean']-CorrectionArr;
             WaveDataWithMaxFilterDicAfterCorr[clr]=pd.Series(savgol_filter(WaveRawDataDicAfterCorr[clr], MaxWaveWindow, 1))
             
         return  WaveRawDataDicAfterCorr,WaveDataWithMaxFilterDicAfterCorr 
     
 

            
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

     def PlotWaveDataAfterApliedAVRGCorrection(self,WaveRawDataDic,WaveRawDataDicAfterCorr,PlotTitle,fileName):
        
            fig = go.Figure()
            
            for clr in self.ColorList:
                
                OffsetBefore= WaveRawDataDic[clr]['Mean'][0] 
        
            
                fig.add_trace(
                        go.Scatter(y=list(WaveRawDataDic[clr]['Mean']-OffsetBefore),line_color= clr,line=dict(dash='dot'),
                                    name='Wave Raw Data Before Corr '+ clr))
                fig.add_trace(
                        go.Scatter(y=list(WaveRawDataDicAfterCorr[clr]-OffsetBefore),line_color= clr,
                                    name='Wave Raw Data After Corr '+ clr))
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
            go.Scatter(y=WaveRawDataDic[clr]['Mean'],line_color= lineColor,
                        name='WaveData Raw '+str('Mean')+' color '+clr), secondary_y=False)
            
            fig.add_trace(
            go.Scatter(y=WaveDataWithMaxFilterDic[clr]['Mean'],line_color= lineColor,
                        name='WaveData with Filter color '+clr), secondary_y=False)
            
            fig.add_trace(
            go.Scatter(y=WaveRawDataDic[clr]['Mean']-WaveDataWithMaxFilterDic[clr]['Mean'],line_color= lineColor,
                        name='Fiter - Raw color '+clr), secondary_y=True)
            
            ymax=max(WaveRawDataDic[ColorList[0]]['Mean']-WaveDataWithMaxFilterDic[self.ColorList[0]]['Mean'])
            
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
    
    
     def PlotWaveDataSUBAveragePerPanelPerCycle(self,WaveRawDataDic,offSetType,PlotTitle,fileName):
         
        fig = go.Figure()
        
        for pnl in range(1,12):
            WaveRawDataDic=CalcWaveFromRawData(pthF+'/',side,pnl).CreateDicOfWaveRawData();
    
     
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
            
        for i in range(len(ListofListDelta)):
            # x2 = 30 * np.ones(len(ListofListDelta[i]))
            fillcolorList.append(np.array(colors)[(abs(np.asarray(ListofListDelta[i]))/DivideByNum).astype(int)])
        
        
            
        figTableDelta = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
                     cells=dict(values=[PHname]+ListofListDelta,fill_color=[backGroundCLR]+fillcolorList,font=dict(color='black', size=15)))
                         ])
        figTableDelta.update_layout(title=self.side+' '+PlotTitle)  
        
        plot(figTableDelta,filename=self.side+' '+fileName+".html")   
        
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
        figTableAverage.update_layout(title=self.side+' '+PlotTitle)  
        
        plot(figTableAverage,filename=self.side+' '+fileName+".html")   
        
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
pthF = filedialog.askdirectory()


f=pthF.split('/')[len(pthF.split('/'))-1]
DirectorypathF=pthF.replace(f,'');
os.chdir(DirectorypathF)

side='Front';

ColorList= CalcWaveFromRawData(pthF+'/',side,Panel).getColors();

LocatorIndex= CalcWaveFromRawData(pthF+'/',side,Panel).GetLocatorIndex(ColorForDisplay);
# FlatList= CalcWaveFromRawData(pthF+'/',side,Panel,ColorForDisplay).getNumberOfFlats();
if CIScurve:
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

#################
 
# fig = go.Figure()

# for pnl in range(1,12):
#     WaveRawDataDic=CalcWaveFromRawData(pthF+'/',side,pnl).CreateDicOfWaveRawData();

 
#     for clr in PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ColorList:     
#         lineColor=clr;
      
        
#         if lineColor=='Yellow':
#             lineColor='gold';
#         for col in WaveRawDataDic[clr].columns:
#             if col == 'Mean':
#                 break;
#             fig.add_trace(
#             go.Scatter(y=WaveRawDataDic[clr][col]-WaveRawDataDic[clr]['Mean'],line_color= clr,
#                                 name='WaveData Raw cycle '+str(col)+' - Mean'+' color '+clr+' Panel '+str(pnl)))    

    
    
    
# fig.update_layout(
#         hoverlabel=dict(
#             namelength=-1
#         )
#     )
# fig.update_layout(title=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).side+' '+PlotTitle)
    
 
# plot(fig,filename=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).side+' '+fileName+".html") 


################ Calc offset and tilt

PHoffSetFRONT,PHtiltFRONT,PHoffsetPerHFRONT,PHtiltPerHFRONT=CalcMeanAndTilt(WaveRawDataDicFRONT,WaveDataWithMaxFilterDicFRONT,PHlocFRONT)

try:
   PHoffSetBACK,PHtiltBACK,PHoffsetPerHBACK,PHtiltPerHBACK=CalcMeanAndTilt(WaveRawDataDicBACK,WaveDataWithMaxFilterDicBACK,PHlocBACK)
except:
    1
############

#################### Calc curev, filetr, offset, tilt after correction

WaveRawDataDicAfterCorrFRONT,WaveDataWithMaxFilterDicAfterCorrFRONT=RepareDistortions(WaveRawDataDicFRONT,WaveDataWithMaxFilterDicFRONT,ColorList).correctWaveRawData();
try:
    WaveRawDataDicAfterCorrBACK,WaveDataWithMaxFilterDicAfterCorrBACK=RepareDistortions(WaveRawDataDicBACK,WaveDataWithMaxFilterDicBACK,ColorList).correctWaveRawData();
except:
    1

PHoffSetFRONTAfterCorr,PHtiltFRONTAfterCorr,PHoffsetPerHFRONTAfterCorr,PHtiltPerHFRONTAfterCorr=CalcMeanAndTilt(WaveRawDataDicAfterCorrFRONT,WaveDataWithMaxFilterDicAfterCorrFRONT,PHlocFRONT)

try:
   PHoffSetBACKAfterCorr,PHtiltBACKAfterCorr,PHoffsetPerHBACKAfterCorrAfterCorr,PHtiltPerHBACKAfterCorr=CalcMeanAndTilt(WaveRawDataDicAfterCorrBACK,WaveDataWithMaxFilterDicAfterCorrBACK,PHlocBACK)
except:
    1



############################Calc Average delta of cycle per panel


#######################################################################################
#######################################################################################
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
    # db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    figLeftsideFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerCycle(PlotTitle,offSetType,fileName,Panel)
    
    
    offSetType='Average Left Right' 
    PlotTitle='- Right Side Offset WAVE RAW DATA --->'+f+' offSetType='+offSetType;# Can modify Plot title
    fileName=f+" Right Side WaveResult_RawDataPerCycle Panel Number ";# Can modify File nmae
    side='Front'
    db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    figRightsideFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerCycle(PlotTitle,offSetType,fileName,Panel)
    
   
    
    PlotTitle='- <b>STD </b> Side Offset WAVE RAW DATA --->'+f# Can modify Plot title
    fileName=f+" STD SideOffset_ WaveResult_RawDataPerColor Panel Number "# Can modify File nmae
    side='Front'
    figSTDFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowSTDforRawWaveWithOffset(PlotTitle,fileName,Panel)
    
    #### Back
    try:
        offSetType='Average All'
        PlotTitle='- Left Side Offset WAVE RAW DATA --->'+f+' offSetType='+offSetType;# Can modify Plot title
        fileName=f+" Left Side WaveResult_RawDataPerCycle Panel Number ";# Can modify File nmae
        side='Back'
        # db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        figLeftsideBACK=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerCycle(PlotTitle,offSetType,fileName,Panel)
        
        
        offSetType='Average Left Right' 
        PlotTitle='- Right Side Offset WAVE RAW DATA --->'+f+' offSetType='+offSetType;# Can modify Plot title
        fileName=f+" Right Side WaveResult_RawDataPerCycle Panel Number ";# Can modify File nmae
        side='Back'
        # db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        figRightsideBACK=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerCycle(PlotTitle,offSetType,fileName,Panel)
        
        # offSetType='Middle'
        # PlotTitle='- Middle Offset WAVE RAW DATA --->'+f;# Can modify Plot title
        # fileName=f+" Middle WaveResult_RawDataPerCycle Panel Number ";# Can modify File nmae
        # side='Back'
        # # db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        # figMiddlesideBACK=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerCycle(PlotTitle,offSetType,fileName,Panel)
        
        PlotTitle='- <b>STD </b> Side Offset WAVE RAW DATA --->'+f# Can modify Plot title
        fileName=f+" STD SideOffset_ WaveResult_RawDataPerColor Panel Number "# Can modify File nmae
        side='Back'
        figSTDBACK=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowSTDforRawWaveWithOffset(PlotTitle,fileName,Panel)
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
    # db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    figLeftsideFRONTperPanel=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
    
    
    
    offSetType='Average Left Right' 
    PlotTitle='- offSetType='+offSetType+' WAVE RAW DATA (For one Cycle)--->'+f;
    fileName=f+'- offSetType='+offSetType+" WaveResult_RawDataPerPanel ";
    side='Front'
    # db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    figRightsideFRONTperPanel=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
    
    # offSetType='Middle'
    # PlotTitle='- Middle Offset WAVE RAW DATA (For one Cycle)--->'+f;
    # fileName=f+" Right Side WaveResult_RawDataPerPanel ";
    # side='Front'
    # # db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
    # figMiddlesideFRONTperPanel=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
    
    
    #### Back
    try:
        offSetType='Average All'
        PlotTitle='- offSetType='+offSetType+' WAVE RAW DATA (For one Cycle)--->'+f;
        fileName=f+'- offSetType='+offSetType+" WaveResult_RawDataPerPanel ";
        side='Back'
        # db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        figLeftsideBACKperPanel=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
        
        
        
        offSetType='Average Left Right' 
        PlotTitle='- offSetType='+offSetType+' WAVE RAW DATA (For one Cycle)--->'+f;
        fileName=f+'- offSetType='+offSetType+" WaveResult_RawDataPerPanel ";
        side='Back'
        # db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        figRightsideBACKperPanel=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
        
        # offSetType='Middle'
        # PlotTitle='- Middle Offset WAVE RAW DATA (For one Cycle)--->'+f;
        # fileName=f+" Right Side WaveResult_RawDataPerPanel ";
        # side='Back'
        # # db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        # figMiddlesideBACKperPanel=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).ShowWaveRawData_SubOffset_PerPanel(PlotTitle,offSetType,fileName,CycleNumber)
    
    except:
        1

########################################################################################
#############################Plot Wave Data SUB Average Per Panel Per Cycle###################################
##### Front
if WaveDataSUBAverage_PerPanel_PerCycle:
    PlotTitle='- WAVE RAW DATA - Average Wave Data Per Panel--->'+f+' offSetType='+offSetType;
    fileName=f+" WAVE RAW DATA - Average Wave Data Per Panel ";
    offSetType='Average All' ;
    side='Front'
    figWaveDataSubAveragePerPanet=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotWaveDataSUBAveragePerPanelPerCycle(WaveRawDataDicFRONT,offSetType,PlotTitle,fileName);

    offSetType='Average Left Right' ;
    PlotTitle='- WAVE RAW DATA - Average Wave Data Per Panel--->'+f+' offSetType='+offSetType;
    fileName=f+" WAVE RAW DATA - Average Wave Data Per Panel ";
    side='Front'
    figWaveDataSubAveragePerPanet=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotWaveDataSUBAveragePerPanelPerCycle(WaveRawDataDicFRONT,offSetType,PlotTitle,fileName);
        
    #####Back
    try:
        PlotTitle='- WAVE RAW DATA - Average Wave Data Per Panel--->'+f+' offSetType='+offSetType;
        fileName=f+" WAVE RAW DATA - Average Wave Data Per Panel ";
        offSetType='Average All' #
        side='Back'
        figWaveDataSubAveragePerPanet=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotWaveDataSUBAveragePerPanelPerCycle(WaveRawDataDicBACK,offSetType,PlotTitle,fileName);
        
        offSetType='Average Left Right' #    
        PlotTitle='- WAVE RAW DATA - Average Wave Data Per Panel--->'+f+' offSetType='+offSetType;
        fileName=f+" WAVE RAW DATA - Average Wave Data Per Panel ";
        side='Back'
        figWaveDataSubAveragePerPanet=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotWaveDataSUBAveragePerPanelPerCycle(WaveRawDataDicBACK,offSetType,PlotTitle,fileName);

    except:
        1

##################################################################################
################################CIS Curve 
######FRONT
if CIScurve:

    try:
        PlotTitle='FRONT CIS curve --->'+f;
        fileName=f+' ';
        side='Front';
        cisCurve=cisFRONT
        figCISFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotCIScurve(cisCurve,PlotTitle,fileName);
    except:
        1
        
    #######BACK
    try:
        PlotTitle='BACK CIS curve --->'+f;
        fileName=f+' ';
        side='Back';
        cisCurve=cisBACK
        figCISBACK=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotCIScurve(cisCurve,PlotTitle,fileName); 
    except:
        1



##################################################################################
################################Registration Between Wave Prints
######FRONT  & BACK
if registrationBetweenWavePrints: ##Yuval method
    try:
        fileName=f
        side='Front'
        figRegistrationBetweenWavePrintsFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotRegistrationBetweenWavePrints(DFdicPerClrFRONT,MainColor,rgistBtwPntStartCycle,rgistBtwPntEndCycle,fileName)
    
        side='Back'
        figRegistrationBetweenWavePrintsBACK=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotRegistrationBetweenWavePrints(DFdicPerClrBACK,MainColor,rgistBtwPntStartCycle,rgistBtwPntEndCycle,fileName)
    except:
        1


##################################################################################
################################Wave Data Before and  After Aplied AVRG Correction
######FRONT 
if BeforAndAfterCorr:
    PlotTitle=' wave raw data before and after correction ---> '+f;
    fileName=f+'wave raw data before and after correction_';
    side='Front';
    figClrBeforeAndAfterFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotWaveDataAfterApliedAVRGCorrection(WaveRawDataDicFRONT,WaveRawDataDicAfterCorrFRONT,PlotTitle,fileName)
    
    ######BACK
    try:
        
        PlotTitle=' wave raw data before and after correction ---> '+f;
        fileName=f+'wave raw data before and after correction_';
        side='Back';
        figClrBeforeAndAfterBACK=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotWaveDataAfterApliedAVRGCorrection(WaveRawDataDicBACK,WaveRawDataDicAfterCorrBACK,PlotTitle,fileName)
    except:
        1

#################################################################################
##################################Plot Wave DataResidue After correction
########FRONT
if WaveFilterResidue_dxPlot:
    PlotTitle=' After Correction Wave Data S.Golay = '+ str(MaxWaveWindow)+'---> '+f
    fileName=f+'  After Correction Wave Data S.Golay _'+ str(MaxWaveWindow)
    side='Front';
    figWaveResidueAfterCorrFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotWaveDataResidue(WaveRawDataDicFRONT,WaveDataWithMaxFilterDicFRONT,PHlocFRONT,PHoffSetFRONT,PHtiltFRONT,PlotTitle,fileName)
    
    ########BACK
    try:
        PlotTitle=' After Correction Wave Data S.Golay = '+ str(MaxWaveWindow)+'---> '+f
        fileName=f+'  After Correction Wave Data S.Golay _'+ str(MaxWaveWindow)
        side='Back';
        figWaveResidueAfterCorrBACK=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotWaveDataResidue(WaveRawDataDicBACK,WaveDataWithMaxFilterDicBACK,PHlocBACK,PHoffSetBACK,PHtiltBACK,PlotTitle,fileName)
    except:
         1     


#################################################################################
####################### Table: Offset Table -  After correction
#############Front
if PlotTables:
    PlotTitle=' offset (Correction-For simplex) table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
    fileName=f+" Offset Table"
    side='Front';
    TableOffsetAfterCorrFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotOffsetTabel(PHoffsetPerHFRONTAfterCorr,PlotTitle,fileName)
    
    
    #####Back
    
    try:
        PlotTitle=' offset (Correction-For simplex) table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
        fileName=f+" Offset Table"
        side='Back';
        TableOffsetAfterCorrBACK=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotOffsetTabel(PHoffsetPerHBACKAfterCorrAfterCorr,PlotTitle,fileName)
    except:
     1     
    
    #################################################################################
    ####################### Table: Tilt Table -  After correction
    #############Front
    
    PlotTitle=' Tilt table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
    fileName=f+" Tilt Table"
    side='Front';
    TableTiltAfterCorrFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotTiltTable(PHtiltPerHFRONTAfterCorr,ColorLevelsTilt,DivideByNumTilt,PlotTitle,fileName)
    
    #####Back
    try:
        PlotTitle=' Tilt table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
        fileName=f+" Tilt Table"
        side='Back';
        TableTiltAfterCorrBACK=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotTiltTable(PHtiltPerHBACKAfterCorr,ColorLevelsTilt,DivideByNumTilt,PlotTitle,fileName)
    except:
        1
    
    #################################################################################
    ####################### Table: FRONT -BACK Delta - After correction
    
    try:
        PlotTitle='Delta offset table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
        fileName=f+" Delta Offset Table"
        side='Front';
        TableFRONT_BACK_AverageAfterCorrFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotFRONT_BACKDeltaTable(PHoffsetPerHFRONT,PHoffsetPerHBACK,DivideByNum,ColorLevels,PlotTitle,fileName);
    except:
        1
        
    #################################################################################
    ####################### Table: FRONT -BACK Average -  After correction
    try:
        PlotTitle='Correction table S.Golay = '+ str(MaxWaveWindow)+'---> '+f
        fileName=f+" FRONT -BACK Average Table"
        side='Front';
        TableFRONT_BACK_AverageAfterCorrFRONT=PlotGraphPlotly(pthF+'/',side,Panel,ColorList).PlotFRONT_BACKAverageTable(PHoffsetPerHFRONT,PHoffsetPerHBACK,PlotTitle,fileName);
    except:
        1

####################FRONT 2 BACK ###########################
# side='Front';
# ColorForDisplay= 'Cyan'
# dbFRONT=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);

# side='Back';
# dbBACK=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);


#############################################
#############################################
#############################################

# figF2B = go.Figure()

# for ColorForDisplay in ColorList:    
#     dbFRONT=CalcWaveFromRawData(pthF+'/','Front',9).ArrangeRawDataForAnalize(ColorForDisplay);
#     dbBACK=CalcWaveFromRawData(pthF+'/','Back',1).ArrangeRawDataForAnalize(ColorForDisplay);
#     if ColorForDisplay=='Yellow':
#         ColorForDisplay='gold'; 
#     col=list(db.columns)
            
#     rnge=range(len(col))
    
#     for i in rnge:    
#         figF2B.add_trace(
#         go.Scatter(y=list(dbFRONT[i+1]-dbBACK[i+1]),line_color= ColorForDisplay,
#                     name='Cycle '+str(i+1)+' '+'Panel '+str(9) +'and' +str(1)+' ' +ColorForDisplay))
    

# figF2B.update_layout(
#      hoverlabel=dict(
#          namelength=-1
#      )
#  )
# figF2B.update_layout(title=side+'- Front 2 Back --->'+f)

# plot(figF2B,filename=f+" Front 2 Back Panel Number "+str(Panel)+"_.html") 
#############################################
#############################################
#############################################    
    
# fig100 = go.Figure()
# fig101 = go.Figure()
# fig102 = go.Figure()
# side='Front';
# # fig00 = make_subplots(rows=3, cols=1,subplot_titles=("LeftSide","Middle", "RightSide"), vertical_spacing=0.1, shared_xaxes=True)
# WaveRawDataDicFRONT
# # rnge=[3,6,7]

# # db=ImagePlacement_Rightpp
# # db=ImagePlacement_pp
# # for ColorForDisplay in ColorList:
# for Panel in  range(1,12):  
#     for ColorForDisplay in ColorList:
#         db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        
        
#         col=list(db.columns)
        
#         rnge=range(len(col))
        
#         if ColorForDisplay=='Yellow':
#             ColorForDisplay='gold';
        
#         # for i in rnge:
#         # for i in rnge:
#             # if SideOffset=='LeftSide':
#             #     offSet=db[i+1][0];
#             # else:
#             #     if SideOffset=='RightSide':    
#             #         offSet=db[i+1][length(len(db[i+1]))]
#             #     else:
#             #         if  SideOffset=='Middle':
#             #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
#             #         else:
#             #             offSet=0;
#         offSet=db[CycleNumber][0];
#         fig100.add_trace(
#         go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
#                     name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
        
        
#         offSet=np.min(db[CycleNumber][int(len(db[CycleNumber])/2)-50:int(len(db[CycleNumber])/2)+50])
#         fig101.add_trace(
#         go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
#                     name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
        
#         offSet=db[CycleNumber][(len(db[CycleNumber]))-1]  
#         fig102.add_trace(
#         go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
#                     name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))



# fig100.update_layout(
#     hoverlabel=dict(
#         namelength=-1
#     )
# )
# fig101.update_layout(
#     hoverlabel=dict(
#         namelength=-1
#     )
# )
# fig102.update_layout(
#     hoverlabel=dict(
#         namelength=-1
#     )
# )
# fig100.update_layout(title=side+'- Left Side Offset WAVE RAW DATA (For one Cycle)--->'+f)
# fig101.update_layout(title=side+'- Middle Offset WAVE RAW DATA (For one Cycle)--->'+f)
# fig102.update_layout(title=side+'- Right Side Offset WAVE RAW DATA (For one Cycle)--->'+f)

# now = datetime.now()


# dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# # plot(fig00)
# if LeftSide:
#     plot(fig100,filename=f+" Left Side WaveResult_RawDataPerPanel "+side+".html") 
# if Middle:    
#     plot(fig101,filename=f+" Middle Side WaveResult_RawDataPerPanel "+side+".html") 
# if RightSide:
#     plot(fig102,filename=f+" Right Side WaveResult_RawDataPerPanel "+side+".html")

#########################################
#########################################
#########################################
# if printPerCycle:
#     if LeftSide+Middle+RightSide:
    
#         fig00 = go.Figure()
#         fig001 = go.Figure()
#         fig002 = go.Figure()
#         side='Front';
#         # fig00 = make_subplots(rows=3, cols=1,subplot_titles=("LeftSide","Middle", "RightSide"), vertical_spacing=0.1, shared_xaxes=True)
        
#         # rnge=[3,6,7]
        
#         # db=ImagePlacement_Rightpp
#         # db=ImagePlacement_pp
#         # for ColorForDisplay in ColorList:
#         for ColorForDisplay in ColorList:    
#             db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
            
#             if ColorForDisplay=='Yellow':
#                 ColorForDisplay='gold'; 
                
#             col=list(db.columns)
            
#             rnge=range(len(col))
            
#             for i in rnge:
#             # for i in rnge:
#                 # if SideOffset=='LeftSide':
#                 #     offSet=db[i+1][0];
#                 # else:
#                 #     if SideOffset=='RightSide':    
#                 #         offSet=db[i+1][length(len(db[i+1]))]
#                 #     else:
#                 #         if  SideOffset=='Middle':
#                 #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
#                 #         else:
#                 #             offSet=0;
#                 offSet=db[i+1][0];
#                 fig00.add_trace(
#                 go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
#                             name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
                
                
#                 offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
#                 fig001.add_trace(
#                 go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
#                             name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
                
#                 offSet=db[i+1][(len(db[i+1]))-1]  
#                 fig002.add_trace(
#                 go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
#                             name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
        
        
        
#         fig00.update_layout(
#             hoverlabel=dict(
#                 namelength=-1
#             )
#         )
#         fig001.update_layout(
#             hoverlabel=dict(
#                 namelength=-1
#             )
#         )
#         fig002.update_layout(
#             hoverlabel=dict(
#                 namelength=-1
#             )
#         )
#         fig00.update_layout(title=side+'- Left Side Offset WAVE RAW DATA --->'+f)
#         fig001.update_layout(title=side+'- Middle Side Offset WAVE RAW DATA --->'+f)
#         fig002.update_layout(title=side+'- Right Side Offset WAVE RAW DATA --->'+f)
        
#         now = datetime.now()
        
        
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#         # plot(fig00)
#         if LeftSide:
#             plot(fig00,filename=f+" Left Side WaveResult_RawDataPerCycle Panel Number "+str(Panel)+' '+side+".html") 
#         if Middle:    
#             plot(fig001,filename=f+" Middle Side WaveResult_RawDataPerCycle Panel Number "+str(Panel)+' '+side+".html") 
#         if RightSide:
#             plot(fig002,filename=f+" Right Side WaveResult_RawDataPerCycle Panel Number "+str(Panel)+' '+side+".html") 
        
        
#         ########## BACK ########
#         try:
#             fig000 = go.Figure()
#             fig011 = go.Figure()
#             fig022 = go.Figure()
#             side='Back';
#             # fig00 = make_subplots(rows=3, cols=1,subplot_titles=("LeftSide","Middle", "RightSide"), vertical_spacing=0.1, shared_xaxes=True)
            
#             # rnge=[3,6,7]
            
#             # db=ImagePlacement_Rightpp
#             # db=ImagePlacement_pp
#             # for ColorForDisplay in ColorList:
#             for ColorForDisplay in ColorList:    
#                 db=CalcWaveFromRawData(pthF+'/','Back',Panel).ArrangeRawDataForAnalize(ColorForDisplay);
                
#                 if ColorForDisplay=='Yellow':
#                     ColorForDisplay='gold';
                    
#                 col=list(db.columns)
                
#                 rnge=range(len(col))
                
#                 for i in rnge:
#                 # for i in rnge:
#                     # if SideOffset=='LeftSide':
#                     #     offSet=db[i+1][0];
#                     # else:
#                     #     if SideOffset=='RightSide':    
#                     #         offSet=db[i+1][length(len(db[i+1]))]
#                     #     else:
#                     #         if  SideOffset=='Middle':
#                     #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
#                     #         else:
#                     #             offSet=0;
#                     offSet=db[i+1][0];
#                     fig000.add_trace(
#                     go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
#                                 name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
                    
                    
#                     offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
#                     fig011.add_trace(
#                     go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
#                                 name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
                    
#                     offSet=db[i+1][(len(db[i+1]))-1]  
#                     fig022.add_trace(
#                     go.Scatter(y=list(db[i+1]-offSet),line_color= ColorForDisplay,
#                                 name='Cycle '+str(i+1)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
            
            
            
#             fig000.update_layout(
#                 hoverlabel=dict(
#                     namelength=-1
#                 )
#             )
#             fig011.update_layout(
#                 hoverlabel=dict(
#                     namelength=-1
#                 )
#             )
#             fig022.update_layout(
#                 hoverlabel=dict(
#                     namelength=-1
#                 )
#             )
            
#             fig000.update_layout(title=side+'- Left Side Offset WAVE RAW DATA --->'+f)
#             fig011.update_layout(title=side+'- Middle Side Offset WAVE RAW DATA --->'+f)
#             fig022.update_layout(title=side+'- Right Side Offset WAVE RAW DATA --->'+f)
        
            
#             now = datetime.now()
            
            
#             dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            
#             if LeftSide:
#                 plot(fig000,filename=f+' Left Side WaveResult_RawDataPerCycle Panel Number '+str(Panel)+' '+side+".html") 
#             if Middle:
#                 plot(fig011,filename=f+' Middle Side WaveResult_RawDataPerCycle Panel Number '+str(Panel)+' '+side+".html") 
#             if RightSide:
#                 plot(fig022,filename=f+' Right Side WaveResult_RawDataPerCycle Panel Number '+str(Panel)+' '+side+".html") 
#         except:
#             1    
        
#     #########################################
#     #########################################
#     #########################################
    
    # if LeftSide+Middle+RightSide:
    
    #     fig01 = go.Figure()
    #     side='Front';
    #     # rnge=[3,6,7]
        
    #     # db=ImagePlacement_Rightpp
    #     # db=ImagePlacement_pp
    #     # for ColorForDisplay in ColorList:
    #     for ColorForDisplay in ColorList:
    #         db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
        
    #         if ColorForDisplay=='Yellow':
    #             ColorForDisplay='gold';    
            
    #         col=list(db.columns)
            
    #         rnge=range(len(col))
    #         middledb=pd.DataFrame()
    #         Rightdb=pd.DataFrame()
    #         Leftdb=pd.DataFrame()
            
    #         middleSTD=[]
    #         RightSTD=[]
    #         LeftSTD=[]
    #         for i in rnge:
    #         # for i in rnge:
    #             # if SideOffset=='LeftSide':
    #             #     offSet=db[i+1][0];
    #             # else:
    #             #     if SideOffset=='RightSide':    
    #             #         offSet=db[i+1][length(len(db[i+1]))]
    #             #     else:
    #             #         if  SideOffset=='Middle':
    #             #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
    #             #         else:
    #             #             offSet=0;
                
    #             offSet1=db[i+1][0];
                
    #             Leftdb=pd.concat([Leftdb,db[i+1]-offSet1],axis=1);
                
    #             offSet2=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
    #             middledb=pd.concat([middledb,db[i+1]-offSet2],axis=1);
                
    #             offSet3=db[i+1][(len(db[i+1]))-1] 
    #             Rightdb=pd.concat([Rightdb,db[i+1]-offSet3],axis=1);
            
            
    #         for j,i in enumerate(Leftdb.index):
    #             if j == 0:
    #                 continue;
    #             LeftSTD.append(np.std(Leftdb.loc[i,:]))
    #             middleSTD.append(np.std(middledb.loc[i,:]))
    #             RightSTD.append(np.std(Rightdb.loc[i,:]))
               
            
#             fig01.add_trace(
#             go.Scatter(y=LeftSTD,line_color= ColorForDisplay,
#                         name='Panel '+str(Panel)+' ' +ColorForDisplay+' LeftSide'))
            
            
#             fig01.add_trace(
#             go.Scatter(y=middleSTD,line_color= ColorForDisplay,
#                         name='Panel '+str(Panel)+' ' +ColorForDisplay+' Middle'))
            
#             fig01.add_trace(
#             go.Scatter(y=RightSTD,line_color= ColorForDisplay,
#                         name='Panel '+str(Panel)+' ' +ColorForDisplay+' RightSide'))
        
        
        
#         fig01.update_layout(
#             hoverlabel=dict(
#                 namelength=-1
#             )
#         )
#         fig01.update_layout(title=side+'- <b>STD </b> Side Offset WAVE RAW DATA --->'+f )
        
#         now = datetime.now()
        
        
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#         if LeftSide or Middle or RightSide:
#         # plot(fig00)
#             plot(fig01,filename=f+" STD SideOffset_ WaveResult_RawDataPerColor Panel Number "+str(Panel)+' '+side+".html") 
        
#         ################# Back  ########################
#         try:
#             fig010 = go.Figure()
#             side='Back';
#         # rnge=[3,6,7]
        
#         # db=ImagePlacement_Rightpp
#         # db=ImagePlacement_pp
#         # for ColorForDisplay in ColorList:
#             for ColorForDisplay in ColorList:
#                 db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
                
#                 if ColorForDisplay=='Yellow':
#                     ColorForDisplay='gold';
                    
#                 col=list(db.columns)
                
#                 rnge=range(len(col))
#                 middledb=pd.DataFrame()
#                 Rightdb=pd.DataFrame()
#                 Leftdb=pd.DataFrame()
                
#                 middleSTD=[]
#                 RightSTD=[]
#                 LeftSTD=[]
#                 for i in rnge:
#                 # for i in rnge:
#                     # if SideOffset=='LeftSide':
#                     #     offSet=db[i+1][0];
#                     # else:
#                     #     if SideOffset=='RightSide':    
#                     #         offSet=db[i+1][length(len(db[i+1]))]
#                     #     else:
#                     #         if  SideOffset=='Middle':
#                     #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
#                     #         else:
#                     #             offSet=0;
                    
#                     offSet1=db[i+1][0];
                    
#                     Leftdb=pd.concat([Leftdb,db[i+1]-offSet1],axis=1);
                    
#                     offSet2=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
#                     middledb=pd.concat([middledb,db[i+1]-offSet2],axis=1);
                    
#                     offSet3=db[i+1][(len(db[i+1]))-1] 
#                     Rightdb=pd.concat([Rightdb,db[i+1]-offSet3],axis=1);
                
                
#                 for j,i in enumerate(Leftdb.index):
#                     if j == 0:
#                         continue;
#                     LeftSTD.append(np.std(Leftdb.loc[i,:]))
#                     middleSTD.append(np.std(middledb.loc[i,:]))
#                     RightSTD.append(np.std(Rightdb.loc[i,:]))
                   
                
#                 fig010.add_trace(
#                 go.Scatter(y=LeftSTD,line_color= ColorForDisplay,
#                             name='Panel '+str(Panel)+' ' +ColorForDisplay+' LeftSide'))
                
                
#                 fig010.add_trace(
#                 go.Scatter(y=middleSTD,line_color= ColorForDisplay,
#                             name='Panel '+str(Panel)+' ' +ColorForDisplay+' Middle'))
                
#                 fig010.add_trace(
#                 go.Scatter(y=RightSTD,line_color= ColorForDisplay,
#                             name='Panel '+str(Panel)+' ' +ColorForDisplay+' RightSide'))
            
            
            
#             fig010.update_layout(
#                 hoverlabel=dict(
#                     namelength=-1
#                 )
#             )
#             fig010.update_layout(title=side+'- <b>STD </b> Side Offset WAVE RAW DATA --->'+f)
            
#             now = datetime.now()
            
            
#             dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#             # plot(fig00)
#             if LeftSide or Middle or RightSide:
        
#                 plot(fig010,filename=f+"STD SideOffset_ WaveResult_RawDataPerColor Panel Number "+str(Panel)+' '+side+".html") 
#         except:
#             1    
# #########################################
# #########################################
# #########################################

# #################PANEL##################
# if plotPerPanel:
#     if LeftSide+Middle+RightSide:
    
#         fig100 = go.Figure()
#         fig101 = go.Figure()
#         fig102 = go.Figure()
#         side='Front';
#         # fig00 = make_subplots(rows=3, cols=1,subplot_titles=("LeftSide","Middle", "RightSide"), vertical_spacing=0.1, shared_xaxes=True)
        
#         # rnge=[3,6,7]
        
#         # db=ImagePlacement_Rightpp
#         # db=ImagePlacement_pp
#         # for ColorForDisplay in ColorList:
#         for Panel in  range(1,12):  
#             for ColorForDisplay in ColorList:
#                 WaveRawDataDicFRONT=CalcWaveFromRawData(pthF+'/',side,Panel).CreateDicOfWaveRawData();
                
                
                
#                 col=list(db.columns)
                
#                 rnge=range(len(col))
                
#                 if ColorForDisplay=='Yellow':
#                     ColorForDisplay='gold';
                
#                 # for i in rnge:
#                 # for i in rnge:
#                     # if SideOffset=='LeftSide':
#                     #     offSet=db[i+1][0];
#                     # else:
#                     #     if SideOffset=='RightSide':    
#                     #         offSet=db[i+1][length(len(db[i+1]))]
#                     #     else:
#                     #         if  SideOffset=='Middle':
#                     #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
#                     #         else:
#                     #             offSet=0;
#                 offSet=db[CycleNumber][0];
#                 fig100.add_trace(
#                 go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
#                             name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
                
                
#                 offSet=np.min(db[CycleNumber][int(len(db[CycleNumber])/2)-50:int(len(db[CycleNumber])/2)+50])
#                 fig101.add_trace(
#                 go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
#                             name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
                
#                 offSet=db[CycleNumber][(len(db[CycleNumber]))-1]  
#                 fig102.add_trace(
#                 go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
#                             name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
        
        
        
#         fig100.update_layout(
#             hoverlabel=dict(
#                 namelength=-1
#             )
#         )
#         fig101.update_layout(
#             hoverlabel=dict(
#                 namelength=-1
#             )
#         )
#         fig102.update_layout(
#             hoverlabel=dict(
#                 namelength=-1
#             )
#         )
#         fig100.update_layout(title=side+'- Left Side Offset WAVE RAW DATA (For one Cycle)--->'+f)
#         fig101.update_layout(title=side+'- Middle Offset WAVE RAW DATA (For one Cycle)--->'+f)
#         fig102.update_layout(title=side+'- Right Side Offset WAVE RAW DATA (For one Cycle)--->'+f)
        
#         now = datetime.now()
        
        
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#         # plot(fig00)
#         if LeftSide:
#             plot(fig100,filename=f+" Left Side WaveResult_RawDataPerPanel "+side+".html") 
#         if Middle:    
#             plot(fig101,filename=f+" Middle Side WaveResult_RawDataPerPanel "+side+".html") 
#         if RightSide:
#             plot(fig102,filename=f+" Right Side WaveResult_RawDataPerPanel "+side+".html") 
        
        
#         ########## BACK ########
#         try:
#             fig110 = go.Figure()
#             fig111 = go.Figure()
#             fig122 = go.Figure()
#             side = 'Back'
#             # fig00 = make_subplots(rows=3, cols=1,subplot_titles=("LeftSide","Middle", "RightSide"), vertical_spacing=0.1, shared_xaxes=True)
            
#             # rnge=[3,6,7]
            
#             # db=ImagePlacement_Rightpp
#             # db=ImagePlacement_pp
#             # for ColorForDisplay in ColorList:
#             for Panel in  range(1,12):  
#                 for ColorForDisplay in ColorList:
#                     db=CalcWaveFromRawData(pthF+'/',side,Panel).ArrangeRawDataForAnalize(ColorForDisplay);
                    
                    
#                     col=list(db.columns)
                    
#                     rnge=range(len(col))
                    
#                     # for i in rnge:
#                     # for i in rnge:
#                         # if SideOffset=='LeftSide':
#                         #     offSet=db[i+1][0];
#                         # else:
#                         #     if SideOffset=='RightSide':    
#                         #         offSet=db[i+1][length(len(db[i+1]))]
#                         #     else:
#                         #         if  SideOffset=='Middle':
#                         #             offSet=np.min(db[i+1][int(len(db[i+1])/2)-50:int(len(db[i+1])/2)+50])
#                         #         else:
#                         #             offSet=0;
            
#                     if ColorForDisplay=='Yellow':
#                         ColorForDisplay='gold';    
                        
#                     offSet=db[CycleNumber][0];
#                     fig110.add_trace(
#                     go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
#                                 name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
                    
                    
#                     offSet=np.min(db[CycleNumber][int(len(db[CycleNumber])/2)-50:int(len(db[CycleNumber])/2)+50])
#                     fig111.add_trace(
#                     go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
#                                 name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
                    
#                     offSet=db[CycleNumber][(len(db[CycleNumber]))-1]  
#                     fig112.add_trace(
#                     go.Scatter(y=list(db[CycleNumber]-offSet),line_color= ColorForDisplay,
#                                 name='Cycle '+str(CycleNumber)+' '+'Panel '+str(Panel)+' ' +ColorForDisplay))
        
        
        
#             fig110.update_layout(
#                 hoverlabel=dict(
#                     namelength=-1
#                 )
#             )
#             fig111.update_layout(
#                 hoverlabel=dict(
#                     namelength=-1
#                 )
#             )
#             fig112.update_layout(
#                 hoverlabel=dict(
#                     namelength=-1
#                 )
#             )
#             fig110.update_layout(title=side+'- Left Side Offset WAVE RAW DATA (For one Cycle)--->'+f)
#             fig111.update_layout(title=side+'- Middle Offset WAVE RAW DATA (For one Cycle)--->'+f)
#             fig112.update_layout(title=side+'- Right Side Offset WAVE RAW DATA (For one Cycle)--->'+f)
            
     
            
#             now = datetime.now()
            
            
#             dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#             # plot(fig00)
#             if LeftSide:
#                 plot(fig110,filename=f+" Left Side WaveResult_RawDataPerPanel "+side+".html") 
#             if Middle:    
#                 plot(fig111,filename=f+" Middle Side WaveResult_RawDataPerPanel "+side+".html") 
#             if RightSide:
#                 plot(fig112,filename=f+" Right Side WaveResult_RawDataPerPanel "+side+".html") 
         
#         except:
#             1    
    
# #########################################
# #########################################
# #########################################
# #########################################
# #########################################
# #########################################
# ######CIS############
# if CIScurve:
#     try:
#         fig012 = go.Figure()
    
                
#         fig012.add_trace(
#         go.Scatter(y=cisFRONT,
#                     name='CIS FRONT curve'))
                
      
    
    
#         fig012.update_layout(
#             hoverlabel=dict(
#                 namelength=-1
#             )
#         )
#         fig012.update_layout(title='FRONT CIS curve --->'+f )
        
#         now = datetime.now()
        
        
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#         # plot(fig00)
#         plot(fig012,filename=f+"CIScurveFRONT.html") 
    
#     except:
#      1
#     if len(cisFRONT)<1:
#         print('************************************************************************************')
#         print(f+' Has No CIS FRONT curve information')
#         print('************************************************************************************')
      
#     ##### BACK
#     try:
#         fig013 = go.Figure()
    
                
#         fig013.add_trace(
#         go.Scatter(y=cisBACK,
#                     name='CIS BACK curve'))
                
      
    
    
#         fig013.update_layout(
#             hoverlabel=dict(
#                 namelength=-1
#             )
#         )
#         fig013.update_layout(title='BACK CIS curve --->'+f)
        
#         now = datetime.now()
        
        
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#         # plot(fig00)
#         plot(fig013,filename=f+"CIScurveBACK.html") 
    
#     except:
#      1
#     if len(cisBACK)<1:
#         print('************************************************************************************')
#         print(f+' Has No CIS BACK curve information')
#         print('************************************************************************************')    

#  ##################################################################################################       
#  ##################################################################################################       
#  ##################################################################################################       
#  ##################################################################################################       
# try:
#     if registrationBetweenWavePrints:
        
#         figClrFRONT = go.Figure()
            
#         side='Front'    
#         for clr in ColorList:
#             if clr == MainColor:
#                 continue;
            
#             for col in range(rgistBtwPntStartCycle,rgistBtwPntEndCycle+1):        
#             # for col in DFdicPerClr[clr].columns:
                
#                 figClrFRONT.add_trace(
#                 go.Scatter(y=DFdicPerClrFRONT[clr][col],line_color= clr,
#                             name='Registration for cycle '+str(col)+' color '+clr))
        
        
#             figClrFRONT.update_layout(
#                     hoverlabel=dict(
#                         namelength=-1
#                     )
#                 )
#             figClrFRONT.update_layout(title=side+' wave registration normalized to '+MainColor+' for Cycle Start ='+str(rgistBtwPntStartCycle)+' Cycle End='+str(rgistBtwPntEndCycle)+' ---> '+f)
            
#         now = datetime.now()
        
        
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#             # plot(fig00)
#         plot(figClrFRONT,filename=f+'Registration for cycle '+str(rgistBtwPntStartCycle)+'_'+str(rgistBtwPntEndCycle)+'_'+side+".html") 
        
        
#         ####BACK
        
#         figClrBACK = go.Figure()
            
#         side='Back'    
#         for clr in ColorList:
#             if clr == MainColor:
#                 continue;
            
#             for col in range(rgistBtwPntStartCycle,rgistBtwPntEndCycle+1):        
#             # for col in DFdicPerClr[clr].columns:
                
#                 figClrBACK.add_trace(
#                 go.Scatter(y=DFdicPerClrBACK[clr][col],line_color= clr,
#                             name='Registration for cycle '+str(col)+' color '+clr))
        
        
#             figClrBACK.update_layout(
#                     hoverlabel=dict(
#                         namelength=-1
#                     )
#                 )
#             figClrBACK.update_layout(title=side+' wave registration normalized to '+MainColor+' for Cycle Start ='+str(rgistBtwPntStartCycle)+' Cycle End='+str(rgistBtwPntEndCycle)+' ---> '+f)
            
#         now = datetime.now()
        
        
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#             # plot(fig00)
#         plot(figClrBACK,filename=f+'Registration for cycle '+str(rgistBtwPntStartCycle)+'_'+str(rgistBtwPntEndCycle)+'_'+side+".html") 
# except:
#     1      
   
# ##################################################################################################       
#  ##################################################################################################       
#  ##################################################################################################       
#  ##################################################################################################          
# if presentAllColors:
#     for clr in ColorList:
#         figPH = make_subplots(specs=[[{"secondary_y": True}]])
#         for col in WaveRawDataDicFRONT[clr].columns:
            
#             figPH.add_trace(
#             go.Scatter(y=WaveRawDataDicFRONT[clr][col],line_color= clr,
#                         name='WaveData Raw '+str(col)+' color '+clr), secondary_y=False)
            
#             figPH.add_trace(
#             go.Scatter(y=WaveDataWithMaxFilterDicFRONT[clr][col],line_color= clr,
#                         name='WaveData with Filter '+str(col)+' color '+clr), secondary_y=False)
            
#             figPH.add_trace(
#             go.Scatter(y=WaveRawDataDicFRONT[clr][col]-WaveDataWithMaxFilterDicFRONT[clr][col],line_color= clr,
#                         name='Fiter - Raw '+str(col)+' color '+clr), secondary_y=True)
        
        
#         figPH.update_layout(
#                 hoverlabel=dict(
#                     namelength=-1
#                 )
#             )
#         figPH.update_layout(title=f+'Wave Data - Filtered color '+clr+' Max Filter = '+ str(MaxWaveWindow))
        
#         now = datetime.now()
        
        
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#         # plot(fig00)
#         plot(figPH,filename=f+'Wave Data - Filtered color '+clr+' Max Filter_'+ str(MaxWaveWindow)+".html") 
 
# ##################################################################################################       
#  ##################################################################################################       
#  ##################################################################################################       
#  ##################################################################################################          
# ### Before and after correction
# if BeforAndAfterCorr:
    
#     figClrBeforeAndAfterFRONT = go.Figure()
#     col='Mean'       
#     side='Front'
#     for clr in ColorList:
        
#         OffsetBefore= WaveRawDataDicFRONT[clr][col][0] 
#         OffsetAfter= WaveRawDataDicAfterCorrFRONT[clr][0] 

    
#         figClrBeforeAndAfterFRONT.add_trace(
#                 go.Scatter(y=list(WaveRawDataDicFRONT[clr][col]-OffsetBefore),line_color= clr,line=dict(dash='dot'),
#                             name='Wave Raw Data Before Corr '+ clr))
#         figClrBeforeAndAfterFRONT.add_trace(
#                 go.Scatter(y=list(WaveRawDataDicAfterCorrFRONT[clr]-OffsetBefore),line_color= clr,
#                             name='Wave Raw Data After Corr '+ clr))
#     figClrBeforeAndAfterFRONT.update_layout(title=side+' wave raw data before and after correction ---> '+f)
            
#     now = datetime.now()
    
    
#     dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#         # plot(fig00)
#     plot(figClrBeforeAndAfterFRONT,filename=f+'wave raw data before and after correction_'+side+".html") 
    
#     try:
#         figClrBeforeAndAfterBACK = go.Figure()
#         col='Mean'       
#         side='Back'    
#         for clr in ColorList:
        
#             OffsetBefore= WaveRawDataDicBACK[clr][col][0] 
#             OffsetAfter= WaveRawDataDicAfterCorrBACK[clr][0] 
            
#             figClrBeforeAndAfterBACK.add_trace(
#                     go.Scatter(y=list(WaveRawDataDicBACK[clr][col]-OffsetBefore),line_color= clr,line=dict(dash='dot'),
#                                 name='Wave Raw Data Before Corr '+ clr))
#             figClrBeforeAndAfterBACK.add_trace(
#                     go.Scatter(y=list(WaveRawDataDicAfterCorrBACK[clr]-OffsetBefore),line_color= clr,
#                                 name='Wave Raw Data After Corr '+ clr))
#         figClrBeforeAndAfterBACK.update_layout(title=side+' wave raw data before and after correction ---> '+f)
                
#         now = datetime.now()
        
        
#         dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#             # plot(fig00)
#         plot(figClrBeforeAndAfterBACK,filename=f+'wave raw data before and after correction_'+side+".html") 
#     except:
#         1
# ##################################################################################################       
#  ##################################################################################################       
#  ##################################################################################################       
#  ##################################################################################################  
# if WaveFilterResidue:
#     if ShowBefore:
#         figPH = make_subplots(specs=[[{"secondary_y": True}]])
#         col='Mean';
#         side='Front'
#         for clr in ColorList:     
#             lineColor=clr;
          
            
#             if lineColor=='Yellow':
#                 lineColor='gold';
            
#             figPH.add_trace(
#             go.Scatter(y=WaveRawDataDicFRONT[clr][col],line_color= lineColor,
#                         name='WaveData Raw '+str(col)+' color '+clr), secondary_y=False)
            
#             figPH.add_trace(
#             go.Scatter(y=WaveDataWithMaxFilterDicFRONT[clr][col],line_color= lineColor,
#                         name='WaveData with Filter '+str(col)+' color '+clr), secondary_y=False)
            
#             figPH.add_trace(
#             go.Scatter(y=WaveRawDataDicFRONT[clr][col]-WaveDataWithMaxFilterDicFRONT[clr][col],line_color= lineColor,
#                         name='Fiter - Raw '+str(col)+' color '+clr), secondary_y=True)
            
#             ymax=max(WaveRawDataDicFRONT[ColorList[0]][col]-WaveDataWithMaxFilterDicFRONT[ColorList[0]][col])
            
#             for i,PHlocMem in enumerate(PHlocFRONT):
#                 figPH.add_trace(go.Scatter(x=[PHlocMem], y=[ymax],
#                                         marker=dict(color="green", size=6),
#                                         mode="markers",
#                                         text='PH #'+str(i),
#                                         # font_size=18,
#                                         hoverinfo='text'),secondary_y=True)
#                 figPH.data[len(figPH.data)-1].showlegend = False
        
#                 figPH.add_vline(x=PHlocMem, line_width=2, line_dash="dash", line_color="green")
            
            
#             if DisplayOffSet:
#                 figPH.add_trace(
#                 go.Scatter(y=PHoffSetFRONT[clr],line_color= lineColor,
#                             name='Average(Fiter - Raw) '+str(col)+' color '+clr), secondary_y=True)
                
            
#             if DisplayTilt:
#                 figPH.add_trace(
#                 go.Scatter(y=PHtiltFRONT[clr],line_color= lineColor,line=dict(dash='dot'),
#                             name='Tilt(Fiter - Raw) '+str(col)+' color '+clr), secondary_y=True)
            
            
#             figPH.update_layout(
#                     hoverlabel=dict(
#                         namelength=-1
#                     )
#                 )
#             figPH.update_layout(title=side+' Wave Data S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
            
#             now = datetime.now()
            
            
#             dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#             # plot(fig00)
#         plot(figPH,filename=f+' '+side+' Wave Data S.Golay _'+ str(MaxWaveWindow)+".html") 
         
#         ## Back ##
#         try:
#             figPHBACK = make_subplots(specs=[[{"secondary_y": True}]])
#             col='Mean';
#             side='Back'
#             for clr in ColorList:     
#                 lineColor=clr;
                
#                 if lineColor=='Yellow':
#                     lineColor='gold';
                
#                 figPHBACK.add_trace(
#                 go.Scatter(y=WaveRawDataDicBACK[clr][col],line_color= lineColor,
#                             name='WaveData Raw '+str(col)+' color '+clr), secondary_y=False)
                
#                 figPHBACK.add_trace(
#                 go.Scatter(y=WaveDataWithMaxFilterDicBACK[clr][col],line_color= lineColor,
#                             name='WaveData with Filter '+str(col)+' color '+clr), secondary_y=False)
                
#                 figPHBACK.add_trace(
#                 go.Scatter(y=WaveRawDataDicBACK[clr][col]-WaveDataWithMaxFilterDicBACK[clr][col],line_color= lineColor,
#                             name='Fiter - Raw '+str(col)+' color '+clr), secondary_y=True)
                
                
#                 for i,PHlocMem in enumerate(PHlocBACK):
#                     figPHBACK.add_trace(go.Scatter(x=[PHlocMem], y=[ymax],
#                                 marker=dict(color="green", size=6),
#                                 mode="markers",
#                                 text='PH #'+str(i),
#                                 # font_size=18,
#                                 hoverinfo='text'),secondary_y=True)
#                     figPHBACK.data[len(figPHBACK.data)-1].showlegend = False
        
#                     figPHBACK.add_vline(x=PHlocMem, line_width=2, line_dash="dash", line_color="green")
                
                
#                 if DisplayOffSet:
#                     figPHBACK.add_trace(
#                     go.Scatter(y=PHoffSetBACK[clr],line_color= lineColor,
#                                 name='Average(Fiter - Raw) '+str(col)+' color '+clr), secondary_y=True)
                
#                 if DisplayTilt:
#                     figPHBACK.add_trace(
#                     go.Scatter(y=PHtiltBACK[clr],line_color= lineColor,line=dict(dash='dot'),
#                                 name='Tilt(Fiter - Raw) '+str(col)+' color '+clr), secondary_y=True)
                
                
#                 figPHBACK.update_layout(
#                         hoverlabel=dict(
#                             namelength=-1
#                         )
#                     )
#                 figPHBACK.update_layout(title=side+' Wave Data S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
                
#                 now = datetime.now()
                
                
#                 dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#                 # plot(fig00)
#             plot(figPHBACK,filename=f+' '+side+' Wave Data S.Golay _'+ str(MaxWaveWindow)+".html") 
#         except:
#             1    
     
#     ################################AFTER CORRECTION
#     ################################################  
#     if ShowAfter:     
#         WaveRawDataDicAfterCorrFRONT,WaveDataWithMaxFilterDicAfterCorrFRONT
        
#         figPHCorr = make_subplots(specs=[[{"secondary_y": True}]])
#         col='Mean';
#         side='Front'
#         for clr in ColorList:     
#             lineColor=clr;
          
            
#             if lineColor=='Yellow':
#                 lineColor='gold';
            
#             figPHCorr.add_trace(
#             go.Scatter(y=WaveRawDataDicAfterCorrFRONT[clr],line_color= lineColor,
#                         name='WaveData Raw AfterCorr '+str(col)+' color '+clr), secondary_y=False)
            
#             figPHCorr.add_trace(
#             go.Scatter(y=WaveDataWithMaxFilterDicAfterCorrFRONT[clr],line_color= lineColor,
#                         name='WaveData AfterCorr with Filter '+str(col)+' color '+clr), secondary_y=False)
            
#             figPHCorr.add_trace(
#             go.Scatter(y=WaveRawDataDicAfterCorrFRONT[clr]-WaveDataWithMaxFilterDicAfterCorrFRONT[clr],line_color= lineColor,
#                         name='Fiter - Raw AfterCorr '+str(col)+' color '+clr), secondary_y=True)
            
#             ymax=max(WaveRawDataDicAfterCorrFRONT[ColorList[0]]-WaveDataWithMaxFilterDicAfterCorrFRONT[ColorList[0]])
            
#             for i,PHlocMem in enumerate(PHlocFRONT):
#                 figPHCorr.add_trace(go.Scatter(x=[PHlocMem], y=[ymax],
#                                         marker=dict(color="green", size=6),
#                                         mode="markers",
#                                         text='PH #'+str(i),
#                                         # font_size=18,
#                                         hoverinfo='text'),secondary_y=True)
#                 figPHCorr.data[len(figPHCorr.data)-1].showlegend = False
        
#                 figPHCorr.add_vline(x=PHlocMem, line_width=2, line_dash="dash", line_color="green")
            
            
#             if DisplayOffSet:
#                 figPHCorr.add_trace(
#                 go.Scatter(y=PHoffSetFRONTAfterCorr[clr],line_color= lineColor,
#                             name='Average(Fiter - Raw) AfterCorr '+str(col)+' color '+clr), secondary_y=True)
                
            
#             if DisplayTilt:
#                 figPHCorr.add_trace(
#                 go.Scatter(y=PHtiltFRONTAfterCorr[clr],line_color= lineColor,line=dict(dash='dot'),
#                             name='Tilt(Fiter - Raw) AfterCorr '+str(col)+' color '+clr), secondary_y=True)
            
            
#             figPHCorr.update_layout(
#                     hoverlabel=dict(
#                         namelength=-1
#                     )
#                 )
#             figPHCorr.update_layout(title=side+' After Correction Wave Data S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
            
#             now = datetime.now()
            
            
#             dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#             # plot(fig00)
#         plot(figPHCorr,filename=f+' '+side+' After Correction Wave Data S.Golay _'+ str(MaxWaveWindow)+".html") 
         
#         ## Back ##
#         try:
#             figPHBACKAfterCorr = make_subplots(specs=[[{"secondary_y": True}]])
#             col='Mean';
#             side='Back'
#             for clr in ColorList:     
#                 lineColor=clr;
                
#                 if lineColor=='Yellow':
#                     lineColor='gold';
                
#                 figPHBACKAfterCorr.add_trace(
#                 go.Scatter(y=WaveRawDataDicAfterCorrBACK[clr],line_color= lineColor,
#                             name='WaveData AfterCorr Raw '+str(col)+' color '+clr), secondary_y=False)
                
#                 figPHBACKAfterCorr.add_trace(
#                 go.Scatter(y=WaveDataWithMaxFilterDicAfterCorrBACK[clr],line_color= lineColor,
#                             name='WaveData AfterCorr with Filter '+str(col)+' color '+clr), secondary_y=False)
                
#                 figPHBACKAfterCorr.add_trace(
#                 go.Scatter(y=WaveRawDataDicAfterCorrBACK[clr]-WaveDataWithMaxFilterDicAfterCorrBACK[clr],line_color= lineColor,
#                             name='Fiter - Raw AfterCorr '+str(col)+' color '+clr), secondary_y=True)
                
                
#                 for i,PHlocMem in enumerate(PHlocBACK):
#                     figPHBACKAfterCorr.add_trace(go.Scatter(x=[PHlocMem], y=[ymax],
#                                 marker=dict(color="green", size=6),
#                                 mode="markers",
#                                 text='PH #'+str(i),
#                                 # font_size=18,
#                                 hoverinfo='text'),secondary_y=True)
#                     figPHBACKAfterCorr.data[len(figPHBACKAfterCorr.data)-1].showlegend = False
        
#                     figPHBACKAfterCorr.add_vline(x=PHlocMem, line_width=2, line_dash="dash", line_color="green")
                
                
#                 if DisplayOffSet:
#                     figPHBACKAfterCorr.add_trace(
#                     go.Scatter(y=PHoffSetBACKAfterCorr[clr],line_color= lineColor,
#                                 name='Average(Fiter - Raw) AfterCorr '+str(col)+' color '+clr), secondary_y=True)
                
#                 if DisplayTilt:
#                     figPHBACKAfterCorr.add_trace(
#                     go.Scatter(y=PHtiltBACKAfterCorr[clr],line_color= lineColor,line=dict(dash='dot'),
#                                 name='Tilt(Fiter - Raw) AfterCorr '+str(col)+' color '+clr), secondary_y=True)
                
                
#                 figPHBACKAfterCorr.update_layout(
#                         hoverlabel=dict(
#                             namelength=-1
#                         )
#                     )
#                 figPHBACKAfterCorr.update_layout(title=side+' AfterCorr Wave Data S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
                
#                 now = datetime.now()
                
                
#                 dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#                 # plot(fig00)
#             plot(figPHBACKAfterCorr,filename=f+' '+side+' AfterCorr Wave Data S.Golay _'+ str(MaxWaveWindow)+".html") 
#         except:
#             1    
        
    
    
    
    
    
    
    
    
    
     
#     ##################################################################################################       
#     ##################################################################################################       
#     ##################################################################################################       
#     ##################################################################################################      
    
#     PHname=[]
#     header=[]
#     ListofListFRONT=[]
#     ListofListBACK=[]
    
#     headerTilt=[]
#     ListofListTiltFRONT=[]
#     ListofListTiltBACK=[]
    
#     side='Front'
#     for i in range(24):
#         PHname.append('PH NUMBER# '+str(i))
    
#     for col in ColorList:
#         header.append(col+' Offset')
#         # header.append(col+' Tilt')
#         new_list = [-number for number in PHoffsetPerHFRONT[col]]
#         ListofListFRONT.append(new_list)
#         # ListofList.append(PHtiltPerH[col])
#     ####FRONT 
#     figTableFRONT = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
#                      cells=dict(values=[PHname]+ListofListFRONT,font=dict(color='black', size=15)))
#                          ])
    
#     figTableFRONT.update_layout(title=side+' offset (Correction-For simplex) table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
    
#     plot(figTableFRONT,filename=f+" Offset Table FRONT.html") 
#     ####BACK
     
#     try:
#         side='Back'
#         new_list=[]
#         for col in ColorList:
#             new_list = [-number for number in PHoffsetPerHBACK[col]]
    
#             ListofListBACK.append(new_list)
            
#         figTableBACK = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
#                      cells=dict(values=[PHname]+ListofListBACK,font=dict(color='black', size=15)))
#                          ])
#         figTableBACK.update_layout(title=side+' offset table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
    
    
#         plot(figTableBACK,filename=f+" Offset Table BACK.html") 
#     except:
#         1
    
    
#     ### Tilt
#     side='Front'
#     headerTilt=[]
#     ListofListTiltFRONT=[]
#     ListofListTiltBACK=[]
#     # PHname=[]
    
#     # for i in range(24):
#     #     PHname.append(i)
    
    
#     for col in ColorList:
#         headerTilt.append(col+' Tilt')
#         # header.append(col+' Tilt')
#         ListofListTiltFRONT.append(PHtiltPerHFRONT[col])
    
#     backGroundCLR='rgb(200, 200, 200)'
#     colors = n_colors(backGroundCLR, 'rgb(200, 0, 0)', ColorLevelsTilt, colortype='rgb')
#     fillcolorList=[]
#     formatList=[]
#     formatList.append("")
#     for i in range(len(ListofListTiltFRONT)):
#         fillcolorList.append(np.array(colors)[(abs(np.asarray(ListofListTiltFRONT[i]))/DivideByNumTilt).astype(int)])
#         formatList.append("0.2f")
        
    
#     ####FRONT Tilt
#     figTableTiltFRONT = go.Figure(data=[go.Table(header=dict(values=['PH#']+headerTilt),
#                      cells=dict(values=[PHname]+ListofListTiltFRONT,fill_color=[backGroundCLR]+fillcolorList,font=dict(color='black', size=15),format=formatList))
#                          ])
    
#     figTableTiltFRONT.update_layout(title=side+' Tilt table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
    
#     plot(figTableTiltFRONT,filename=f+" Tilt Table FRONT.html") 
    
#     ####BACK Tilt
    
    
#     try:
#         side='Back'
#         # headerTilt=[]
#         for col in ColorList:
#             # headerTilt.append(col+' Tilt')
#             # header.append(col+' Tilt')
#             ListofListTiltBACK.append(PHtiltPerHBACK[col])
            
#         fillcolorList=[]
#         for i in range(len(ListofListTiltBACK)):
#             fillcolorList.append(np.array(colors)[(abs(np.asarray(ListofListTiltBACK[i]))/DivideByNumTilt).astype(int)]) 
            
#         figTableTiltBACK = go.Figure(data=[go.Table(header=dict(values=['PH#']+headerTilt),
#                      cells=dict(values=[PHname]+ListofListTiltBACK,fill_color=[backGroundCLR]+fillcolorList,font=dict(color='black', size=15),format=formatList))
#                          ])
        
#         figTableTiltBACK.update_layout(title=side+' Tilt table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
        
#         plot(figTableTiltBACK,filename=f+" Tilt Table BACK.html")  
#     except:
#         1    
    
    
    
     
#     #### FRONT -BACK delta
     
    
#     try:
#         ListofListDelta=[]    
#         header=[]
#         fillcolorList=[]  
#         backGroundCLR='rgb(200, 200, 200)'
#         colors = n_colors(backGroundCLR, 'rgb(200, 0, 0)', ColorLevels, colortype='rgb')
    
#         for col in ColorList:
#             header.append(col+'Delta(Front-Back) Offset')
#         for col in ColorList:
#             ListofListDelta.append(list(np.asarray(PHoffsetPerHFRONT[col])-np.asarray(PHoffsetPerHBACK[col])))
            
#         for i in range(len(ListofListDelta)):
#             # x2 = 30 * np.ones(len(ListofListDelta[i]))
#             fillcolorList.append(np.array(colors)[(abs(np.asarray(ListofListDelta[i]))/DivideByNum).astype(int)])
        
        
            
#         figTableDelta = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
#                      cells=dict(values=[PHname]+ListofListDelta,fill_color=[backGroundCLR]+fillcolorList,font=dict(color='black', size=15)))
#                          ])
#         figTableDelta.update_layout(title='Delta offset table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
    
    
#         plot(figTableDelta,filename=f+" Delta Offset Table.html") 
#     except:
#         1
    
    
    
#     #### FRONT -BACK Average
     
    
#     try:
#         ListofListAverage=[]    
#         header=[]
#         fillcolorList=[]  
        
    
#         for col in ColorList:
#             header.append(col+'Average(Front&Back) Offset')
#         for col in ColorList:
#             ListofListAverage.append(list(-(np.asarray(PHoffsetPerHFRONT[col])+np.asarray(PHoffsetPerHBACK[col]))/2))
            
      
        
        
            
#         figTableDelta = go.Figure(data=[go.Table(header=dict(values=['PH#']+header),
#                      cells=dict(values=[PHname]+ListofListAverage,font=dict(color='black', size=15)))
#                          ])
#         figTableDelta.update_layout(title='Correction table S.Golay = '+ str(MaxWaveWindow)+'---> '+f)
    
    
#         plot(figTableDelta,filename=f+" Correction Table.html") 
#     except:
#         1


# # cells=dict(
# #     values=[a, b, c],
# #     line_color=[np.array(colors)[a],np.array(colors)[b], np.array(colors)[c]],
#     fill_color=[np.array(colors)[a],np.array(colors)[b], np.array(colors)[c]],
#     align='center', font=dict(color='white', size=11)
#     ))



 