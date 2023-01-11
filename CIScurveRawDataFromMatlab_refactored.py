# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:51:03 2022

@author: Ireneg

"""
#######################################################
global  MaxWaveWindow,limitDataCount,BarNum,CISsavgolWindow,PixelSize_um

YuriFormat=0;

MaxWaveWindow=100;
limitDataCount=0.05;
BarNum=20
CISsavgolWindow=9
FileNameCSV='CIS_B2.csv';
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








class ReduceNoise():
    def __init__(self,RawData):
      self.RawData = RawData;
    
    
    def FixRawDatFromat_OneRow(self):
      RawDataFix=self.RawData.reset_index(drop=False)
      RawDataFix=RawDataFix.rename(columns={'index':0,0:1})
      return RawDataFix;
  
    def CalcAndRemoveTilT(self):
  
        z=np.polyfit(RawData[0], self.RawData[1], 1)
        tlt=(z[0]*(self.RawData[0])+z[1])              
        RawData_Tilt=self.RawData[1]-tlt
        
        return RawData_Tilt,tlt,z;
    
    def CalcHistOfData(self):
        RawData_Tilt,tlt,z = self.CalcAndRemoveTilT();
        l=len(RawData_Tilt)

        # plt.Figure()
        m=plt.hist(abs(RawData_Tilt),BarNum)
        # plt.show()
        
        DataCount=m[0]
        DataRange=m[1]
        
        NumberOfvalidData=int((1-limitDataCount)*l);

        
        return DataCount,DataRange,NumberOfvalidData,RawData_Tilt,tlt,z
    
    def RemoveUnwantedData(self):        
        
        DataCount,DataRange,NumberOfvalidData,RawData_Tilt,tlt,z = self.CalcHistOfData();
        DataSum=0
        for i,dt in enumerate(DataCount):
            DataSum=DataSum+dt;
            if DataSum>NumberOfvalidData:
                break;
        inx2delete=[]
        fixedRawData=[]
        for j,dt in enumerate(RawData_Tilt):
            if abs(dt)<  DataRange[i]:
                fixedRawData.append(self.RawData[1][j])
            else:
               inx2delete.append(j)
               fixedRawData.append(tlt[j]) 
                
        # RawData[1]=fixedRawData
        
        RawDataCopy=self.RawData.copy()
        
        RawDataCopy.drop(index=inx2delete,inplace=True)
        RawDataCopy=RawDataCopy.reset_index(drop=True)
        plt.figure()
        plt.plot(RawDataCopy[0],RawDataCopy[1],'o')
        plt.plot(self.RawData[0],self.RawData[1],'x')
        
        return RawDataCopy;
    
    def CutDataTo385Points(self):
        
        # Data385=pd.DataFrame();
        RawDataCopy=self.RemoveUnwantedData();
        
        DistBtwPFULL=int((self.RawData[0][len(self.RawData[0])-1])/385)
        XvalueMeanFULL=[]
        xinxFULL=[]
        PxFull=self.RawData[0][0]
        for i in range(385):
            XvalueMeanFULL.append(PxFull);
            st= np.where(self.RawData[0] == PxFull)
            xinxFULL.append(st)
            PxFull=PxFull+DistBtwPFULL;
            if PxFull>self.RawData[0][len(self.RawData[0])-1]:
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
                
        # YvalueMeanFULL=YvalueMeanFULL[0:3]+YvalueMeanFULL
        plt.figure()
        plt.plot(RawDataCopy[0],RawDataCopy[1],'-x')
        plt.plot(XvalueMeanFULL[1:],YvalueMeanFULL,'-o')
        
        return XvalueMeanFULL,YvalueMeanFULL,RawDataCopy;
    
    def PrepareData4Saving(self,fileName):
        
        XvalueMeanFULL,YvalueMeanFULL,RawDataCopy=self.CutDataTo385Points();
        Data385=pd.DataFrame();
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
        
    
        
        CIScurve[0]=[0]
        for i,yy in enumerate(y):
            CIScurve[i+1]=[yy]
        
        CIScurve.to_csv(fileName,index=False,header=False);
        
       
        return Data385,CIScurve,y,z1,tlt1,z,tlt
        

 
class plotPlotly():
   def __init__(self,xdb,ydb,plotTile,fileName,tlt,z):
      self.xdb = xdb;
      self.ydb = ydb;

      self.z=z
      self.tlt=tlt
      self.plotTile=plotTile;
      self.fileName=fileName;
      


   def PlotCIS(self):
        fig = go.Figure()


        # Add traces, one for each slider step
        fig.add_trace(
            go.Scatter(x=list(self.xdb),y=list(self.ydb),line_color='red' , 
                        name='raw Data'))
        
        fig.add_trace(
            go.Scatter(x=list(self.xdb),y=self.tlt,line_color='blue' , 
                        name='Tilt '+'Slope(x1000)='+"{0:.3f}".format(self.z[0]*1000)))
        # fig.add_trace(
        #     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
        #                 name=ColorForDisplay+'_After'), row=2, col=1)
        
        ##### Fiter Vs Befor ####
        for step in  np.arange(3, MaxWaveWindow+3, 2):
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color='green', width=2),
                    name="Window Size = " + str(step),x=list(self.xdb),
                    y=savgol_filter(self.ydb, step, 1)))
        
        
        
        # Make 10th trace visible
        fig.data[10].visible = True
        
        
        
        
        
        
        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title":self.plotTile + str(i)}],  # layout attribute
            )
        
                
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
        
       
        
        plot(fig,filename=self.fileName) 
        
        return fig;

        
        
 ###############################################################################################################       
 ###############################################################################################################       
 ###############################################################################################################       
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthF = filedialog.askdirectory()
        

f=pthF.split('/')[len(pthF.split('/'))-1]
DirectorypathF=pthF.replace(f,'');
os.chdir(pthF)

RawData=pd.read_csv(pthF+'/RawData.csv',header = None);


#### FIX FORMAT from ariel raw data to yuri rawdata
if YuriFormat:
    RawData=ReduceNoise(RawData).FixRawDatFromat_OneRow();
    

DataCount,DataRange,NumberOfvalidData,RawData_Tilt,tlt,z=  ReduceNoise(RawData).CalcHistOfData();
Data385,CIScurve,y,z1,tlt1,z,tlt=ReduceNoise(RawData).PrepareData4Saving(FileNameCSV)

###########################Plot
xdb=Data385[0]
ydb=Data385[3]
plotTitle=pthF+'-->'+f+" 385 points - For CIS (for comparison) Slider switched to Step: " # Can modify Plot title
fileName=f +" CIS curve raw data and filter 385 compre"+ ".html";

figCompare=plotPlotly(xdb,ydb,plotTitle,fileName,tlt,z).PlotCIS();

###### To Implament 
xdb=Data385[0]
ydb=Data385[2]
plotTitle=pthF+'-->'+f+' Tilt in um=' +"{0:.3f}".format(tlt1[0]-tlt1[len(tlt1)-1])+" 385 points - For CIS (for implamentation) Slider switched to Step: " # Can modify Plot title
fileName=f +" CIS curve raw data and filter 385 implament"+ ".html";

figCIScalc=plotPlotly(xdb,ydb,plotTitle,fileName,tlt1,z1).PlotCIS();

