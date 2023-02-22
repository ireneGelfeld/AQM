# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:24:26 2023

@author: Ireneg
"""
#################################################################
global RecDimX,RecDimY

RecDimX= 5
RecDimY= 5

#################################################################
#######################################################
global  MaxWaveWindow,StpWindowSize,SvGolPol,limitDataCount,BarNum,CISsavgolWindow,CISsavgolWindow12k,PixelSize_um
global limitDataCount
YuriFormat=0;

MaxWaveWindow=100;
MaxWaveWindow12k=1000;

StpWindowSize=2
StpWindowSize12k=10

SvGolPol=1
limitDataCount=0.001;
BarNum=20
CISsavgolWindow=11
CISsavgolWindow12k=373
FileNameCSV='CIS_B2_filter_'+str(CISsavgolWindow)+'.csv';
FileNameCSV12k='CIS_B2_filter12k_'+str(CISsavgolWindow12k)+'.csv';


PixelSize_um=84.6666
#######################################################


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from collections import OrderedDict
from scipy.signal import savgol_filter
from plotly.colors import n_colors
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.subplots import make_subplots

import scipy.io
from datetime import datetime
import glob
from zipfile import ZipFile 
from pathlib import Path

from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog




class CroppImageClass():
    def __init__(self):
        pass        
    def  CroppImage(self,img):
        
        cropped_img1 =img[:int(img.shape[0]/20), :int(img.shape[1]/20)]

        # Crop the second upper corner
        cropped_img2 = img[:int(img.shape[0]/20), int(img.shape[1])-int(img.shape[1]/20):int(img.shape[1])]
    
        
        # Get the user's selected coordinates from each cropped image
        point1 = cv2.selectROI("LEFT- choose above the upper strip line and press SPACEBAR", cropped_img1)
        point2 = cv2.selectROI("RIGHT- choose the center of the strip and press SPACEBAR", cropped_img2)
        
        
        
        # Convert the coordinates from the cropped images to the original image
        point01 = (0, point1[1])
        point02 = (int(img.shape[1])-(int(cropped_img2.shape[1])-point2[0]), point2[1])
        
        print("Selected point 1 in the original image:", point01)
        print("Selected point 2 in the original image:", point02)
        
        cv2.destroyAllWindows()
        
        x1,y1=point01
        x2,y2=point02
        
        cropped_img = img[y1:y2, :int(img.shape[1])]

        if not os.path.exists("Cropped images"):
            os.makedirs("Cropped images")
        
        cv2.imwrite("Cropped images/1.bmp", cropped_img)   
        
        return cropped_img


class CIScurveFromImage():
    def __init__(self,ImageGL):
      self.ImageGL = ImageGL;
       
    def  AplyFilters(self,T_Lum, RecDimX, RecDimY):
        
        I2 = self.ImageGL
        # I2 = cv2.convertScaleAbs(I2)
        bw = cv2.threshold(I2, 255*T_Lum, 255, cv2.THRESH_BINARY)[1]
        BWdfill = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((RecDimX, RecDimY)))
        diff_bw = np.diff(BWdfill, axis=0)
        max_val=[]
        max_index=[]
        
        for i in range(diff_bw.shape[1]):
            max_val.append(np.amax(diff_bw[:,i]))
            max_index.append(np.argmax(diff_bw[:,i]))
            
        return max_val,max_index

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
        plt.title('LimitDataCount='+str(limitDataCount))
        

        
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
        if len(XvalueMeanFULL[1:])>len(YvalueMeanFULL):
            dlt=len(XvalueMeanFULL[1:])-len(YvalueMeanFULL);
            YvalueMeanFULL=YvalueMeanFULL[0:dlt]+YvalueMeanFULL
        plt.figure()
        plt.plot(RawDataCopy[0],RawDataCopy[1],'-x')
        plt.plot(XvalueMeanFULL[1:],YvalueMeanFULL,'-o')
        
        return XvalueMeanFULL,YvalueMeanFULL,RawDataCopy;
    
    def PrepareData4Saving(self,fileName,saveCSV):
        
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
        
        y=savgol_filter(Data385[2], CISsavgolWindow, SvGolPol)       
        
    
        
        CIScurve[0]=[0]
        for i,yy in enumerate(y):
            CIScurve[i+1]=[yy]
            
        if saveCSV:
            CIScurve.to_csv(fileName,index=False,header=False);
        
       
        return Data385,CIScurve,y,z1,tlt1,z,tlt
    
    def PrepareData4Saving12k(self,fileName):
            
            
            
            CIScurve=pd.DataFrame()
            
            y=savgol_filter(self.RawData[1], CISsavgolWindow12k, SvGolPol)       
            
        
            
            for i,yy in enumerate(y):
                CIScurve[i]=[yy]-y[0]
            
            CIScurve.to_csv(fileName,index=False,header=False);
            
           
            return CIScurve

class plotPlotly(CIScurveFromImage):
   def __init__(self,ImageGL,plotTitle,fileName,RecDimX, RecDimY,xdb,ydb,tlt,z):
      super().__init__(ImageGL)

      self.plotTitle=plotTitle;
      self.fileName=fileName;
      self.RecDimX=RecDimX
      self.RecDimY=RecDimY
      self.xdb = xdb;
      self.ydb = ydb;

      self.z=z
      self.tlt=tlt

   def PlotCIS385_12k(self,MaxWaveWindow,StpWindowSize):
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
        for step in  np.arange(3, MaxWaveWindow+3, StpWindowSize):
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color='green', width=2),
                    name="Window Size = " + str(step),x=list(self.xdb),
                    y=savgol_filter(self.ydb, step, SvGolPol)))
        
        
        
        # Make 10th trace visible
        fig.data[10].visible = True
        
        
        
        
        
        
        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title":self.plotTitle + str(i)}],  # layout attribute
            )
        
                
            if i+1 < len(fig.data):
                step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
        
            step["args"][0]["visible"][0] = True 
            step["args"][0]["visible"][1] = True
        
            steps.append(step)
        
        
        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Window Size: "},
            pad={"t": int(MaxWaveWindow/StpWindowSize)},
            steps=steps
        )]
        
        fig.update_layout(
            sliders=sliders
        )
        
        
        fig.show()
        
       
        
        plot(fig,filename=self.fileName) 
        
        return fig;



   def PlotCIS(self):
        fig = go.Figure()


        # Add traces, one for each slider step
        NumberSteps= 101
        StepSize= 1 / NumberSteps;
        ##### Fiter Vs Befor ####
        for T_Lum in  np.arange(0, 1, StepSize):
            
            MaxValue,CISedge=self.AplyFilters(T_Lum,RecDimX, RecDimY);
            
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color='red', width=2),
                    name="T_Lum = " + "{:.2f}".format(T_Lum),x=list(range(len(CISedge))),
                    y=CISedge))
        
        
        
        # Make 10th trace visible
        fig.data[10].visible = True
        
        
        
        
        
        
        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title":self.plotTitle + "{:.2f}".format(i/NumberSteps)}],  # layout attribute
            )
        
                
            if i+1 < len(fig.data):
                step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
        
            # step["args"][0]["visible"][0] = True 
            # step["args"][0]["visible"][1] = True
        
            steps.append(step)
        
        
        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Window Size: "},
            pad={"t": int(NumberSteps)},
            steps=steps
        )]
        
        fig.update_layout(
            sliders=sliders
        )
        
        
        fig.show()
        
       
        
        plot(fig,filename=self.fileName) 
        
        return fig;




################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

root = Tk()
root.withdraw()
# pthF = filedialog.askdirectory()


pthF = filedialog.askopenfilename()

if not "Cropped images" in pthF:
    
    f2delete=pthF.split('/')[len(pthF.split('/'))-1]
    f = "1.bmp"
    pth4save=pthF.replace(f2delete,"")
    os.chdir(pth4save)
    img = cv2.imread(pthF);
    I1 =  CroppImageClass().CroppImage(img);

    
else:
    f=pthF.split('/')[len(pthF.split('/'))-1]
    I1 = cv2.imread(pthF)
    pth4save = pthF.replace("Cropped images/"+f,"");
   




ImageGL = 0.2989 * I1[:, :, 0] + 0.5870 * I1[:, :, 1] + 0.1140 * I1[:, :, 2]


##########PLOT

plotTitle=pthF+" CIS edge T_Lum= "
fileName=pthF.replace('/','_').replace(f,"").replace(":","")+ "CIS edge"+ ".html";
xdb=0
ydb=0
tlt=0
z=0
figCIScalc=plotPlotly(ImageGL,plotTitle,fileName,RecDimX, RecDimY,xdb,ydb,tlt,z).PlotCIS();

# root = Tk()
# root.withdraw()
T_Lum = simpledialog.askstring("Input", "Enter T_Lum value:", parent=root)
RawData=pd.DataFrame();

max_val,max_index = CIScurveFromImage(ImageGL).AplyFilters(float(T_Lum)+0.01, RecDimX, RecDimY)

RawData['Value']=list(max_index)

RawData.to_csv(pth4save+'RawData.csv',header=None)



plt.figure()
plt.plot(max_index)
plt.title(" T_Lum value: "+T_Lum)
         
       

###############Step 2
print('**************************************************************************')
print('Please Enter  machine Name in the Dialog box')        
MachineName = simpledialog.askstring("Input", "Enter The machine Name:", parent=root)
print('Done');
print('**************************************************************************')

print('**************************************************************************')
print('Please Enter Data Limit in the Dialog box')
limitDataCount = float(simpledialog.askstring("Input", "Enter Data Pracentege to ignore:", parent=root))
print('Done');
print('**************************************************************************')

pthF1=pth4save[:-1];
f1=pthF1.split('/')[len(pthF1.split('/'))-1]
DirectorypathF=pthF1.replace(f1,'');
os.chdir(pthF1)

# RawData=pd.read_csv(pthF1+'/RawData.csv',header = None);
RawData=RawData.reset_index()
RawData=RawData.rename(columns={'index':0,'Value':1})

#### FIX FORMAT from ariel raw data to yuri rawdata
if YuriFormat:
    RawData=ReduceNoise(RawData).FixRawDatFromat_OneRow();
    

DataCount,DataRange,NumberOfvalidData,RawData_Tilt,tlt12k,z12k=  ReduceNoise(RawData).CalcHistOfData();

Data385,CIScurve,y,z1,tlt1,z,tlt=ReduceNoise(RawData).PrepareData4Saving(FileNameCSV,0)




###### To Implament 
xdb=Data385[0]
ydb=Data385[2]
plotTitle=pthF1+'-->'+f1+' Tilt in um=' +"{0:.3f}".format(tlt1[0]-tlt1[len(tlt1)-1])+" _385 points - For CIS (for implamentation) Slider switched to Step: " # Can modify Plot title
fileName=f1 +" CIS curve raw data and filter 385 implament"+ ".html";

figCIScalc=plotPlotly(ImageGL,plotTitle,fileName,RecDimX, RecDimY,xdb,ydb,tlt1,z1).PlotCIS385_12k(MaxWaveWindow,StpWindowSize);
print('**************************************************************************')
print('Please Enter  WindowSize in the Dialog box')   
CISsavgolWindow = int(simpledialog.askstring("Input", "Enter WindowSize value:", parent=root))
print('Done');
print('**************************************************************************')
FileNameCSV='CIS_'+MachineName+'_filter_'+str(CISsavgolWindow)+'.csv';
Data385,CIScurve,y,z1,tlt1,z,tlt=ReduceNoise(RawData).PrepareData4Saving(FileNameCSV,1)


########### 12k point
xdb=RawData[0]
ydb=RawData[1]
plotTitle=pthF1+'-->'+f1+' Tilt in um=' +"{0:.3f}".format(tlt1[0]-tlt1[len(tlt1)-1])+" _12k points - For CIS (for implamentation) Slider switched to Step: " # Can modify Plot title
fileName=f1 +" CIS curve raw data and filter 12k implament"+ ".html";

figCIScalc=plotPlotly(ImageGL,plotTitle,fileName,RecDimX, RecDimY,xdb,ydb,tlt12k,z12k).PlotCIS385_12k(MaxWaveWindow12k,StpWindowSize12k);
print('**************************************************************************')
print('Please Enter  WindowSize12k in the Dialog box')   
CISsavgolWindow12k = int(simpledialog.askstring("Input", "Enter WindowSize12k value:", parent=root))
print('Done');
print('**************************************************************************')

FileNameCSV12k='CIS_'+MachineName+'_filter12k_'+str(CISsavgolWindow12k)+'.csv';
CIScurve12k=ReduceNoise(RawData).PrepareData4Saving12k(FileNameCSV12k)


plt.figure();
plt.plot(CIScurve.loc[0,:])
plt.title('385 points'+' windowSize='+str(CISsavgolWindow))



plt.figure();
plt.plot(CIScurve12k.loc[0,:])
plt.title('12k points'+' windowSize='+str(CISsavgolWindow12k))        

            
            
            
