# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:24:26 2023

@author: Ireneg
"""
#################################################################
from zipfile import ZipFile
from tkinter import simpledialog
from tkinter import *
from tkinter import filedialog
from pathlib import Path
import glob
from datetime import datetime
import scipy.io
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import plotly.graph_objects as go
from plotly.colors import n_colors
from scipy.signal import savgol_filter
from collections import OrderedDict
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
global RecDimX, RecDimY
import subprocess 
import chart_studio.plotly as py 
import webbrowser
import shutil

%matplotlib


# url = "https://your-plotly-visualization-url" 
# subprocess.Popen(['start', 'chrome', url], shell=True)




RecDimX = 5
RecDimY = 5

#################################################################
#######################################################
global MaxWaveWindow, StpWindowSize, SvGolPol, limitDataCount, BarNum, CISsavgolWindow, CISsavgolWindow12k, PixelSize_um
global limitDataCount
YuriFormat = 0
fullAQMPixelCount=12480

MaxWaveWindow = 100
MaxWaveWindow12k = 1000

StpWindowSize = 2
StpWindowSize12k = 10

SvGolPol = 1
limitDataCount = 0.001
BarNum = 20
CISsavgolWindow = 11
CISsavgolWindow12k = 373
FileNameCSV = 'CIS_B2_filter_'+str(CISsavgolWindow)+'.csv'
FileNameCSV12k = 'CIS_B2_filter12k_'+str(CISsavgolWindow12k)+'.csv'

plot12k = 1
plot385 = 1


PixelSize_um = 84.6666
#######################################################


# Load the Pandas libraries with alias 'pd'


class CroppImageClass():
    def __init__(self):
        pass

    def CroppImage(self, img):

        cropped_img1 = img[:int(img.shape[0]/20), :int(img.shape[1]/20)]

        # Crop the second upper corner
        cropped_img2 = img[:int(
            img.shape[0]/20), int(img.shape[1])-int(img.shape[1]/20):int(img.shape[1])]

        # Get the user's selected coordinates from each cropped image
        point1 = cv2.selectROI(
            "LEFT- choose above the upper strip line and press SPACEBAR", cropped_img1)
        point2 = cv2.selectROI(
            "RIGHT- choose the center of the strip and press SPACEBAR", cropped_img2)

        # Convert the coordinates from the cropped images to the original image
        point01 = (0, point1[1])
        point02 = (
            int(img.shape[1])-(int(cropped_img2.shape[1])-point2[0]), point2[1])

        print("Selected point 1 in the original image:", point01)
        print("Selected point 2 in the original image:", point02)

        cv2.destroyAllWindows()

        x1, y1 = point01
        x2, y2 = point02

        cropped_img = img[y1:y2, :int(img.shape[1])]

        if not os.path.exists("Cropped images"):
            os.makedirs("Cropped images")

        cv2.imwrite("Cropped images/1.bmp", cropped_img)

        return cropped_img


class CIScurveFromImage():
    def __init__(self, ImageGL):
        self.ImageGL = ImageGL

    def AplyFilters(self, T_Lum, RecDimX, RecDimY):

        I2 = self.ImageGL
        # I2 = cv2.convertScaleAbs(I2)
        bw = cv2.threshold(I2, 255*T_Lum, 255, cv2.THRESH_BINARY)[1]
        BWdfill = cv2.morphologyEx(
            bw, cv2.MORPH_OPEN, np.ones((RecDimX, RecDimY)))
        diff_bw = np.diff(BWdfill, axis=0)
        max_val = []
        max_index = []

        for i in range(diff_bw.shape[1]):
            max_val.append(np.amax(diff_bw[:, i]))
            max_index.append(np.argmax(diff_bw[:, i]))

        return max_val, max_index


class ReduceNoise():
    def __init__(self, RawData):
        self.RawData = RawData

    def FixRawDatFromat_OneRow(self):
        RawDataFix = self.RawData.reset_index(drop=False)
        RawDataFix = RawDataFix.rename(columns={'index': 0, 0: 1})
        return RawDataFix

    def CalcAndRemoveTilT(self):

        z = np.polyfit(RawData[0], self.RawData[1], 1)
        tlt = (z[0]*(self.RawData[0])+z[1])
        RawData_Tilt = self.RawData[1]-tlt


        # # Calculate the 1st percentile of the data
        # percentile_limitDataCount = np.percentile(RawData_Tilt, limitDataCount)
       
        # # Filter out any values less than the 1st percentile
        # filtered_data = [x for x in RawData_Tilt if x >= percentile_limitDataCount]      
        
        # plt.figure('12kpoints')
        
        # plt.plot(self.RawData[0], self.RawData[1], 'o')
        # plt.plot(RawDataCopy[0], RawDataCopy[1], 'x')
        # plt.title('LimitDataCount='+str(limitDataCount))
        
        

        return RawData_Tilt, tlt, z


    def RemoveUnwantedData(self,pName):

       
         RawData_Tilt, tlt, z = self.CalcAndRemoveTilT()
 
         RawData_Tilt_list = list(RawData_Tilt)
        
         dataPracentage= (1-limitDataCount)*100
        
         percentile_x_1 = np.percentile(RawData_Tilt_list, dataPracentage)
         percentile_1 = np.percentile(RawData_Tilt_list, 100-dataPracentage)
        
         inx2delete = [i for i, x in enumerate(RawData_Tilt_list) if x <= percentile_1 or x >= percentile_x_1]
        
         RawDataCopy =  self.RawData.copy()
         RawDataCopy.drop(index=inx2delete, inplace=True)
         meanDrop=np.mean(RawDataCopy[1])

         RawDataCopy_2=  self.RawData.copy()


         RawDataCopy_2[1][inx2delete]=meanDrop

         RawDataCopy_2 = RawDataCopy_2.reset_index(drop=True)
            
         plt.figure(pName)
            
         plt.plot(ReduceNoise(RawData).RawData[0], ReduceNoise(RawData).RawData[1], 'o')
         plt.plot(RawDataCopy[0], RawDataCopy[1], 'x')
         plt.plot(RawDataCopy_2[0], RawDataCopy_2[1], '+')

         plt.title('LimitDataCount='+str(limitDataCount))
         
         
         return RawDataCopy_2

    def CutDataTo385Points(self):

        # Data385=pd.DataFrame();
        RawDataCopy = self.RemoveUnwantedData('p385')

        DistBtwPFULL = int((self.RawData[0][len(self.RawData[0])-1])/385)
        XvalueMeanFULL = []
        xinxFULL = []
        PxFull = self.RawData[0][0]
        for i in range(385):
            XvalueMeanFULL.append(PxFull)
            st = np.where(self.RawData[0] == PxFull)
            xinxFULL.append(st)
            PxFull = PxFull+DistBtwPFULL
            if PxFull > self.RawData[0][len(self.RawData[0])-1]:
                break
        stLoc = []
        enLoc = []
        YvalueMeanFULL = []

        for i in range(len(XvalueMeanFULL)-1):
            st = np.where(RawDataCopy[0] == XvalueMeanFULL[i])
            en = np.where(RawDataCopy[0] == XvalueMeanFULL[i+1])
            if not (len(st[0]) == 0) and not len(en[0]) == 0:
                stLoc.append(st[0][0])
                enLoc.append(en[0][0])
            if not len(enLoc) == 0:
                YvalueMeanFULL.append(
                    np.mean(RawDataCopy[1][stLoc[len(stLoc)-1]:enLoc[len(enLoc)-1]]))

        YvalueMeanFULL.append(RawDataCopy[1][len(RawDataCopy[1])-1])
        # YvalueMeanFULL=YvalueMeanFULL[0:3]+YvalueMeanFULL
        if len(XvalueMeanFULL) > len(YvalueMeanFULL):
            dlt = len(XvalueMeanFULL)-len(YvalueMeanFULL)
            YvalueMeanFULL = YvalueMeanFULL[0:dlt]+YvalueMeanFULL
        # plt.figure()
        # plt.plot(RawDataCopy[0], RawDataCopy[1], '-x')
        # plt.plot(XvalueMeanFULL, YvalueMeanFULL, '-o')

        return XvalueMeanFULL, YvalueMeanFULL, RawDataCopy

    def PrepareData4Saving(self):

        XvalueMeanFULL, YvalueMeanFULL, RawDataCopy = self.CutDataTo385Points()
        Data385 = pd.DataFrame()
        Data385[0] = XvalueMeanFULL

        Data385[1] = YvalueMeanFULL
        Data385[2] = -Data385[1]*PixelSize_um
        Data385[3] = (Data385[1]-Data385[1][0])

        # Data385[1]=Data385[1]-Data385[1][0]

        z = np.polyfit(Data385[0], Data385[3], 1)

        tlt = (z[0]*(Data385[0])+z[1])

        z1 = np.polyfit(Data385[0], Data385[2], 1)

        tlt1 = (z1[0]*(Data385[0])+z1[1])


        y = savgol_filter(Data385[2], CISsavgolWindow, SvGolPol)

     
        return Data385,  y, z1, tlt1, z, tlt

    def PrepareData4Saving12k(self):
        
        # # Calculate the 1st percentile of the data
        # percentile_limitDataCount = np.percentile(self.RawData[1], limitDataCount)
        
        # # Filter out any values less than the 1st percentile
        # filtered_data = [x for x in self.RawData[1] if x >= percentile_limitDataCount]

        y = savgol_filter(self.RawData[1], CISsavgolWindow12k, SvGolPol)

        return y


    def SaveCSV(self, fileName,y):
        
        CIScurve = pd.DataFrame()

        for i, yy in enumerate(y):
            CIScurve[i] = [yy]-y[0]
        
        
        CIScurve.to_csv(fileName, index=False, header=False)
        
        return CIScurve


class plotPlotly(CIScurveFromImage):
    def __init__(self, ImageGL, plotTitle, fileName, RecDimX, RecDimY, xdb, ydb, tlt, z):
        super().__init__(ImageGL)

        self.plotTitle = plotTitle
        self.fileName = fileName
        self.RecDimX = RecDimX
        self.RecDimY = RecDimY
        self.xdb = xdb
        self.ydb = ydb

        self.z = z
        self.tlt = tlt

    def PlotCIS385_12k(self, MaxWaveWindow, StpWindowSize):
        fig = go.Figure()

        # Add traces, one for each slider step
        # fig.add_trace(
        #     go.Scatter(x=list(self.xdb),y=list(self.ydb),line_color='red' ,
        #                 name='raw Data'))

        # fig.add_trace(
        #     go.Scatter(x=list(self.xdb),y=self.tlt,line_color='blue' ,
        #                 name='Tilt '+'Slope(x1000)='+"{0:.3f}".format(self.z[0]*1000)))
        fig.add_trace(
            go.Scatter(y=list(self.ydb), line_color='red',
                       name='raw Data'))

        fig.add_trace(
            go.Scatter(y=self.tlt, line_color='blue',
                       name='Tilt '+'Slope(x1000)='+"{0:.3f}".format(self.z[0]*1000)))
        # fig.add_trace(
        #     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
        #                 name=ColorForDisplay+'_After'), row=2, col=1)

        ##### Fiter Vs Befor ####
        for step in np.arange(3, MaxWaveWindow+3, StpWindowSize):
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color='green', width=2),
                    name="Window Size = " + str(step),
                    y=savgol_filter(self.ydb, step, SvGolPol)))

        # Make 10th trace visible
        fig.data[10].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": self.plotTitle + str(i)}],  # layout attribute
            )

            if i+1 < len(fig.data):
                # Toggle i'th trace to "visible"
                step["args"][0]["visible"][i+1] = True

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

        plot(fig, filename=self.fileName)

        return fig

    def PlotCIS(self):
        fig = go.Figure()

        # Add traces, one for each slider step
        NumberSteps = 101
        StepSize = 1 / NumberSteps
        ##### Fiter Vs Befor ####
        for T_Lum in np.arange(0, 1, StepSize):

            MaxValue, CISedge = self.AplyFilters(T_Lum, RecDimX, RecDimY)

            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color='red', width=2),
                    name="T_Lum = " + "{:.2f}".format(T_Lum), x=list(range(len(CISedge))),
                    y=CISedge))

        # Make 10th trace visible
        fig.data[10].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": self.plotTitle + "{:.2f}".format(i/NumberSteps)}],  # layout attribute
            )

            if i+1 < len(fig.data):
                # Toggle i'th trace to "visible"
                step["args"][0]["visible"][i+1] = True

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

        # fig.show()

        plot(fig, filename=self.fileName)


        return fig


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
root = Tk()
root.withdraw()

yTotalPics385p=pd.DataFrame();
yTotalPics12kp=pd.DataFrame();

MachineName =''

# pthF = filedialog.askdirectory()
while 1:

    pthF = filedialog.askopenfilename()
    
    if not "Cropped images" in pthF:
    
        f2delete = pthF.split('/')[len(pthF.split('/'))-1]
        f = "1.bmp"
        pth4save = pthF.replace(f2delete, "")
        os.chdir(pth4save)
        img = cv2.imread(pthF)
        I1 = CroppImageClass().CroppImage(img)
    
    
    else:
        f = pthF.split('/')[len(pthF.split('/'))-1]
        I1 = cv2.imread(pthF)
        pth4save = pthF.replace("Cropped images/"+f, "")
    
    
    ImageGL = 0.2989 * I1[:, :, 0] + 0.5870 * I1[:, :, 1] + 0.1140 * I1[:, :, 2]
    
    
    # PLOT
    
    plotTitle = pthF+" CIS edge T_Lum= "
    fileName = pthF.replace('/', '_').replace(f,
                                              "").replace(":", "") + "CIS" + ".html"
    xdb = 0
    ydb = 0
    tlt = 0
    z = 0
    figCIScalc = plotPlotly(ImageGL, plotTitle, fileName,
                            RecDimX, RecDimY, xdb, ydb, tlt, z).PlotCIS()
    os.chdir(pth4save)
    plotly_html_content = figCIScalc.to_html(full_html=False)
    # Save the HTML content to a file with a custom name
    custom_html_name = fileName
    with open(custom_html_name, 'w') as f:
        f.write(plotly_html_content)


    # url = py.plot(figCIScalc, filename=fileName)
    url = 'file:///' + pth4save + fileName
    subprocess.Popen(['start', 'chrome', url], shell=True)
    
    
    # # url = 'file:///C:/Users/gevay/Downloads/C_Users_gevay_Downloads_Cropped%20images_CIS.html'

    # webbrowser.open(url)
    
    # chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
    # webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
    # webbrowser.get('chrome').open(url)
    # root = Tk()
    # root.withdraw()
    T_Lum = simpledialog.askstring(
        "Input", "Enter T_Lum value (range 0 - 1):", parent=root)
    RawData = pd.DataFrame()
    
    max_val, max_index = CIScurveFromImage(ImageGL).AplyFilters(
        float(T_Lum), RecDimX, RecDimY)
    
    RawData['Value'] = list(max_index)
    
    RawData.to_csv(pth4save+'RawData.csv', header=None)
    
    
    plt.figure()
    plt.plot(max_index)
    plt.title(" T_Lum value: "+T_Lum)
    
    
    # Step 2
    
    print('**************************************************************************')
    if not len(MachineName):
        print('Please Enter  machine Name in the Dialog box')
        MachineName = simpledialog.askstring(
            "Input", "Enter The machine Name:", parent=root)
        print('Done')
    print('Machine Name: '+MachineName)
    print('**************************************************************************')
    
    print('**************************************************************************')
    print('Please Enter Data Limit in the Dialog box')
    limitDataCount = float(simpledialog.askstring(
        "Input", "Enter Data Pracentege to ignore(from 1- 0):", parent=root))
    print('Done')
    print('**************************************************************************')
    
    pthF1 = pth4save[:-1]
    f1 = pthF1.split('/')[len(pthF1.split('/'))-1]
    DirectorypathF = pthF1.replace(f1, '')
    os.chdir(pthF1)
    
    # RawData=pd.read_csv(pthF1+'/RawData.csv',header = None);
    RawData = RawData.reset_index()
    RawData = RawData.rename(columns={'index': 0, 'Value': 1})
    
    # FIX FORMAT from ariel raw data to yuri rawdata
    if YuriFormat:
        RawData = ReduceNoise(RawData).FixRawDatFromat_OneRow()
    
    
    RawData_Tilt, tlt12k, z12k = ReduceNoise(
        RawData).CalcAndRemoveTilT()
    
    # RawData_Tilt_df=pd.DataFrame({0:RawData[0],1:RawData_Tilt})
    RawData_12k = ReduceNoise(
        RawData).RemoveUnwantedData('p12k')
    
    
    Data385,  y, z1, tlt1, z, tlt = ReduceNoise(
        RawData).PrepareData4Saving()
    
     
    current_date = datetime.now().date().strftime("%Y_%m_%d")

    # To Implament
    
    if plot385:
        xdb = Data385[0]
        ydb = Data385[2]
        plotTitle = pthF1+'-->'+f1+' Tilt in um=' + "{0:.3f}".format(tlt1[0]-tlt1[len(
            tlt1)-1])+" _385 points - For CIS (for implamentation) Slider switched to Step: "  # Can modify Plot title
        fileName = f1 + " CIS curve raw data and filter 385 implament" + ".html"
    
        figCIScalc = plotPlotly(ImageGL, plotTitle, fileName, RecDimX, RecDimY,
                                xdb, ydb, tlt1, z1).PlotCIS385_12k(MaxWaveWindow, StpWindowSize)
        print('**************************************************************************')
        print('Please Enter  WindowSize in the Dialog box')
        CISsavgolWindow = int(simpledialog.askstring(
            "Input", "Enter WindowSize value:", parent=root))
        print('Done')
        print('**************************************************************************')
        FileNameCSV = 'CURVE_' +MachineName+ '_385_'+current_date+'.csv'
        # Data385,  y, z1, tlt1, z, tlt = ReduceNoise(
        #     RawData).PrepareData4Saving()
        
        yTotalPics385p=pd.concat([yTotalPics385p,pd.Series(y)],axis=1)
     
    
            
    
        # plt.figure()
        # plt.plot(CIScurve.loc[0, :])
        # plt.title('385 points'+' windowSize='+str(CISsavgolWindow))
    
    
    # 12k point
    if plot12k:
        xdb = RawData_12k[0]
        ydb = RawData_12k[1]
        plotTitle = pthF1+'-->'+f1+' Tilt in um=' + "{0:.3f}".format(tlt1[0]-tlt1[len(
            tlt1)-1])+" _12k points - For CIS (for implamentation) Slider switched to Step: "  # Can modify Plot title
        fileName = f1 + " CIS curve raw data and filter 12k implament" + ".html"
    
        figCIScalc = plotPlotly(ImageGL, plotTitle, fileName, RecDimX, RecDimY, xdb,
                                ydb, tlt12k, z12k).PlotCIS385_12k(MaxWaveWindow12k, StpWindowSize12k)
        print('**************************************************************************')
        print('Please Enter  WindowSize12k in the Dialog box')
        CISsavgolWindow12k = int(simpledialog.askstring(
            "Input", "Enter WindowSize12k value:", parent=root))
        print('Done')
        print('**************************************************************************')
    
        FileNameCSV12k = 'CURVE_' +MachineName+ '_12k_'+current_date+'.csv'
        y12k = ReduceNoise(RawData_12k).PrepareData4Saving12k()
    
    
        # y = savgol_filter(ReduceNoise(RawData_12k).RawData[1], CISsavgolWindow12k, SvGolPol)    
    
        yTotalPics12kp=pd.concat([yTotalPics12kp,pd.Series(y12k)],axis=1)
    
        # plt.figure()
        # plt.plot(ReduceNoise(RawData_12k).RawData[1])
        # plt.plot(y)

        # plt.title('12k points'+' windowSize='+str(CISsavgolWindow12k))
    
    
    
    print('**************************************************************************')
    print('Please Enter  if you want to continue uplaoding pictures')
    morePic = int(simpledialog.askstring(
        "Input", "Do you want to up load another picture?(yes-1, no-0)", parent=root))
    if plot12k:
        if not morePic:
            ymean12kp = yTotalPics12kp.mean(axis=1)
            CIScurve12kp= ReduceNoise(RawData).SaveCSV(FileNameCSV12k, ymean12kp)
            
            plt.figure('12k points')
            plt.plot(CIScurve12kp.loc[0, :])
            plt.title('12k points'+' windowSize='+str(CISsavgolWindow12k))
            
    if plot385:
        if not morePic:
            ymean385p = yTotalPics385p.mean(axis=1)
            CIScurve385p= ReduceNoise(RawData).SaveCSV(FileNameCSV, ymean385p)
            
            plt.figure('385 points')
            plt.plot(CIScurve385p.loc[0, :])
            plt.title('385 points'+' windowSize='+str(CISsavgolWindow))
            
    if not morePic:
        break;
    
    

#########################################################################################





