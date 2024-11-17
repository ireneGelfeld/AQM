# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import medfilt2d, medfilt
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.signal import correlate2d
from scipy.ndimage import morphology
from scipy.ndimage import convolve
import os
from sklearn.cluster import KMeans
import pandas as pd 

import pandas as pd 
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.subplots import make_subplots
import re
from scipy.signal import savgol_filter
import pickle
import seaborn as sns
import matplotlib.colors as mcolors
from PIL import Image
import time
import threading
from queue import Queue
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QComboBox, QPushButton, QFileDialog,QLineEdit,QTextEdit
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtCore import pyqtSignal, QObject
# import plotly.io as pio
# pio.renderers
# pio.renderers.default='browser'

# %matplotlib
############################################################################
global ColorDic,pixSize,MaxWaveWindow,sideDic,CircleArea,DistBtwCrcle,UseTarget,DistanceBetweenColumns,NumOfSec,CircleLvl,allowed_time,PointsToIgnore

allowed_time= 10

SatndardDistanceBetweenColumns= 406.5 #404.9626381019503;
LargeDistanceBetweenColumns= 1431.057469809592;
UseTarget =1 # False- uses the average distance between circles in pixels, True- uses DistBtwCrcle*AQMscale = (18 x 0.9832)

DistBtwCrcle=18
ScaleY=1#0.9965001812608202
AQMscale=0.9832
CircleArea=7
MaxWaveWindow=11
S_g_Degree=1
ColorDic={0:'Magenta',1:'Yellow',2:'Blue',3:'Orange',4:'Cyan',5:'Green',6:'Black'}
pixSize = 84.6666 # [um]
sideDic={0:'Left Side',1:'Middle',2:'Right Side'}

ColorDicNum={'Magenta':0,'Yellow':1,'Blue':2,'Orange':3,'Cyan':4,'Green':5,'Black':6}

PointOfSpeedCange= 223;
numberOfPoints=288 #439
# numberOfPoints=439#288 #439
# NumOfSec=18
NumOfSec=15

# DistanceBetweenColumns={i:SatndardDistanceBetweenColumns*1 for i in range(2,17)}
# DistanceBetweenColumns[0]=0

# DistanceBetweenColumns[1]=LargeDistanceBetweenColumns*1
# DistanceBetweenColumns[17]=LargeDistanceBetweenColumns*1

DistanceBetweenColumns={i:SatndardDistanceBetweenColumns*1 for i in range(1,NumOfSec)}
DistanceBetweenColumns[0]=0

# DistanceBetweenColumns[1]=LargeDistanceBetweenColumns*1
# DistanceBetweenColumns[17]=LargeDistanceBetweenColumns*1


CircleLvl=70
x_Cut_coord=[710,8000]


sectionNumber2Show=[1,5,6,7,10,14]


# colorInUseName=['Magenta','Yellow','Blue','Orange','Cyan','Green','Black']
colorInUseName=['Magenta','Yellow','Cyan','Black']

# clrToUseInJOB=['Magenta','Yellow','Blue','Orange','Cyan','Green','Black']
clrToUseInJOB=['Magenta','Yellow','Cyan','Black']


# clrToUseInJOB=['Black','Cyan']
scaling_factor=70;
offset_factor=10;

############ For SHARTER MEDIA#####################
PointsToIgnore=80 #439-359 
continuesPoints=0
pointsForPanel=numberOfPoints- PointsToIgnore
# 
# x_Cut_coord=[1000,8380]
############################################################################
class LoggerSignal(QObject):
    log_signal = pyqtSignal(str)

    def log(self, message):
        self.log_signal.emit(message)

class DataStructureUI(QWidget):
    closed = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Structure UI")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        # Combo box for selecting data structure
        self.number_of_colors_label = QLabel("Color combination:")
        layout.addWidget(self.number_of_colors_label)
        self.number_of_colors_combo = QComboBox()
        self.four_colors='Magenta,Yellow,Cyan,Black'
        self.seven_colors='Magenta,Yellow,Blue,Orange,Cyan,Green,Black'
        self.number_of_colors_combo.addItems([self.seven_colors,self.four_colors ])
        layout.addWidget(self.number_of_colors_combo)



        self.DisplayStrips_label = QLabel('Display Strips:', self)
        layout.addWidget(self.DisplayStrips_label)
        
        sectionNumber2Show_str='1,5,6,7,10,14,17'
        self.DisplayStrips_edit = QLineEdit(sectionNumber2Show_str, self)
        layout.addWidget(self.DisplayStrips_edit)
        

        # Button to load CSV file
        self.load_button = QPushButton("Load CSV File")
        self.load_button.clicked.connect(self.load_folder)
        layout.addWidget(self.load_button)

        # Button to update
        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_data)
        layout.addWidget(self.update_button)
        
        self.log_text = QTextEdit()
        layout.addWidget(self.log_text)

        self.clear_button = QPushButton("Clear Log")
        self.clear_button.clicked.connect(self.clear_log)
        layout.addWidget(self.clear_button)

        self.setLayout(layout)

        self.logger_signal = LoggerSignal()
        self.logger_signal.log_signal.connect(self.log_message)
        
    def log_message(self, message):
        self.log_text.append(message)

    def clear_log(self):
        self.log_text.clear()

    def load_folder(self):
        
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        
        if folder_dialog.exec_():
            self.folder_path = folder_dialog.selectedFiles()[0]
            # Implement the logic to load the CSV file
            # print(f"CSV file loaded: {file_path}")
        self.logger_signal.log("load path:"+self.folder_path)
        print("load path:"+self.folder_path)
        os.chdir(self.folder_path)
        
        self.update_data()

    def update_data(self):
        
        operation_hndlr=Operations()
        operation_hndlr.ReadAllFlats(self.folder_path)
        
        number_of_colors= self.number_of_colors_combo.currentText()

        if number_of_colors== self.four_colors:
            colorInUseName=['Magenta','Yellow','Cyan','Black']
   
        if number_of_colors== self.seven_colors:
            
            colorInUseName=['Magenta','Yellow','Blue','Orange','Cyan','Green','Black']
            
        
        sectionNumber2Show_str=self.DisplayStrips_edit.text()  
        sectionNumber2Show = [int(num) for num in sectionNumber2Show_str.split(',')]
        self.logger_signal.log("Colors: "+number_of_colors)
        print("Colors: "+number_of_colors)
        self.logger_signal.log("Starts Image Process... ")
        print("Starts Image Process... ")
        operation_hndlr.CalculateCenterOfMass_and_C2C(self.folder_path,colorInUseName)
        operation_hndlr.caculate_continues(colorInUseName)
            
        # operation_hndlr.Save_pickle(self.folder_path)
    
        #Plot C2C
        db= operation_hndlr.C2Cmat_allPanels_continues;
        PlotTitle='C2C'
        fileName=PlotTitle+'.html'
        figC2C_multiPanel= PlotSingle_Basic_multiPanel(db,PlotTitle,fileName,operation_hndlr.indexPanelNameDic,100)
        
        ##########################################################################################
        db= operation_hndlr.Color_Black_Sgoly_allPanels_continues;
        RefC01='Cyan'
        RefCl=RefC01
        if RefC01 in colorInUseName:
            PlotTitle='2 color diff - '+RefCl +' Vs color'
            
            fileName=PlotTitle+'.html'
            figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,operation_hndlr.indexPanelNameDic,100,sectionNumber2Show)

        ##########################################################################################
        db= operation_hndlr.Green_Black_Sgoly_allPanels_continues;
        RefC02='Magenta'

        RefCl=RefC02
        if RefC02 in colorInUseName:


            PlotTitle='2 color diff -'+RefCl +' Vs color'
            
            fileName=PlotTitle+'.html'
            figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,operation_hndlr.indexPanelNameDic,100,sectionNumber2Show)
        ##########################################################################################
        ##########################################################################################
        db= operation_hndlr.col1_col2_Sgoly_allPanels_continues;
        RefC03='Orange'
        if RefC03 in colorInUseName:
            RefCl=RefC03
            
            PlotTitle='2 color diff - '+RefCl +' Vs color'
            
            fileName=PlotTitle+'.html'
            figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,operation_hndlr.indexPanelNameDic,100,sectionNumber2Show)

        ##########################################################################################
        db= operation_hndlr.col2_col3_Sgoly_allPanels_continues;

        RefC04='Yellow'
        RefCl=RefC04

        if RefC04 in colorInUseName:
            PlotTitle='2 color diff -'+RefCl +' Vs color'
            
            fileName=PlotTitle+'.html'
            figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,operation_hndlr.indexPanelNameDic,100,sectionNumber2Show)


        ##########################################################################################
        db= operation_hndlr.col3_col4_Sgoly_allPanels_continues;
        RefC05='Black'
        RefCl=RefC05

        if RefC05 in colorInUseName:

            PlotTitle='2 color diff - '+RefCl +' Vs color'
            
            fileName=PlotTitle+'.html'
            figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,operation_hndlr.indexPanelNameDic,100,sectionNumber2Show)

        ##########################################################################################
        db= operation_hndlr.col4_col5_Sgoly_allPanels_continues;

        RefC06='Blue'
        if RefC06 in colorInUseName:

            RefCl=RefC06
            
            
            PlotTitle='2 color diff -'+RefCl +' Vs color'
            
            fileName=PlotTitle+'.html'
            figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,operation_hndlr.indexPanelNameDic,100,sectionNumber2Show)


        ##########################################################################################
        db= operation_hndlr.col5_col6_Sgoly_allPanels_continues;

        RefC07='Green'
        if RefC07 in colorInUseName:

            RefCl=RefC07
            
            
            PlotTitle='2 color diff -'+RefCl +' Vs color'
            
            fileName=PlotTitle+'.html'
            figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,operation_hndlr.indexPanelNameDic,100,sectionNumber2Show)


        ##########################################################################################
        
        db= operation_hndlr.ClrDF_fromTarget_allPanels_continues;
        PlotTitle='Single color from Target-NO FILTER'
        
        fileName=PlotTitle+'.html'
        figCyanVsClr_multiPanel_colorFromTarget= PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,operation_hndlr.indexPanelNameDic,100,sectionNumber2Show,0)
                
        ##########################################################################################
        db= operation_hndlr.ClrDF_fromTargetS_goly_allPanels_continues;
        PlotTitle='Single color from Target'

        fileName=PlotTitle+'.html'
        figCyanVsClr_multiPanel_colorFromTarget= PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,operation_hndlr.indexPanelNameDic,100,sectionNumber2Show,figCyanVsClr_multiPanel_colorFromTarget)
   
        #######################Save
        self.logger_signal.log("Done! ")

            
    def closeEvent(self, event):
        # emit the custom signal
        self.closed.emit()
        super().closeEvent(event)       
        
        
        



class TimeMonitorThread(threading.Thread):
    def __init__(self, func, timeout,ImRoi,StartRoiCoed):
        super().__init__()
        self.func = func
        self.timeout = timeout
        self.result = None
        self.elapsed_time = None
        self.exception = None
        self.ImRoi=ImRoi
        self.StartRoiCoed=StartRoiCoed

    def run(self):
        start_time = time.time()
        try:
            self.result = self.func(self.ImRoi,self.StartRoiCoed)
        except Exception as e:
            self.exception = e
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print('start_time:'+str(start_time))
        print('end_time:'+str(end_time))
        print('elapsed_time:'+str(elapsed_time))
        print('self.timeout:'+str(self.timeout))


        if elapsed_time > self.timeout:
            print('Time Over')
            self.exception=TimeoutError(f"Function execution time exceeded {self.timeout} seconds")


def plot_histogram(arr, num_bins='auto'):
    flattened_arr = arr.flatten()
    plt.hist(flattened_arr, bins=num_bins)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

def calculate_average_2d_array(arr):
    row_count = len(arr)
    column_count = len(arr[0])

    total_sum = 0
    element_count = 0

    for row in arr:
        for element in row:
            total_sum += element
            element_count += 1

    average = total_sum / element_count
    return average



class Circles():
    def __init__(self, pthF):
        self.pthF = pthF;
    
    
    def loadImageOLD(self):
        
        imInp_Orig = cv2.imread(self.pthF)
    
        Roi={}
    
        Roi['x']=[560,8130]
        Roi[0]=[600,1000]
        Roi[1]=[6300,6700]
        Roi[2]=[11400,11800]
        
        ImRoi={}
    
        ImRoi[0]=imInp_Orig[Roi['x'][0]:Roi['x'][1],Roi[0][0]:Roi[0][1],:]
        ImRoi[1]=imInp_Orig[Roi['x'][0]:Roi['x'][1],Roi[1][0]:Roi[1][1],:]
        ImRoi[2]=imInp_Orig[Roi['x'][0]:Roi['x'][1],Roi[2][0]:Roi[2][1],:]
        
        return ImRoi
    
    def loadImage_SmallSubstrate(self):
    
        imInp_Orig = cv2.imread(self.pthF)
    
        Roi={}

        if len(x_Cut_coord):
           Roi['x'] =  x_Cut_coord;
        else:
           Roi['x']=[660,8380]
           
        Roi[0]=[2200,2500]
        Roi[1]=[2750,3050]
        Roi[2]=[3250,3550]
        Roi[3]=[3750,4050]
        Roi[4]=[4300,4600]
        Roi[5]=[4800,5100]
        Roi[6]=[5300,5600]
        Roi[7]=[5800,6100]
        Roi[8]=[6300,6600]
        Roi[9]=[6850,7150]
        Roi[10]=[7350,7650]
        Roi[11]=[7850,8150]
        Roi[12]=[8400,8700]
        Roi[13]=[8900,9200]
        Roi[14]=[9400,9700]
        Roi[15]=[9900,10200]
        
        StartRoiCoed={key:value[0]-Roi[0][0] for key,value in Roi.items() if key != 'x'}
        
        ImRoi={}


        for i in range(NumOfSec):
           ImRoi[i]=imInp_Orig[Roi['x'][0]:Roi['x'][1],Roi[i][0]:Roi[i][1],:]
           
        # for key,value in ImRoi.items():
        #    plt.figure(key)
        #    plt.imshow(value)    
   
        return ImRoi,StartRoiCoed,Roi




    
    def loadImage(self):
    
        imInp_Orig = cv2.imread(self.pthF)
    
        Roi={}
        
        if len(x_Cut_coord):
           Roi['x'] =  x_Cut_coord;
        else:
           Roi['x']=[660,8380]
        
        Roi[0]=[700,1000]
        Roi[1]=[2200,2500]
        Roi[2]=[2750,3050]
        Roi[3]=[3250,3550]
        Roi[4]=[3750,4050]
        Roi[5]=[4300,4600]
        Roi[6]=[4800,5100]
        Roi[7]=[5300,5600]
        Roi[8]=[5800,6100]
        Roi[9]=[6300,6600]
        Roi[10]=[6850,7150]
        Roi[11]=[7350,7650]
        Roi[12]=[7850,8150]
        Roi[13]=[8400,8700]
        Roi[14]=[8900,9200]
        Roi[15]=[9400,9700]
        Roi[16]=[9900,10200]
        Roi[17]=[11450,11750]
        
        StartRoiCoed={key:value[0]-Roi[0][0] for key,value in Roi.items() if key != 'x'}
        
        ImRoi={}


        for i in range(NumOfSec):
           ImRoi[i]=imInp_Orig[Roi['x'][0]:Roi['x'][1],Roi[i][0]:Roi[i][1],:]
           
        # for key,value in ImRoi.items():
        #    plt.figure(key)
        #    plt.imshow(value)    
   
        return ImRoi,StartRoiCoed,Roi
    
    def find_circles(self,image,AreaTH):
        
        gray=np.min(image, axis=2)
        circle_image = np.ones_like(gray)*220
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # plt.Figure()
        # plt.imshow(edges)
        
        
        # Find contours in the edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty array to store the circle centers
        circle_centers = []
        areas=[]
        
        # Iterate over the contours and find the center of each circle
        for contour in contours:
            area = cv2.contourArea(contour)
            areas.append(area)
            # if area > 70 and area <90:  # Adjust the area threshold as needed
            if area > AreaTH:  # Adjust the area threshold as needed
        
                # Calculate the moments of the contour
                M = cv2.moments(contour)
        
                # Calculate the center of the contour
                center_xINT = int(M["m10"] / M["m00"])
                center_yINT = int(M["m01"] / M["m00"])
                
                center_x = (M["m10"] / M["m00"])
                center_y = (M["m01"] / M["m00"])
        
        
                # Append the center coordinates to the array
                circle_centers.append((center_x, center_y))
                # Draw a circle at the center of the contour
                cv2.circle(circle_image, (center_xINT, center_yINT), 5, (CircleLvl, CircleLvl, CircleLvl), -1)
                
                
        return circle_centers,gray,circle_image,edges
    
    
    def CreateCicles(self,center_x,center_y,gray):
        
        circle_image = np.ones_like(gray)*220

        
        for cx,cy in zip(center_x.columns,center_y.columns):
            for i in range(len(center_x[cx])):
                center_xINT = int(center_x[cx][i])
                center_yINT = int(center_y[cy][i])
                cv2.circle(circle_image, (center_xINT, center_yINT), 5, (CircleLvl, CircleLvl, CircleLvl), -1)
        
        return circle_image

    def SortCircle_Coord_y(self,circle_centers):
        
        circles_sorted2 = sorted(circle_centers, key=lambda x: x[1])

        x_values = [center[0] for center in circles_sorted2]

        y_values = [center[1] for center in circles_sorted2]


        # plt.figure()
        # # plt.plot(y_values,'-x')
        # plt.plot(np.diff(y_values),'-x')

        current_line={}
        indeces_y=np.argwhere(abs(np.diff(y_values))>(DistBtwCrcle/2))
        ii=0
        for indx in indeces_y:
            if ii==0:
                for k in range(indx[0]+1):
                    current_line[k]=[(x_values[k],y_values[k])]
                    ii=indx[0]+1
                    continue;
            for j in range(ii,indx[0]+1):
                gotValue=False

                for k in current_line.keys():
                    if  abs(current_line[k][len( current_line[k])-1][0]- x_values[j])<(DistBtwCrcle/5):
                        current_line[k].append((x_values[j],y_values[j]))
                        gotValue=True
                        break;
                if not gotValue:
                        current_line[len(current_line)]=[(x_values[j],y_values[j])]
            ii=indx[0]+1


        for j in range(ii,len(y_values)):
            gotValue=False

            for k in current_line.keys():
                if  abs(current_line[k][len( current_line[k])-1][0]- x_values[j])<(DistBtwCrcle/5):
                    current_line[k].append((x_values[j],y_values[j]))
                    gotValue=True
                    break;
            if not gotValue:
                    current_line[len(current_line)]=[(x_values[j],y_values[j])]
                     



        current_line_sorted={}
        avregeLoc={}

        for key in current_line.keys():
            x_values = [center[0] for center in current_line[key]]
            avregeLoc[np.mean(x_values)]=key

        sorted_avregeLoc = sorted(list(avregeLoc.keys()))
        sorted_Index=[avregeLoc[itm] for itm in sorted_avregeLoc]

        for i,k in enumerate(sorted_Index):
            current_line_sorted[i]=current_line[k]
            
        return current_line_sorted        
    
    def SortCircle_Coord(self,circle_centers):
        
        
        circles_sorted1 = sorted(circle_centers, key=lambda x: x[0])
        
        x_values = [center[0] for center in circles_sorted1]
    
        y_values = [center[1] for center in circles_sorted1]
        
        current_line = {}
    
        k=0
        l=[]
    
        for i in range(len(x_values) - 1):
      
            diff = x_values[i + 1] - x_values[i]
            l.append((x_values[i],y_values[i]))
      
            if abs(diff) > 5:  # Adjust the threshold as needed
                if len(l)<10:
                    l=[] 
                    continue;
                current_line[k]=l
                l=[]        
                k=k+1;    
    
        l.append((x_values[len(x_values)-1],y_values[len(x_values)-1]))
    
        current_line[k]=l
    
        current_line_sorted={}
        for k,l in current_line.items():
            l_sorted = sorted(l, key=lambda x: x[1])
            current_line_sorted[k]=l_sorted
            
        return current_line_sorted
    def Find_MIssDetected_Circles(self,current_line_sorted):   
        
        
        yFirst=[]
        yLast=[]
        current_line_sorted_copy={}

        for k in current_line_sorted.keys():   
            y = [center[1] for center in current_line_sorted[k]] 
            yFirst.append(y[0])
            yLast.append(y[len(y)-1])

        yF= np.min(yFirst )
        yL=np.max(yLast)

        for k in current_line_sorted.keys():    
            
            y = [center[1] for center in current_line_sorted[k]]
            
            dymean=np.mean(np.diff(y))
            
            if dymean > DistBtwCrcle*1.02:
                dymean = DistBtwCrcle*AQMscale
            
            dymin=min(np.diff(y))

            x = [center[0] for center in current_line_sorted[k]]
            degree = 4

            coefficients = np.polyfit(y, x, degree)

            # Create a polynomial function using the coefficients
            poly_function = np.poly1d(coefficients)

            # Generate x values for the fitted curve
            y_fit = np.linspace(yF, yL,numberOfPoints)

            # Calculate corresponding y values based on the fitted polynomial
            x_fit = poly_function(y_fit)
            
            current_line_sorted_copy[k]=[]

            indices =np.argwhere(abs(np.diff(np.array(y)))< (dymean*2/3))
            
            if not len(indices):
                current_line_sorted_copy[k]=current_line_sorted[k]
                continue;
            
            indices_lst=[]
            
            for indice in indices:
                indices_lst.append(indice[0])
            
            try:
               pairs_dict,strtInx,endInx=check_and_store_pairs(indices_lst)
            except:
               print(k)
               print(indices_lst)
               current_line_sorted_copy[k]=current_line_sorted[k]
               continue;
            
           
            st=0;
            for PairNum in range(len(strtInx)):
               
                current_line_sorted_copy[k]=current_line_sorted_copy[k]+current_line_sorted[k][st:strtInx[PairNum]+1]
          
                    
                distBtwnDots=current_line_sorted[k][endInx[PairNum]+1][1]- current_line_sorted[k][strtInx[PairNum]][1]
                for i in range(1,int(distBtwnDots/dymean)):
                    # print(i)
                    current_line_sorted_copy[k].append((x_fit[strtInx[PairNum]+i],current_line_sorted[k][strtInx[PairNum]][1]+dymean*i))
                    
                # current_line_sorted_copy.append(current_line_sorted[4][endInx[k]+1])
                st=endInx[PairNum]+1
                
                
                
            current_line_sorted_copy[k]=current_line_sorted_copy[k]+current_line_sorted[k][st:len(current_line_sorted[k])]
            
        return current_line_sorted_copy
        
        
    
    def AddMissingCircles_withPolyFIT(self,current_line_sorted):   
        cntMissingCircles={}
        
        
        
        yFirst=[]
        yLast=[]
        
        for k in current_line_sorted.keys():   
            y = [center[1] for center in current_line_sorted[k]] 
            yFirst.append(y[0])
            yLast.append(y[len(y)-1])
        
        yF= np.min(yFirst )
        yL=np.max(yLast)
       
        for k in current_line_sorted.keys():    
            
            if len(current_line_sorted[k])>=numberOfPoints:
                continue;
            
            y = [center[1] for center in current_line_sorted[k]]
            
            dymean=np.mean(np.diff(y))
            
            if dymean > DistBtwCrcle*1.02:
                dymean = DistBtwCrcle*AQMscale
            
            dymax=max(np.diff(y))
            cntMissingCircles[k]=0;
           
            while abs(dymax)> abs(dymean*1.3):
                x = [center[0] for center in current_line_sorted[k]]
                degree = 4
        
                coefficients = np.polyfit(y, x, degree)
        
                # Create a polynomial function using the coefficients
                poly_function = np.poly1d(coefficients)
        
                # Generate x values for the fitted curve
                y_fit = np.linspace(yF, yL,numberOfPoints)
        
                # Calculate corresponding y values based on the fitted polynomial
                x_fit = poly_function(y_fit)
                
                if abs(dymax)> abs(dymean*1.3):
                    maxIn= np.argmax(np.diff(y))
                    closest_index = min(range(len(y_fit)), key=lambda i: abs(y_fit[i] - (y[maxIn]+dymean)))
        
                    current_line_sorted[k].insert(maxIn+1, (x_fit[closest_index],y[maxIn]+dymean))
                y = [center[1] for center in current_line_sorted[k]]
                dymean=np.mean(np.diff(y))
                
                if dymean > DistBtwCrcle*1.02:
                    dymean = DistBtwCrcle*AQMscale
                    
                dymax=max(np.diff(y))
                cntMissingCircles[k]=cntMissingCircles[k]+1
                
                
                # print('abs('+str(dymax)+')> abs('+str(dymean*1.1)+')')
                
                
 
        
        
        
        for k in current_line_sorted.keys(): 
            
            if len(current_line_sorted[k])>numberOfPoints:
                continue;
            y = [center[1] for center in current_line_sorted[k]]
            x = [center[0] for center in current_line_sorted[k]]
            
            
            degree = 4
        
            coefficients = np.polyfit(y, x, degree)
        
            # Create a polynomial function using the coefficients
            poly_function = np.poly1d(coefficients)
        
            # Generate x values for the fitted curve
            y_fit = np.linspace(yF, yL,numberOfPoints)
        
            # Calculate corresponding y values based on the fitted polynomial
            x_fit = poly_function(y_fit)
            
        
            while len(y) != numberOfPoints:
        
                dyF=int(abs(y[0]-yF)/(dymean-5))
                if dyF>0:
                    closest_index = min(range(len(y_fit)), key=lambda i: abs(y_fit[i] - (current_line_sorted[k][0][1]-dymean)))
                    current_line_sorted[k].insert(0, (x_fit[closest_index],current_line_sorted[k][0][1]-dymean ))
                    cntMissingCircles[k]=cntMissingCircles[k]+1
                else:
                    # if abs(y[len(y)-1]- np.mean(yLast))>10:
                    closest_index = min(range(len(y_fit)), key=lambda i: abs(y_fit[i] - (y[len(y)-1] + dymean)))
            
                    current_line_sorted[k].append((x_fit[closest_index],y[len(y)-1] + dymean))
                    cntMissingCircles[k]=cntMissingCircles[k]+1
                y = [center[1] for center in current_line_sorted[k]]
                x = [center[0] for center in current_line_sorted[k]] 
                degree = 4
        
                coefficients = np.polyfit(y, x, degree)
        
                # Create a polynomial function using the coefficients
                poly_function = np.poly1d(coefficients)
        
                # Generate x values for the fitted curve
                y_fit = np.linspace(yF, yL,numberOfPoints)
        
                # Calculate corresponding y values based on the fitted polynomial
                x_fit = poly_function(y_fit)
                
                
                
        return current_line_sorted,cntMissingCircles

    
    
    
    
    
    
 ###   
    
    
    def AddMissingCircles(self,current_line_sorted):
        
        cntMissingCircles={}
        
        for k in current_line_sorted.keys():           
            
            y = [center[1] for center in current_line_sorted[k]]
            
            dymean=np.mean(np.diff(y))
            
            dymax=max(np.diff(y))
            cntMissingCircles[k]=0;
            
            while abs(dymax)> abs(dymean*1.5):
                x = [center[0] for center in current_line_sorted[k]]
                if abs(dymax)> abs(dymean*1.5):
                    maxIn= np.argmax(np.diff(y))
                    current_line_sorted[k].insert(maxIn+1, (np.mean(x),y[maxIn]+dymean))
                y = [center[1] for center in current_line_sorted[k]]
                dymean=np.mean(np.diff(y))
                dymax=max(np.diff(y))
                cntMissingCircles[k]=cntMissingCircles[k]+1
                

        yFirst=[]
        
        for k in current_line_sorted.keys():   
            y = [center[1] for center in current_line_sorted[k]] 
            yFirst.append(y[0])
        
        yF= np.min(yFirst )
        for k in current_line_sorted.keys():   
            y = [center[1] for center in current_line_sorted[k]]
            x = [center[0] for center in current_line_sorted[k]]
            
            while len(y) != numberOfPoints:
        
                dyF=int(abs(y[0]-yF)/(dymean-5))
                if dyF>0:
                    current_line_sorted[k].insert(0, (np.mean(x),current_line_sorted[k][0][1]-dymean ))
                    cntMissingCircles[k]=cntMissingCircles[k]+1
                else:
                    # if abs(y[len(y)-1]- np.mean(yLast))>10:
                    current_line_sorted[k].append((np.mean(x),y[len(y)-1] + dymean))
                    cntMissingCircles[k]=cntMissingCircles[k]+1
                y = [center[1] for center in current_line_sorted[k]]
                x = [center[0] for center in current_line_sorted[k]]     
                    
        return current_line_sorted,cntMissingCircles
        
    
    def CreatColorList(self,Clr,current_line_sorted):
        y= [center[1] for center in current_line_sorted[Clr]]
        return y
    
    
    def CreatColorList_x(self,Clr,current_line_sorted,StartRoiCoed_side):
        x= [center[0]+StartRoiCoed_side for center in current_line_sorted[Clr]]
        return x
    

    def CreatDF_meanMinusY(self,ClrDF):
        DFdiffVSmean = pd.DataFrame()
        tmp_df=pd.DataFrame()
        
        for col in ClrDF.columns:
            tmp_df = pd.DataFrame({col: np.diff(ClrDF[col])- np.mean(np.diff(ClrDF[col]))})
            DFdiffVSmean = pd.concat([DFdiffVSmean, tmp_df], axis=1)
         
        return DFdiffVSmean

    def Creat2ClrDiff(self,Clr1,Clr2,DFdiffVSmean):

        lnClr1= len(DFdiffVSmean[Clr1].notna())
        lnClr2= len(DFdiffVSmean[Clr2].notna())
        
        if lnClr1 == lnClr2:
            if Clr1 == Clr2:
                return [0]
            else:
                y = np.cumsum(DFdiffVSmean[Clr1])- np.cumsum(DFdiffVSmean[Clr2])
                return y
        else:
            return [0]
        
    def CalcIntegralError(self,ImRoi,Clr1):
       
           
       ImgClr={}
       ClrDF_raw={}

       for i in range(3):
           
           circle_centers,gray,circle_image,edges=self.find_circles(ImRoi[i],CircleArea);

           
           current_line_sorted=self.SortCircle_Coord(circle_centers);
           
           current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);

           
           ClrDF=pd.DataFrame()
           tmp_df=pd.DataFrame()
           for Key,Value in ColorDic.items():
               tmp_df = pd.DataFrame({Value: self.CreatColorList(Key,current_line_sorted)})
               ClrDF = pd.concat([ClrDF, tmp_df], axis=1)
           
           ClrDF_raw[i]=ClrDF
           DFdiffVSmean = self.CreatDF_meanMinusY(ClrDF)
               
           CoupleClr=  pd.DataFrame()
           tmp_df=pd.DataFrame()
           for Key,Value in ColorDic.items():
               y = self.Creat2ClrDiff(Clr1,Value,DFdiffVSmean)
               if len(y)>1:
                   tmp_df = pd.DataFrame({Clr1+'-'+Value: y*pixSize})  
                   CoupleClr = pd.concat([CoupleClr, tmp_df], axis=1)
                   
           ImgClr[i]=CoupleClr
           
       return ImgClr,gray,circle_image,edges,ClrDF_raw
 
    def CalcorColorMat(self,ImRoi,StartRoiCoed):
       
           
       ClrDF_raw={}
       ClrDF_rawXY={}

       current_line_sortedDIC={}
       
       
       circle_image_all={}
       gray_all={}
       
       for i in range(len(ImRoi.keys())):
           
           circle_centers,gray,circle_image,edges=self.find_circles(ImRoi[i],CircleArea);
           
           
           circle_image_all[i]=circle_image
           
           gray_all[i]=gray
           
           current_line_sorted=self.SortCircle_Coord(circle_centers);
           
           if len(current_line_sorted.keys()) < 7:
               current_line_sorted=self.SortCircle_Coord_y(circle_centers)

           
           if len(current_line_sorted.keys()) > 7:
                keysOverLimit=[key for key in  current_line_sorted.keys() if key>6]
                for key in keysOverLimit:
                    del current_line_sorted[key]           
           # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
           
           
           current_line_sorted,cntMissingCircles=self.AddMissingCircles_withPolyFIT(current_line_sorted);

           
           
           current_line_sorted = self.filterCircles_WithPol(current_line_sorted)
           
           current_line_sorted,cntMissingCircles=self.AddMissingCircles_withPolyFIT(current_line_sorted);
           
           current_line_sorted=self.Find_MIssDetected_Circles(current_line_sorted);

           current_line_sorted = self.filterCircles_WithPol(current_line_sorted)


           
           # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
           
           
           
           
           # if len(current_line_sorted.keys()) > 7:
           #     del current_line_sorted[7]
           
           # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
           
           # current_line_sorted = self.filterCircles(current_line_sorted)
           
           # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);

           
           ClrDF=pd.DataFrame()
           ClrDF_x=pd.DataFrame()

           tmp_df=pd.DataFrame()
           for Key,Value in ColorDic.items():
               tmp_df = pd.DataFrame({Value: self.CreatColorList(Key,current_line_sorted)})
               ClrDF = pd.concat([ClrDF, tmp_df], axis=1)
               ClrDF_x = pd.concat([ClrDF_x, tmp_df], axis=1)
               tmp_df_x = pd.DataFrame({Value+'_x': self.CreatColorList_x(Key,current_line_sorted,StartRoiCoed[i])})
               ClrDF_x = pd.concat([ClrDF_x, tmp_df_x], axis=1)

               
               
           
            
           current_line_sortedDIC[i] = current_line_sorted
           ClrDF_raw[i]=ClrDF
           ClrDF_rawXY[i]=ClrDF_x

           
       return gray_all,circle_image_all,edges,ClrDF_raw,ClrDF_rawXY,current_line_sortedDIC,StartRoiCoed
 
    
    def CalcorColorMat_forThreading(self,ImRoi,StartRoiCoed):
       
           
       ClrDF_raw={}
       ClrDF_rawXY={}

       current_line_sortedDIC={}
       
       
       circle_image_all={}
       gray_all={}
       
       for i in range(len(ImRoi.keys())):
           
           circle_centers,gray,circle_image,edges=self.find_circles(ImRoi[i],CircleArea);
           
           
           circle_image_all[i]=circle_image
           
           gray_all[i]=gray
           
           current_line_sorted=self.SortCircle_Coord(circle_centers);
           
           if len(current_line_sorted.keys()) < 7:
               current_line_sorted=self.SortCircle_Coord_y(circle_centers)

           
           if len(current_line_sorted.keys()) > 7:
                keysOverLimit=[key for key in  current_line_sorted.keys() if key>6]
                for key in keysOverLimit:
                    del current_line_sorted[key]           
           # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
           
           
           current_line_sorted,cntMissingCircles=self.AddMissingCircles_withPolyFIT(current_line_sorted);

           
           
           current_line_sorted = self.filterCircles_WithPol(current_line_sorted)
           
           current_line_sorted,cntMissingCircles=self.AddMissingCircles_withPolyFIT(current_line_sorted);
           
           current_line_sorted=self.Find_MIssDetected_Circles(current_line_sorted);

           current_line_sorted = self.filterCircles_WithPol(current_line_sorted)


           
           # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
           
           
           
           
           # if len(current_line_sorted.keys()) > 7:
           #     del current_line_sorted[7]
           
           # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
           
           # current_line_sorted = self.filterCircles(current_line_sorted)
           
           # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);

           
           ClrDF=pd.DataFrame()
           ClrDF_x=pd.DataFrame()

           tmp_df=pd.DataFrame()
           for Key,Value in ColorDic.items():
               tmp_df = pd.DataFrame({Value: self.CreatColorList(Key,current_line_sorted)})
               ClrDF = pd.concat([ClrDF, tmp_df], axis=1)
               ClrDF_x = pd.concat([ClrDF_x, tmp_df], axis=1)
               tmp_df_x = pd.DataFrame({Value+'_x': self.CreatColorList_x(Key,current_line_sorted,StartRoiCoed[i])})
               ClrDF_x = pd.concat([ClrDF_x, tmp_df_x], axis=1)

               
               
           
            
           current_line_sortedDIC[i] = current_line_sorted
           ClrDF_raw[i]=ClrDF
           ClrDF_rawXY[i]=ClrDF_x

       result={}
       result['gray_all']=gray_all
       result['circle_image_all']=circle_image_all
       result['edges']=edges
       result['ClrDF_raw']=ClrDF_raw
       result['ClrDF_rawXY']=ClrDF_rawXY
       result['current_line_sortedDIC']=current_line_sortedDIC
       result['StartRoiCoed']=StartRoiCoed
           
       return result
 
 
    
    def delete_indices_with_numpy(self,lst, indices):
        
        indices = np.sort(indices)[::-1]  # Sort indices in reverse order to avoid index shifting
        for index in indices:
            # print(index)
            if 0 <= index < len(lst):
                lst.pop(index[0])
        return lst
                
    def filterCircles(self,current_line_sorted):
        
        for k in  current_line_sorted.keys():
            y_values = [center[1] for center in current_line_sorted[k]]
            dymean= np.mean(np.diff(np.array(y_values)))
            indices = np.argwhere(abs(np.diff(np.array(y_values)))<(dymean/2))
            current_line_sorted[k] = self.delete_indices_with_numpy(current_line_sorted[k],indices)
                    
        return current_line_sorted

    def filterCircles_WithPol(self,current_line_sorted):
        
        yFirst=[]
        yLast=[]
        for k in current_line_sorted.keys():   
            y = [center[1] for center in current_line_sorted[k]] 
            yFirst.append(y[0])
            yLast.append(y[len(y)-1])

        yF= np.min(yFirst )
        yL=np.max(yLast)
       
        for k in  current_line_sorted.keys():
            y_values = [center[1] for center in current_line_sorted[k]]

            dymean= np.mean(np.diff(np.array(y_values)))
            indices = np.argwhere(abs(np.diff(np.array(y_values)))<(dymean*2/3))
            current_line_sorted[k] = self.delete_indices_with_numpy(current_line_sorted[k],indices)
            
            y_values = [center[1] for center in current_line_sorted[k]]
            x_values = [center[0] for center in current_line_sorted[k]]
            
            degree = 4

            coefficients = np.polyfit(y_values, x_values, degree)

            # Create a polynomial function using the coefficients
            poly_function = np.poly1d(coefficients)

            # Generate x values for the fitted curve
            y_fit = np.linspace(yF, yL,len(y_values))

            # Calculate corresponding y values based on the fitted polynomial
            x_fit = poly_function(y_fit)
            
            
            indeces_x=np.argwhere(abs(x_fit-x_values)>(DistBtwCrcle/2))
            current_line_sorted[k] = self.delete_indices_with_numpy(current_line_sorted[k],indeces_x)
        return current_line_sorted            
            

        
        
        
        
    def calcC2C(self,refClr,ClrDF_rawSide):
        
        
        l=[]

        C2Cmat=[]
        maxColorMinColor=[]
        Color=[]
        for i in ClrDF_rawSide.index:
            for col in ClrDF_rawSide.columns:
                if col == refClr:
                    continue;
                l.append(ClrDF_rawSide[refClr][i]-ClrDF_rawSide[col][i])
                Color.append(col)

            min_index = l.index(min(l)) 
            max_index = l.index(max(l))    
            maxColorMinColor.append([Color[max_index],Color[min_index]])
            C2Cmat.append((max(l)-min(l))*pixSize)
            l=[]
            Color=[]

        return C2Cmat,maxColorMinColor
    
    def calcC2C_v1(self,refClr,ClrDF_rawSide):
        
        
        l=[]
        
        C2Cmat=[]
        maxColorMinColor=[]
        Color=[]
        for i in ClrDF_rawSide.index:
            for col in ClrDF_rawSide.columns:
                if col == refClr:
                    continue;
                l.append(ClrDF_rawSide[refClr][i]-ClrDF_rawSide[col][i])
                Color.append(col)
        
        
            if min(l) <0 :
        
                C2Cmat.append((max(l)-min(l))*pixSize)
                min_index = l.index(min(l)) 
                max_index = l.index(max(l))    
                maxColorMinColor.append([Color[max_index],Color[min_index]])
                l=[] 
                Color=[]
            else:
                min_index = l.index(min(l)) 
                refClr=Color[min_index]
                l=[]
                Color=[]
                for col in ClrDF_rawSide.columns:
                    if col == refClr:
                        continue;
                    l.append(ClrDF_rawSide[refClr][i]-ClrDF_rawSide[col][i])
                    Color.append(col)
                    
                C2Cmat.append((max(l)-min(l))*pixSize)
                min_index = l.index(min(l)) 
                max_index = l.index(max(l))    
                maxColorMinColor.append([Color[max_index],Color[min_index]])
                l=[] 
                Color=[]

        return C2Cmat,maxColorMinColor
    
    def calcDiffernceFromeTarget(self,ClrDF_rawSide,dymeanList,colorInUseName,strartPos):
        
        yTarget=[]

        ClrDF_fromTarget=pd.DataFrame()
        ClrDF_fromTargetS_goly=pd.DataFrame()

        if not strartPos:
            strartPos=ClrDF_rawSide['Magenta'][0]
        # for col in ClrDF_rawSide.columns:
        #     dymean= np.mean(np.diff(ClrDF_rawSide[col])[:200])
        #     dymeanList.append(dymean)
        
        if  UseTarget:
           dy= DistBtwCrcle*AQMscale
        else:
           dy= np.mean(np.array(dymeanList))


        yTargetDF=pd.DataFrame()

        for col in colorInUseName:
            # dymean= np.mean(np.diff(ClrDF_rawSide[col])[:200])
            # dymeanList.append(dymean)
            yTarget = [i * dy + strartPos for i in range(len(ClrDF_rawSide[col]))]
            ClrDF_fromTarget[col]=(pd.Series(yTarget)- ClrDF_rawSide[col])
            ClrDF_fromTargetS_goly[col]=savgol_filter((pd.Series(yTarget)- ClrDF_rawSide[col]), MaxWaveWindow, S_g_Degree)
            yTargetDF=pd.concat((yTargetDF,pd.Series(yTarget).rename(col)),axis=1)
            yTarget=[]

        return ClrDF_fromTarget,ClrDF_fromTargetS_goly,dymeanList,yTargetDF,strartPos
    
    
    def calcDiffernceFromeTargetXY(self,ClrDF_rawSide,dymeanList,colorInUseName,strartPos,strartPos_x,DistanceBetweenColumns_side):
        
        yTarget=[]

        ClrDF_fromTarget=pd.DataFrame()
        ClrDF_fromTargetS_goly=pd.DataFrame()

        if not strartPos:
            strartPos=ClrDF_rawSide['Magenta'][0]
        # for col in ClrDF_rawSide.columns:
        #     dymean= np.mean(np.diff(ClrDF_rawSide[col])[:200])
        #     dymeanList.append(dymean)
        
        if not strartPos_x:
            strartPos_x=ClrDF_rawSide['Magenta_x'][0]+DistBtwCrcle
        
        
        strartPos_x = strartPos_x + DistanceBetweenColumns_side
        
        if  UseTarget:
           dy= DistBtwCrcle*AQMscale
        else:
           dy= np.mean(np.array(dymeanList))


        yTargetDF=pd.DataFrame()
        xTargetDF = pd.DataFrame()

        for j,col in enumerate(colorInUseName):
            # dymean= np.mean(np.diff(ClrDF_rawSide[col])[:200])
            # dymeanList.append(dymean)
            yTarget = [i * dy + strartPos for i in range(len(ClrDF_rawSide[col]))]
            xTarget = [strartPos_x for i in range(len(ClrDF_rawSide[col]))]
            ClrDF_fromTarget[col]=(pd.Series(yTarget)- ClrDF_rawSide[col])
            ClrDF_fromTargetS_goly[col]=savgol_filter((pd.Series(yTarget)- ClrDF_rawSide[col]), MaxWaveWindow, S_g_Degree)
            yTargetDF=pd.concat((yTargetDF,pd.Series(yTarget).rename(col)),axis=1)
            yTarget=[]
            xTargetDF=pd.concat((xTargetDF,pd.Series(xTarget).rename(col+'_x')),axis=1)
            xTarget=[]
            if j < len(colorInUseName)-1:
                # strartPos_x = DistBtwCrcle/AQMscale +strartPos_x
                strartPos_x = DistBtwCrcle/ScaleY +strartPos_x
        

        return ClrDF_fromTarget,ClrDF_fromTargetS_goly,dymeanList,yTargetDF,strartPos,strartPos_x,xTargetDF
    
    def CalcDiffTarget(self,ClrDF,dymeanList):
        
            for col in ClrDF.columns:
                dymean= np.mean(np.diff(ClrDF[col])[:200])
                dymeanList.append(dymean)
                
            return dymeanList
            

def Plot3subPlots(db,PlotTitle,fileName):
    
    fig = go.Figure()
    #fig_back = go.Figure()
    fig = make_subplots(rows=3, cols=1,subplot_titles=('left-side', 'Middle', 'right-side'), vertical_spacing=0.1, shared_xaxes=True)
   
    ln_list=[len(db[0].columns),len(db[1].columns),len(db[2].columns)]
    min_index = ln_list.index(min(ln_list))

    
    for c in db[min_index].columns:
        
        lineColor=c.split('-')[1]
        if lineColor=='Yellow':
                    lineColor='gold';
    # for i in rnge:
        fig.add_trace(go.Scatter(y=list(db[0][c]), line_color=lineColor,
                    name=c+' left-side'),row=1, col=1)
        
        fig.add_trace(go.Scatter(y=list(db[1][c]), line_color=lineColor,
                    name=c+' Middle'),row=2, col=1)
        
        fig.add_trace(go.Scatter(y=list(db[2][c]), line_color=lineColor,
                    name=c+' right-side'),row=3, col=1)
   
    
    fig.update_layout(title=PlotTitle)
    #fig_back.update_layout(title='ImagePlacement_Left-Back')
    fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
   
    
    fig.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    
    # datetime object containing current date and time
   
    plot(fig,auto_play=True,filename=fileName)  
    # plot(fig)  

    #plot(fig_back,filename="AQM-Back.html")  
    fig.show()
    
    return fig

def PlotSingle(db,PlotTitle,fileName, ColList, PageSide, IntegralError):
    
   fig = go.Figure()
   

   
   for c in ColList:  
       
       if IntegralError:
           lineColor=c.split('-')[1]
       else:
           lineColor = c
           
       if lineColor=='Yellow':
           lineColor='gold';
       
       fig.add_trace(go.Scatter(y=list(db[c]), line_color=lineColor,name=c+' ' +PageSide))
       
   
   titleColor = 'black'
   if IntegralError:
       titleColor=c.split('-')[0]
       if titleColor == 'Cyan':
           titleColor = '#008B8B';
       
       if titleColor == 'Yellow':
           titleColor = 'gold'; 
   
   fig.update_layout(title={
        'text': PlotTitle,
        'font': {'color': titleColor}
    })
     #fig_back.update_layout(title='ImagePlacement_Left-Back')
  
     
   fig.update_layout(
         hoverlabel=dict(
             namelength=-1
         )
     )
     
     # datetime object containing current date and time
    
   plot(fig,auto_play=True,filename=fileName)  
     # plot(fig)  
 
     #plot(fig_back,filename="AQM-Back.html")  
   fig.show()
     
   return fig   


def PlotSingle_wAverage(x,db,PlotTitle,fileName, ColList, PageSide, IntegralError,AverageOffset):
    
   fig = go.Figure()
   

   
   for c in ColList:  
       
       if IntegralError:
           lineColor=c.split('-')[1]
       else:
           lineColor = c
           
       if lineColor=='Yellow':
           lineColor='gold';
       if AverageOffset == 'Average':
           fig.add_trace(go.Scatter(x=x,y=list(db[c]-np.mean(db[c])), line_color=lineColor,name=c+' ' +PageSide))
       else:
           fig.add_trace(go.Scatter(x=x,y=list(db[c]-db[c][0]), line_color=lineColor,name=c+' ' +PageSide))

       
   
   titleColor = 'black'
   if IntegralError:
       titleColor=c.split('-')[0]
       if titleColor == 'Cyan':
           titleColor = '#008B8B';
       
       if titleColor == 'Yellow':
           titleColor = 'gold'; 
   
   fig.update_layout(title={
        'text': PlotTitle,
        'font': {'color': titleColor}
    })
     #fig_back.update_layout(title='ImagePlacement_Left-Back')
  
     
   fig.update_layout(
         hoverlabel=dict(
             namelength=-1
         )
     )
     
     # datetime object containing current date and time
    
   plot(fig,auto_play=True,filename=fileName)  
     # plot(fig)  
 
     #plot(fig_back,filename="AQM-Back.html")  
   fig.show()
     
   return fig   



def PlotSingleWstarvitsky(x,db,PlotTitle,fileName, ColList, PageSide, IntegralError):
    
   fig = go.Figure()
   

   
   for c in ColList:  
       
       if IntegralError:
           lineColor=c.split('-')[1]
       else:
           lineColor = c
           
       if lineColor=='Yellow':
           lineColor='gold';
       
       fig.add_trace(go.Scatter(x=x,y=list(db[c]),mode='lines',
                                line=dict(color=lineColor, dash='solid'),name=c+' ' +PageSide))
       fig.data[len(fig.data)-1].visible = 'legendonly';
       fig.add_trace(go.Scatter(x=x,y=savgol_filter(db[c], MaxWaveWindow, S_g_Degree), 
                                line=dict(color=lineColor, dash='solid') ,name='S.Goly ' + c+' ' +PageSide))

       
   
   titleColor = 'black'
   if IntegralError:
       titleColor=c.split('-')[0]
       if titleColor == 'Cyan':
           titleColor = '#008B8B';
       
       if titleColor == 'Yellow':
           titleColor = 'gold'; 
   
   fig.update_layout(title={
        'text': PlotTitle,
        'font': {'color': titleColor}
    })
     #fig_back.update_layout(title='ImagePlacement_Left-Back')
  
     
   fig.update_layout(
         hoverlabel=dict(
             namelength=-1
         )
     )
     
     # datetime object containing current date and time
    
   plot(fig,auto_play=True,filename=fileName)  
     # plot(fig)  
 
     #plot(fig_back,filename="AQM-Back.html")  
   fig.show()
     
   return fig  

def PlotSingle_Wstarvitsky_allInOneGraph(x,dbwAll,PlotTitle,fileName,colorInUseName):
    
   fig = go.Figure()
   
   

   for i in range(len(dbwAll)):
       db=dbwAll[i]
       
       for c in colorInUseName:  
           
           lineColor = c
               
           if lineColor=='Yellow':
               lineColor='gold';
           
           fig.add_trace(go.Scatter(x=x,y=list(db[c]),mode='lines',
                                    line=dict(color=lineColor),name=c+' section ' +str(i)))
           fig.data[len(fig.data)-1].visible = 'legendonly';
           fig.add_trace(go.Scatter(x=x,y=savgol_filter(db[c], MaxWaveWindow, S_g_Degree), 
                                    line=dict(color=lineColor) ,name='S.Goly ' + c+' section ' +str(i)))

       
   

   
   fig.update_layout(title={
        'text': PlotTitle,
        'font': {'color': 'black'}
    })
     #fig_back.update_layout(title='ImagePlacement_Left-Back')
  
     
   fig.update_layout(
         hoverlabel=dict(
             namelength=-1
         )
     )
     
     # datetime object containing current date and time
    
   plot(fig,auto_play=True,filename=fileName)  
     # plot(fig)  
 
     #plot(fig_back,filename="AQM-Back.html")  
   fig.show()
     
   return fig  



def PlotSingle_Basic(x,db,PlotTitle,fileName):
    
   fig = go.Figure()
   

   
   for c in db.columns:  
       
       if c in list(ColorDic.values()):
            
           fig.add_trace(go.Scatter(x=x,y=list(db[c]),line_color=c ,name=c))
       else:
           fig.add_trace(go.Scatter(x=x,y=list(db[c]),name=c))
 
       
   
   titleColor = 'black'

   
   fig.update_layout(title={
        'text': PlotTitle,
        'font': {'color': titleColor}
    })
     #fig_back.update_layout(title='ImagePlacement_Left-Back')
  
     
   fig.update_layout(
         hoverlabel=dict(
             namelength=-1
         )
     )
     
     # datetime object containing current date and time
    
   plot(fig,auto_play=True,filename=fileName)  
     # plot(fig)  
 
     #plot(fig_back,filename="AQM-Back.html")  
   fig.show()
     
   return fig   

def PlodSide2Side(db,PlotTitle,fileName,indexPanelNameDic,ymax):
    
    fig = go.Figure()

    for c in db.columns:  
       
        lineColor=c
            
        if lineColor=='Yellow':
            lineColor='gold';
            
        fig.add_trace(go.Scatter(y=list(db[c]*pixSize), line=dict(color=lineColor, dash='solid')  ,name=c))
            

     
    fig= Plot_Panel_number(fig,ymax,indexPanelNameDic)
     

    titleColor=c.split('-')[0]
    if titleColor == 'Cyan':
         titleColor = '#008B8B';
     
    if titleColor == 'Yellow':
         titleColor = 'gold'; 
     
    fig.update_layout(title={
          'text': PlotTitle,
          'font': {'color': titleColor}
      })
      #fig_back.update_layout(title='ImagePlacement_Left-Back')
      
      
    fig.update_layout(
          hoverlabel=dict(
              namelength=-1
          )
      )
      
      # datetime object containing current date and time
     
    plot(fig,auto_play=True,filename=fileName)  
      # plot(fig)  
     
      #plot(fig_back,filename="AQM-Back.html")  
    fig.show()
    
    return fig
    
    
def PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,ymax,sectionNumber,fig):
    
   preWord='S_g_'
   dontShow=1
   if not fig:
       fig = go.Figure()
       preWord=''
       dontShow=9
   

   
   for i in sectionNumber:
       for c in db[i].columns:  
           lineColor = c
                
           if lineColor=='Yellow':
                lineColor='gold';
                
           fig.add_trace(go.Scatter(y=list(db[i][c]*pixSize),line=dict(color=lineColor) , name=preWord+c+' section '+str(i)))
       
   
    
   fig= Plot_Panel_number(fig,ymax,indexPanelNameDic)
    
   titleColor = 'black'

    
   fig.update_layout(title={
        'text': PlotTitle,
        'font': {'color': titleColor}
    })
     #fig_back.update_layout(title='ImagePlacement_Left-Back')
  
     
   fig.update_layout(
         hoverlabel=dict(
             namelength=-1
         )
     )
     
     # datetime object containing current date and time
   if dontShow:
    
       plot(fig,auto_play=True,filename=fileName)  
         # plot(fig)  
     
         #plot(fig_back,filename="AQM-Back.html")  
       fig.show()
     
   return fig   

def PlotSingle_Basic_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,ymax):
    
   fig = go.Figure()
   

   for c in db.columns:  

             
        fig.add_trace(go.Scatter(y=list(db[c]), name=c))
       
   
    
   fig= Plot_Panel_number(fig,ymax,indexPanelNameDic)
    
   titleColor = 'black'

    
   fig.update_layout(title={
        'text': PlotTitle,
        'font': {'color': titleColor}
    })
     #fig_back.update_layout(title='ImagePlacement_Left-Back')
  
     
   fig.update_layout(
         hoverlabel=dict(
             namelength=-1
         )
     )
     
     # datetime object containing current date and time
    
   plot(fig,auto_play=True,filename=fileName)  
     # plot(fig)  
 
     #plot(fig_back,filename="AQM-Back.html")  
   fig.show()
     
   return fig   

def PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,ymax,sectionNumber):
    
   fig = go.Figure()
   
   # dashSolidDot={0:'solid',1:'dash',2:'dot'}
   # PageSide={0:'Left',1:'Middle',2:'Right'}
  
    
   for i in sectionNumber:
        for c in db[i].columns:  
           
            lineColor=c.split('-')[1]
                
            if lineColor=='Yellow':
                lineColor='gold';
                
                # fig.add_trace(go.Scatter(y=list(db[i][c]*pixSize), line=dict(color=lineColor, dash=dashSolidDot[i])  ,name=c+' '+PageSide[i]))
            fig.add_trace(go.Scatter(y=list(db[i][c]*pixSize), line=dict(color=lineColor)  ,name=c+' section '+str(i)))

   
    
   fig= Plot_Panel_number(fig,ymax,indexPanelNameDic)
    

   titleColor=c.split('-')[0]
   if titleColor == 'Cyan':
        titleColor = '#008B8B';
    
   if titleColor == 'Yellow':
        titleColor = 'gold'; 
    
   fig.update_layout(title={
         'text': PlotTitle,
         'font': {'color': titleColor}
     })
     #fig_back.update_layout(title='ImagePlacement_Left-Back')
  
     
   fig.update_layout(
         hoverlabel=dict(
             namelength=-1
         )
     )
     
     # datetime object containing current date and time
    
   plot(fig,auto_play=True,filename=fileName)  
     # plot(fig)  
 
     #plot(fig_back,filename="AQM-Back.html")  
   fig.show()
     
   return fig   

def PlotSingle_for_a_panel(db,PlotTitle,fileName,sectionNumber,cutLength):
    
   fig = go.Figure()
   
   # dashSolidDot={0:'solid',1:'dash',2:'dot'}
   # PageSide={0:'Left',1:'Middle',2:'Right'}
  
   for numOfpanels in range(len(db)): 
       for i in sectionNumber:
            for c in db[numOfpanels][i].columns:  
               
                lineColor=c.split('-')[1]
                    
                if lineColor=='Yellow':
                    lineColor='gold';
                    
                    # fig.add_trace(go.Scatter(y=list(db[i][c]*pixSize), line=dict(color=lineColor, dash=dashSolidDot[i])  ,name=c+' '+PageSide[i]))
                fig.add_trace(go.Scatter(y=list(db[numOfpanels][i][c][:cutLength]*pixSize), line=dict(color=lineColor)  ,name=c+' section '+str(i)+' cycle= '+str(numOfpanels)))

   
    
    

   titleColor=c.split('-')[0]
   if titleColor == 'Cyan':
        titleColor = '#008B8B';
    
   if titleColor == 'Yellow':
        titleColor = 'gold'; 
    
   fig.update_layout(title={
         'text': PlotTitle,
         'font': {'color': titleColor}
     })
     #fig_back.update_layout(title='ImagePlacement_Left-Back')
  
     
   fig.update_layout(
         hoverlabel=dict(
             namelength=-1
         )
     )
     
     # datetime object containing current date and time
    
   plot(fig,auto_play=True,filename=fileName)  
     # plot(fig)  
 
     #plot(fig_back,filename="AQM-Back.html")  
   fig.show()
     
   return fig   

def Plot_Panel_number(fig,ymax,indexPanelNameDic):
    pValue=0
    colorFlag=0
    greenRedLine={1:'green',0:'red'}
    for key, value in indexPanelNameDic.items():
         if (int(value.split('-')[-1])-pValue)>1:
             colorFlag=abs(colorFlag-1)
         fig.add_trace(go.Scatter(x=[key], y=[ymax],
                                 marker=dict(color=greenRedLine[colorFlag], size=10),
                                 mode="markers",
                                 text=value,
                                 # font_size=18,
                                 hoverinfo='text'))
         
         fig.data[len(fig.data)-1].showlegend = False
         fig.add_vline(x=key, line_width=0.5, line_dash="dash", line_color="green")
 
    return fig


# def PlotRegistrationMapForPnls(ClrDF_fromTargetS_goly_allPanels,colorInUseName,clrToUse,pnelToShow,NumOfSecToShow,sInput,fileNME,scaling_factor,offset_factor):
    
#     # clrToUse=['Cyan','Black','Yellow']
#     # clrInUse=['Cyan','Black','Magenta']
#     # clrInUse=['Magenta','Yellow','Blue','Orange','Cyan','Green','Black']

#     # clrInUse=['Yellow','Orange','Magenta']
#     # Convert lists to sets
#     set1 = set(colorInUseName)
#     set2 = set(clrToUse)
    
#     # Find elements that are in one set but not the other
 
#     unequal_elements = set1.symmetric_difference(set2)
    
        
    
#     C2Cmat_allPanels={}
#     MaxMinColor_allPanels={}
#     for Pnl in pnelToShow: 
#         try:
#             C2Cmat=pd.DataFrame()
#             MaxMinColor=pd.DataFrame()
#             for i in range(len(ClrDF_fromTargetS_goly_allPanels[Pnl])):
#                 if not len(unequal_elements):
#                     C2Cmat[i],MaxMinColor[i] = Circles(sInput+'\\'+Pnl+fileNME).calcC2C_v1('Magenta', ClrDF_fromTargetS_goly_allPanels[Pnl][i][clrToUse]);
#                 else:
#                     C2Cmat[i],MaxMinColor[i] = Circles(sInput+'\\'+Pnl+fileNME).calcC2C(list(unequal_elements)[0], ClrDF_fromTargetS_goly_allPanels[Pnl][i][clrToUse+[list(unequal_elements)[0]]]);
                
#             # C2Cmat_allPanels= pd.concat([C2Cmat_allPanels, C2Cmat])  
#             C2Cmat_allPanels[Pnl]=   C2Cmat
#             MaxMinColor_allPanels[Pnl]= MaxMinColor
#         except:
#             1
            
    
#     for Pnl in pnelToShow:
#         try:
#             fig = go.Figure()
        
            
#             for sec in range(NumOfSecToShow):
            
#                 l=list(MaxMinColor_allPanels[Pnl][sec])
#                 result = find_change_index(l)
#                 ll=list(C2Cmat_allPanels[Pnl][sec])
#                 mirrored_result,modSignal = mirror_signal_along_x_axis(ll,scaling_factor,offset_factor)
#                 secPos=np.mean(ClrDF_rawXY_allPanels[Pnl][sec]['Magenta_x'])
#                 # mirrored_result=list(C2Cmat_allPanels[Pnl][0]-1)
#                 ll1=list(np.asarray(modSignal)+secPos)
#                 mirrored_result1=list(np.asarray(mirrored_result)+secPos)
            
#                 rnge=list(range(int(len(ll))))
#                 i=0
#                 clr1=l[i][0]
#                 clr2=l[i][1]
#                 for res in result:
#                     clr1=l[i][0]
#                     clr2=l[i][1]
#                     if clr1 == 'Yellow':
#                         clr1='Gold'
#                     if clr2 == 'Yellow':
#                         clr2='Gold'    
#                     string_list = ["{:.2f}".format(element) for element in list(ll[i:res])]    
#                     fig.add_trace(go.Scatter(y=list(rnge[i:res]),x=list(ll1[i:res]), line=dict(color=clr1,width=1.5) ,mode='lines' ,text=string_list,
#                     # font_size=18,
#                     hoverinfo='text', showlegend=False))
#                     fig.add_trace(go.Scatter(y=list(rnge[i:res]),x=list(mirrored_result1[i:res]), line=dict(color=clr2,width=1.5),mode='lines',text=string_list,
#                     # font_size=18,
#                     hoverinfo='text', showlegend=False))
#                     i=res-1
                    
#                 string_list = ["{:.2f}".format(element) for element in list(ll[i:])]    
#                 fig.add_trace(go.Scatter(y=list(rnge[i:]),x=list(ll1[i:]), line=dict(color=clr1,width=2)  , mode='lines',text=string_list,
#                 # font_size=18,
#                 hoverinfo='text',showlegend=False))
#                 fig.add_trace(go.Scatter(y=list(rnge[i:]),x=list(mirrored_result1[i:]), line=dict(color=clr2,width=2),mode='lines',text=string_list,
#                 # font_size=18,
#                 hoverinfo='text', showlegend=False))
        
#                     # plt.plot(rnge[i:res],ll[i:res],color=clr1,linewidth=2)
#                     # plt.plot(rnge[i:res],mirrored_result[i:res],color=clr2,linewidth=2)
            
#             fig.update_layout(title={
#                  'text': 'Flat Number '+Pnl,
#                  'font': {'color': 'black'}
#              })
#             plot(fig,auto_play=True,filename='Flat Number '+Pnl+'.html')  
#               # plot(fig)  
          
#               #plot(fig_back,filename="AQM-Back.html")  
#             fig.show()
#         except:
#             1
        
    
# def   CalcStatistics(Color_vs_Sgoly_allPanels_continues):
    
#     statisticalData={}
#     ColMean= {col: [] for col in Color_vs_Sgoly_allPanels_continues[sec].columns}
#     Colstd={col: [] for col in Color_vs_Sgoly_allPanels_continues[sec].columns}
#     Col95p={col: [] for col in Color_vs_Sgoly_allPanels_continues[sec].columns}
#     for  sec in Color_vs_Sgoly_allPanels_continues.keys():

#         for col in Color_vs_Sgoly_allPanels_continues[sec].columns:
#             ColMean[col].append(np.mean(Color_vs_Sgoly_allPanels_continues[sec][col]))
#             Colstd[col].append(np.std(Color_vs_Sgoly_allPanels_continues[sec][col]))
#             Col95p[col].append(np.percentile(Color_vs_Sgoly_allPanels_continues[sec][col], 95))
            
#     return ColMean,Colstd,Col95p

# def   CalcStatistics_perPanel(Color_vs_Sgoly_allPanels):
    
#     Pnl=list(Color_vs_Sgoly_allPanels.keys())[0]
#     # statisticalData={i: {col: [] for col in Color_vs_Sgoly_allPanels[Pnl][0].columns} for i in range(11)}
#     statisticalData_mean={i: {col: [] for col in Color_vs_Sgoly_allPanels[Pnl][0].columns} for i in range(11)}
#     statisticalData_std={i: {col: [] for col in Color_vs_Sgoly_allPanels[Pnl][0].columns} for i in range(11)}
#     statisticalData_95p={i: {col: [] for col in Color_vs_Sgoly_allPanels[Pnl][0].columns} for i in range(11)}

#     for Pnl in Color_vs_Sgoly_allPanels.keys():
        
#         pNumber=float(Pnl.split('-')[-1])%11 
        
#         ColMean={col: [] for col in Color_vs_Sgoly_allPanels[Pnl][0].columns}
#         Colstd={col: [] for col in Color_vs_Sgoly_allPanels[Pnl][0].columns}
#         Col95p={col: [] for col in Color_vs_Sgoly_allPanels[Pnl][0].columns}
#         for  sec in Color_vs_Sgoly_allPanels[Pnl].keys():
    
#             for col in Color_vs_Sgoly_allPanels[Pnl][sec].columns:
#                 ColMean[col].append(np.mean(abs(Color_vs_Sgoly_allPanels[Pnl][sec][col])))
#                 Colstd[col].append(np.std(abs(Color_vs_Sgoly_allPanels[Pnl][sec][col])))
#                 Col95p[col].append(np.percentile(abs(Color_vs_Sgoly_allPanels[Pnl][sec][col]), 95))
        
#         for col in ColMean.keys():
#             statisticalData_mean[pNumber][col].append(np.mean(ColMean[col]))
#             statisticalData_std[pNumber][col].append(np.mean(Colstd[col]))
#             statisticalData_95p[pNumber][col].append(np.mean(Col95p[col]))

#     return statisticalData_mean,statisticalData_std,statisticalData_95p



# def plotTable(statisticalData_mean,statisticalData_std,statisticalData_95p,TableTitle,FileName):
    
#     statisticalDataDF=pd.DataFrame()
#     tmp=pd.DataFrame()
#     for key in statisticalData_mean.keys():
#         for col in statisticalData_mean[key].keys():
#             tmp[col]=['Mean='+"{:.2f}".format(np.mean(statisticalData_mean[key][col])*pixSize)+' ,STD='+"{:.2f}".format(np.mean(statisticalData_std[key][col])*pixSize)+' ,95p='+"{:.2f}".format(np.mean(statisticalData_95p[key][col])*pixSize)]
#         statisticalDataDF= pd.concat([statisticalDataDF,tmp],axis=0)
#     statisticalDataDF=statisticalDataDF.reset_index(drop=True)
    
    
#     listForTable=[]
#     listForTable.append(list(statisticalDataDF.index+1))
#     for col in statisticalDataDF.columns:
#         listForTable.append(statisticalDataDF[col])
    
#     table = go.Figure(data=[go.Table(
#             header=dict(values=['Panel Number']+list(statisticalDataDF.columns),
#                         fill_color='paleturquoise',
#                         align='left'),
#             cells=dict(values=listForTable,
#                        fill_color='lavender',
#                        align='left'))
#         ])
    
#     fig = go.Figure()
#     fig.add_trace(table.data[0])
    
#     fig.update_layout(title='Summarize '+TableTitle)
#     plot(fig,auto_play=True,filename=FileName)  

    
    
    
############################################################################################
############################################################################################
############################################################################################
############################################################################################


def FFT_Plot(db,PlotTitle,fileName, Fs, PageSide,freqUnits,colorInUseName):






    N = len(db.index)
    
 
    p1=pd.DataFrame()
    T = 1/(Fs);
    x = np.linspace(0.0, N*T, N)
    y =pd.DataFrame();
    # Yfft=scipy.fftpack.fft(y)
    for col in colorInUseName:
        y[col]= np.fft.fft(db[col],axis=0)
        p1[col]=(1/(Fs*N))*pow(abs(y[col][:N//2]),2);
        
        p1[col][1:-1]=2*p1[col][1:-1];
  
    
    
    
    
    xf = np.linspace(0.0, Fs/2, int(N/2))
    
    
    fig = go.Figure()
    
    # rnge=[3,6,7]
    
    # db=ImagePlacement_Rightpp
    
    for col in p1.columns:
        
        try:
            lineColor=col.split('-')[1]
        except:
            lineColor=col
        
        fig.add_trace(
        
        go.Scatter(x=list(xf),y=list(10*np.log10(p1[col])),line_color=lineColor,name=col+' ' +PageSide))
    
    
    
    fig.update_layout( xaxis=dict(
            title='Freq '+freqUnits,
        ),
        yaxis=dict(
            title='db',
            # type='log'
        ))
    

    fig.update_layout(title=PlotTitle)
    
    # fig.show()        
    
    plot(fig,filename=fileName)  
    
    return fig


def sorted_indices(arr):
    # Enumerate the list to keep track of original indices
    enumerated_list = list(enumerate(arr))
    # Sort the enumerated list based on the values
    sorted_enumerated_list = sorted(enumerated_list, key=lambda x: x[1])
    # Extract and return the sorted indices
    sorted_indices_list = [index for index, _ in sorted_enumerated_list]
    return sorted_indices_list


def CalcCumError(clrFomTarget,LngthLimit):
    
   
    cumSmPnl=pd.DataFrame() 
    integral_value=pd.DataFrame() 
    
    for col in clrFomTarget.columns:
        y= clrFomTarget[col][:LngthLimit]*pixSize
        
        x=np.array(range(LngthLimit))*DistBtwCrcle*AQMscale*pixSize*1e-3;
        coefficients = np.polyfit(x, y, deg=1)
     
        # Get the slope and intercept of the linear line
        slope = coefficients[0]
        intercept = coefficients[1]
     
        # # Print the equation of the line
        # print(f"Equation of the line: y = {slope:.2f}x + {intercept:.2f}")
     
        # Predict y-values for the original x-values
        y_predicted = np.polyval(coefficients, x)
        cumSmPnl[col]=list(np.cumsum(y_predicted-y))
        
        x_vals=np.array(range(len(y_predicted-y)))
        y_vals=y_predicted-y
        intCalc=[]
        for j in range(len(y_vals)-1):
            intCalc.append(np.trapz(y_vals[:j+1], x_vals[:j+1]))
        integral_value[col] = intCalc


    return  x, cumSmPnl


def Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefCl,colorInUseName,SecNum):
    
    Color_Black_Sgoly={};

    for i in range(SecNum):
        Color_Black_Sgoly[i]=pd.DataFrame()
        for col in colorInUseName:
            if RefCl == col:
                continue;
            Color_Black_Sgoly[i] = pd.concat([Color_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)

    return Color_Black_Sgoly


def createContinues(Var_allPanels_continues,Var_allPanels,secRnge,PointsToIgnore):
    
    for i in range(secRnge):
        for pnl in Var_allPanels.keys():
            try:
                Var_allPanels_continues[i]=pd.concat([Var_allPanels_continues[i],Var_allPanels[pnl][i][:-PointsToIgnore]]).reset_index(drop=True) 
            except:
                continue;
                
    return Var_allPanels_continues;

def DeleteCommonTredLineFronData(ClrDF_allPanels):
    
    degree = 4
        
    for Pnl in  ClrDF_allPanels.keys():
        mean_value={}
        for i in ClrDF_allPanels[Pnl].keys():
            
            clr_FromTaget=ClrDF_allPanels[Pnl][i]
            y_fitList={}
            for clr in clr_FromTaget.columns:
                
                y_values= clr_FromTaget[clr][PointOfSpeedCange:].reset_index(drop=True)
                
                x_values = np.linspace(0, len(y_values),len(y_values))
                
                
                coefficients = np.polyfit(x_values, y_values,degree)
                
                # Create a polynomial function using the coefficients
                poly_function = np.poly1d(coefficients)
                
                
                # plt.figure(1)
                # plt.plot(y_values)
                # plt.plot(poly_function(x_values))
                
                y_values_0_PointOfSpeedCange= clr_FromTaget[clr][:PointOfSpeedCange].reset_index(drop=True)
                
                x_values_0_PointOfSpeedCange = np.linspace(0, len(y_values_0_PointOfSpeedCange),len(y_values_0_PointOfSpeedCange))
                
                
                coefficients_PointOfSpeedCange = np.polyfit(x_values_0_PointOfSpeedCange, y_values_0_PointOfSpeedCange,degree)
                
                # Create a polynomial function using the coefficients
                poly_function_PointOfSpeedCange = np.poly1d(coefficients_PointOfSpeedCange)            
                
                # Generate x values for the fitted curve
                
                # plt.figure(2)
                # plt.plot(list(y_values_0_223))
                # plt.plot(list(poly_function_223(x_values_0_223)))
                
                # Calculate corresponding y values based on the fitted polynomial
                y_fit= poly_function(x_values)
                y_fitPointOfSpeedCange= poly_function_PointOfSpeedCange(x_values_0_PointOfSpeedCange)
                
                       
                # plt.figure(1)
                # plt.plot(list(y_values_0_223)+list(y_values))
                # plt.plot(list(y_fit223)+list(y_fit))
                
                # y_fitList[clr]=[0] * 223+list(y_fit)
                y_fitList[clr]=list(y_fitPointOfSpeedCange)+list(y_fit)
            
             
            all_values = np.zeros(numberOfPoints)
            
            for value in y_fitList.values():
                all_values= all_values+ np.array(value)
                
            
            mean_value[i] = all_values / len(clr_FromTaget.columns)
            for clr in ClrDF_allPanels[Pnl][i].columns:
                ClrDF_allPanels[Pnl][i][clr] = ClrDF_allPanels[Pnl][i][clr] -mean_value[i]
        
            
    return ClrDF_allPanels
            
def find_change_index(list_of_lists):
    change_indices = []
    
    # Iterate through the list of lists starting from the second element
    for i in range(1, len(list_of_lists)):
        # Compare the current inner list with the previous one
        if list_of_lists[i] != list_of_lists[i - 1]:
            change_indices.append(i)
    
    return change_indices        


def mirror_signal_along_x_axis(signal,scaling_factor,offset_factor):
    
    modSignal=(signal-np.min(signal))/(np.max(signal)-np.min(signal))*scaling_factor+offset_factor
    
    mirrored_signal = [-value for value in modSignal]
    return mirrored_signal,modSignal

def check_and_store_pairs(lst):
    if len(lst) % 2 != 0:
        return None  # If the list has an odd number of elements, it cannot form pairs

    pairs_dict = {}
    pair_number = 1

    strtInx=[]
    endInx=[]
    for i in range(0, len(lst) - 1, 2):
        pair = [lst[i], lst[i + 1]]

        # Check the difference between consecutive pairs
        if abs(pair[0] - pair[1]) > 3:
            continue;  # If the difference is greater than 3, the condition is not met

        strtInx.append(lst[i])
        endInx.append(lst[i+1])
        pairs_dict[pair_number] = pair
        pair_number += 1

    return pairs_dict,strtInx,endInx  # Return the dictionary with pairs



def Find_panelsWithPoints(current_line_sortedDIC):
    
    for sec in current_line_sortedDIC.keys():
        for key in current_line_sortedDIC[sec]:
            yy = [center[1] for center in current_line_sortedDIC[sec][key]]
            mean=np.mean(yy)



class Operations:
    def __init__(self):
        
        self.ClrDF_fromTargetPnl={}
        self.C2Cmat_allPanels_continues=pd.DataFrame()
        self.C2Cmat_allPanels={}
        self.MaxMinColor_allPanels={}
        self.ClrDF_fromTargetS_goly_allPanels={}
        self.ClrDF_fromTarget_allPanels={}

        self.Color_Black_Sgoly_allPanels={}
        self.Green_Black_Sgoly_allPanels={};
        self.Green_Black_NO_FILTER_allPanels={};

        self.col1_col2_Sgoly_allPanels={};
        self.col2_col3_Sgoly_allPanels={};
        self.col3_col4_Sgoly_allPanels={};
        self.col4_col5_Sgoly_allPanels={};
        self.col5_col6_Sgoly_allPanels={};


        self.ClrDF_fromTargetS_goly_allPanels_continues={}
        self.ClrDF_fromTarget_allPanels_continues={}

        self.Color_Black_Sgoly_allPanels_continues={}
        self.Green_Black_Sgoly_allPanels_continues={};
        self.col1_col2_Sgoly_allPanels_continues={};
        self.col2_col3_Sgoly_allPanels_continues={};
        self.col3_col4_Sgoly_allPanels_continues={};
        self.col4_col5_Sgoly_allPanels_continues={};
        self.col5_col6_Sgoly_allPanels_continues={};


        for i in range(NumOfSec):
            self.ClrDF_fromTargetS_goly_allPanels_continues[i]=pd.DataFrame();
            self.ClrDF_fromTarget_allPanels_continues[i]=pd.DataFrame();
            self.Color_Black_Sgoly_allPanels_continues[i]=pd.DataFrame();
            self.Green_Black_Sgoly_allPanels_continues[i]=pd.DataFrame();
            self.col1_col2_Sgoly_allPanels_continues[i]=pd.DataFrame();
            self.col2_col3_Sgoly_allPanels_continues[i]=pd.DataFrame();
            self.col3_col4_Sgoly_allPanels_continues[i]=pd.DataFrame();
            self.col4_col5_Sgoly_allPanels_continues[i]=pd.DataFrame();
            self.col5_col6_Sgoly_allPanels_continues[i]=pd.DataFrame();

        self.indexPanelNameDic={}
        self.ClrDF_rawXY_allPanels={}
        self.yTargetDF_allpanel={}
        self.xTargetDF_allpanel={}
        self.calcTarget = True

        self.circle_image_pnl={}
        self.gray_pnl={}
        self.Target_image_pnl={}
    
    def ReadAllFlats(self,sInput):
        
        sInputList=[folder for folder in os.listdir(sInput) if os.path.isdir(os.path.join(sInput, folder)) and ('-' in folder)]
        pnl = sInputList[0]
        listLastDig=[int(ll.split('-')[2])   for ll in sInputList]
        sorted_indices_list = sorted_indices(listLastDig)
        self.sInputListSORTED=[]
        for ll in  sorted_indices_list:
            self.sInputListSORTED.append(sInputList[ll])
            
    
    def CalculateCenterOfMass_and_C2C(self,sInput,colorInUseName):
        continuesPoints=0
        fileNME='\\FullImage.bmp'
        for Pnl in self.sInputListSORTED:
            
            try:
                pth=sInput+'\\'+Pnl+fileNME
                print('Start calc center of mass: '+pth+' '+Pnl)

                if NumOfSec == 18:
                    self.ImRoi,StartRoiCoed,Roi= Circles(pth).loadImage();
                else:
                    self.ImRoi,StartRoiCoed,Roi= Circles(pth).loadImage_SmallSubstrate();
                
                

                print('Finish calc center of mass: '+Pnl)
                gray_all,circle_image_all,edges,ClrDF_raw, ClrDF_rawXY,current_line_sortedDIC,StartRoiCoed = Circles(pth).CalcorColorMat(self.ImRoi,StartRoiCoed)
                
                
                self.circle_image_pnl[Pnl]=circle_image_all
                self.gray_pnl[Pnl]=gray_all
                
                
                
                self.ClrDF_rawXY_allPanels[Pnl]=ClrDF_rawXY
                
                ClrDF_fromTargetS_goly={}
                ClrDF_fromTarget={}
                dymeanListdic={}
                x={}
                dymeanList=[]
                for i in range(len(self.ImRoi.keys())):
                  dymeanList=Circles(pth).CalcDiffTarget(ClrDF_raw[i],dymeanList)  
                
                ClrDF_fromTargetSide={}
            
                strartPos =0
                strartPos_x =np.mean(ClrDF_rawXY[0]['Magenta_x'])
                yTargetDF_all={}
                xTargetDF_all={}
                Target_image_pnl={}
            
                for i in range(len(self.ImRoi.keys())):
                    

                    ClrDF_fromTarget[i],ClrDF_fromTargetS_goly[i],dymeanListdic[i],yTargetDF,strartPos,strartPos_x,xTargetDF = Circles(pth).calcDiffernceFromeTargetXY(ClrDF_rawXY[i],dymeanList,colorInUseName,strartPos,strartPos_x,DistanceBetweenColumns[i]);
                    xTargetDF_all[i]=xTargetDF
                    yTargetDF_all[i]=yTargetDF
            
                    x[i]=np.mean(dymeanListdic[i])
                self.ClrDF_fromTargetS_goly_allPanels[Pnl]=ClrDF_fromTargetS_goly
                self.ClrDF_fromTarget_allPanels[Pnl]=ClrDF_fromTarget
                self.ClrDF_fromTargetPnl[Pnl]=ClrDF_fromTargetSide
                self.yTargetDF_allpanel[Pnl]=yTargetDF_all
                self.xTargetDF_allpanel[Pnl]=xTargetDF_all
                
                
                
                C2Cmat=pd.DataFrame()
                MaxMinColor=pd.DataFrame()
                for i in range(len(self.ImRoi.keys())):
                    C2Cmat[i],MaxMinColor[i] = Circles(sInput+'\\'+Pnl+fileNME).calcC2C('Magenta', self.ClrDF_fromTargetS_goly_allPanels[Pnl][i]);
                    
                # C2Cmat_allPanels= pd.concat([C2Cmat_allPanels, C2Cmat])  
                self.C2Cmat_allPanels[Pnl]=   C2Cmat
                self.MaxMinColor_allPanels[Pnl]= MaxMinColor
                RefC01='Cyan'
                self.Color_Black_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC01,colorInUseName,len(self.ImRoi.keys()))
                
                RefC02='Magenta'
                self.Green_Black_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC02,colorInUseName,len(self.ImRoi.keys()))
                self.Green_Black_NO_FILTER_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTarget,RefC02,colorInUseName,len(self.ImRoi.keys()))

                RefC03='Orange'
                if RefC03 in colorInUseName:
                    self.col1_col2_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC03,colorInUseName,len(self.ImRoi.keys()))
            
                RefC04='Yellow'
                self.col2_col3_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC04,colorInUseName,len(self.ImRoi.keys()))
                
                RefC05='Black'
                self.col3_col4_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC05,colorInUseName,len(self.ImRoi.keys()))
                
                RefC06='Blue'
                if RefC06 in colorInUseName:
                    self.col4_col5_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC06,colorInUseName,len(self.ImRoi.keys()))
            
                RefC07='Green'
                if RefC07 in colorInUseName:
                    self.col5_col6_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC07,colorInUseName,len(self.ImRoi.keys()))
            
               
                
                continuesPoints=continuesPoints+pointsForPanel
                self.C2Cmat_allPanels_continues= pd.concat([self.C2Cmat_allPanels_continues, C2Cmat[:-PointsToIgnore]])  

                self.indexPanelNameDic[continuesPoints] = Pnl
                
                print(Pnl)
                
            except:
                continue
            
            self.Save_pickle(sInput)

        
        
    def caculate_continues(self,colorInUseName):
        
        self.ClrDF_fromTarget_allPanels=DeleteCommonTredLineFronData(self.ClrDF_fromTarget_allPanels)


        self.ClrDF_fromTargetS_goly_allPanels=DeleteCommonTredLineFronData(self.ClrDF_fromTargetS_goly_allPanels)


        self.C2Cmat_allPanels_continues= self.C2Cmat_allPanels_continues.reset_index(drop=True);
            
        self.ClrDF_fromTargetS_goly_allPanels_continues = createContinues(self.ClrDF_fromTargetS_goly_allPanels_continues,self.ClrDF_fromTargetS_goly_allPanels,len(self.ImRoi.keys()),PointsToIgnore);

        self.ClrDF_fromTarget_allPanels_continues = createContinues(self.ClrDF_fromTarget_allPanels_continues,self.ClrDF_fromTarget_allPanels,len(self.ImRoi.keys()),PointsToIgnore);

        RefC01='Cyan'
        RefCl=RefC01
        if RefC01 in colorInUseName:
            self.Color_Black_Sgoly_allPanels_continues = createContinues(self.Color_Black_Sgoly_allPanels_continues,self.Color_Black_Sgoly_allPanels,len(self.ImRoi.keys()),PointsToIgnore);

        RefC02='Magenta'

        RefCl=RefC02
        if RefC02 in colorInUseName:
            self.Green_Black_Sgoly_allPanels_continues = createContinues(self.Green_Black_Sgoly_allPanels_continues,self.Green_Black_Sgoly_allPanels,len(self.ImRoi.keys()),PointsToIgnore);

        RefC03='Orange'
        if RefC03 in colorInUseName:
            self.col1_col2_Sgoly_allPanels_continues = createContinues(self.col1_col2_Sgoly_allPanels_continues,self.col1_col2_Sgoly_allPanels,len(self.ImRoi.keys()),PointsToIgnore);

        RefC05='Black'
        RefCl=RefC05

        if RefC05 in colorInUseName:
            self.col3_col4_Sgoly_allPanels_continues = createContinues(self.col3_col4_Sgoly_allPanels_continues,self.col3_col4_Sgoly_allPanels,len(self.ImRoi.keys()),PointsToIgnore);



        RefC04='Yellow'
        if RefC04 in colorInUseName:
            self.col2_col3_Sgoly_allPanels_continues = createContinues(self.col2_col3_Sgoly_allPanels_continues,self.col2_col3_Sgoly_allPanels,len(self.ImRoi.keys()),PointsToIgnore);
          
        RefC06='Blue'
        if RefC06 in colorInUseName:
            self.col4_col5_Sgoly_allPanels_continues = createContinues(self.col4_col5_Sgoly_allPanels_continues,self.col4_col5_Sgoly_allPanels,len(self.ImRoi.keys()),PointsToIgnore);
        RefC07='Green'
        if RefC07 in colorInUseName:
            self.col5_col6_Sgoly_allPanels_continues = createContinues(self.col5_col6_Sgoly_allPanels_continues,self.col5_col6_Sgoly_allPanels,len(self.ImRoi.keys()),PointsToIgnore);




        self.Side2Side=self.ClrDF_fromTargetS_goly_allPanels_continues[0]-self.ClrDF_fromTargetS_goly_allPanels_continues[len(self.ClrDF_fromTargetS_goly_allPanels_continues)-1]

    def Save_pickle(self,sInput):
          path_FnameClrDF=sInput+'\ClrDF_rawXY_allPanels.pkl'
          with open(path_FnameClrDF, 'wb') as f:
              pickle.dump(self.ClrDF_rawXY_allPanels, f)
        
        
            
            

        
        
            
        
        
##############################################################################################################
##############################################################################################################
##############################################################################################################


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataStructureUI()
    window.closed.connect(app.quit)
    window.show()
    sys.exit(app.exec_())

# from tkinter import filedialog
# from tkinter import *
# root = Tk()
# root.withdraw()
# sInput = filedialog.askdirectory()
# os.chdir(sInput)
# # os.chdir(r'D:\BTDencoder\B1')


# sInputList=[folder for folder in os.listdir(sInput) if os.path.isdir(os.path.join(sInput, folder)) and ('-' in folder)]


# pnl = sInputList[0]

# listLastDig=[int(ll.split('-')[2])   for ll in sInputList]


# sorted_indices_list = sorted_indices(listLastDig)

# sInputListSORTED=[]

# for ll in  sorted_indices_list:
#     sInputListSORTED.append(sInputList[ll])
    
# Pnl = sInputListSORTED[0]


# # colorInUseName=['Magenta','Yellow','Blue','Orange','Cyan','Green','Black']
# # colorInUseName=['Magenta','Yellow','Blue','Orange','Green','Black']


# # colorInUseName=['Magenta','Yellow','Cyan','Black']
# # colorInUseNum=[ColorDicNum[itm] for itm in colorInUseName]







# fileNME='\\FullImage.bmp'
# # ImRoi,StartRoiCoed= Circles(sInput+'\\'+pnl+fileNME).loadImage();

# ClrDF_fromTargetPnl={}
# C2Cmat_allPanels_continues=pd.DataFrame()
# C2Cmat_allPanels={}
# MaxMinColor_allPanels={}
# ClrDF_fromTargetS_goly_allPanels={}
# ClrDF_fromTarget_allPanels={}

# Color_Black_Sgoly_allPanels={}
# Green_Black_Sgoly_allPanels={};
# Green_Black_NO_FILTER_allPanels={};

# col1_col2_Sgoly_allPanels={};
# col2_col3_Sgoly_allPanels={};
# col3_col4_Sgoly_allPanels={};
# col4_col5_Sgoly_allPanels={};
# col5_col6_Sgoly_allPanels={};


# ClrDF_fromTargetS_goly_allPanels_continues={}
# ClrDF_fromTarget_allPanels_continues={}

# Color_Black_Sgoly_allPanels_continues={}
# Green_Black_Sgoly_allPanels_continues={};
# col1_col2_Sgoly_allPanels_continues={};
# col2_col3_Sgoly_allPanels_continues={};
# col3_col4_Sgoly_allPanels_continues={};
# col4_col5_Sgoly_allPanels_continues={};
# col5_col6_Sgoly_allPanels_continues={};


# for i in range(NumOfSec):
#     ClrDF_fromTargetS_goly_allPanels_continues[i]=pd.DataFrame();
#     ClrDF_fromTarget_allPanels_continues[i]=pd.DataFrame();
#     Color_Black_Sgoly_allPanels_continues[i]=pd.DataFrame();
#     Green_Black_Sgoly_allPanels_continues[i]=pd.DataFrame();
#     col1_col2_Sgoly_allPanels_continues[i]=pd.DataFrame();
#     col2_col3_Sgoly_allPanels_continues[i]=pd.DataFrame();
#     col3_col4_Sgoly_allPanels_continues[i]=pd.DataFrame();
#     col4_col5_Sgoly_allPanels_continues[i]=pd.DataFrame();
#     col5_col6_Sgoly_allPanels_continues[i]=pd.DataFrame();

# indexPanelNameDic={}




# # ImRoi= Circles(sInput+'\\'+pnl+fileNME).loadImage();
# # Clr1='Cyan'


       

# PointsToIgnore=40  
# continuesPoints=0
# pointsForPanel=numberOfPoints- PointsToIgnore
# Pnl=sInputListSORTED[0]

# # for Pnl in sInputListL:
# ClrDF_rawXY_allPanels={}
# yTargetDF_allpanel={}
# xTargetDF_allpanel={}
# calcTarget = True

# circle_image_pnl={}
# gray_pnl={}
# Target_image_pnl={}
# for Pnl in sInputListSORTED:
    
#     try:
#         pth=sInput+'\\'+Pnl+fileNME
    
#         if NumOfSec == 18:
#             ImRoi,StartRoiCoed,Roi= Circles(pth).loadImage();
#         else:
#             ImRoi,StartRoiCoed,Roi= Circles(pth).loadImage_SmallSubstrate();
        
        

#         # pth=r'D:\B8\new file\95-0-18\FullImage.bmp'
        
#         # ImRoi,StartRoiCoed= Circles(pth).loadImage();
#         # monitor_thread = TimeMonitorThread(Circles(pth).CalcorColorMat_forThreading,2,ImRoi,StartRoiCoed)  # Timeout set to 2 seconds
#         # monitor_thread.start()
#         # monitor_thread.join()  # Wait for the thread to finish
    
#         # if monitor_thread.exception:
#         #     raise monitor_thread.exception
#         # else:
#         #     Result = monitor_thread.result  # Unpacking the tuple into v1 and v2
#         #     # print("v1:", v1)
#         #     # print("v2:", v2)
    
#         gray_all,circle_image_all,edges,ClrDF_raw, ClrDF_rawXY,current_line_sortedDIC,StartRoiCoed = Circles(pth).CalcorColorMat(ImRoi,StartRoiCoed)
        
#         # gray_all=Result['gray_all']
#         # circle_image_all=Result['circle_image_all']
#         # edges=Result['edges']
#         # ClrDF_raw=Result['ClrDF_raw']
#         # ClrDF_rawXY=Result['ClrDF_rawXY']
#         # current_line_sortedDIC=Result['current_line_sortedDIC']
#         # StartRoiCoed=Result['StartRoiCoed']
        
#         # del Result
        
#         circle_image_pnl[Pnl]=circle_image_all
#         gray_pnl[Pnl]=gray_all
        
        
        
#         ClrDF_rawXY_allPanels[Pnl]=ClrDF_rawXY
#         # C2CmatOP_side = Circles(sInput).calcC2C('Magenta', ClrDF_raw[0]);
        
#         ClrDF_fromTargetS_goly={}
#         ClrDF_fromTarget={}
#         dymeanListdic={}
#         x={}
#         dymeanList=[]
#         for i in range(len(ImRoi.keys())):
#           dymeanList=Circles(pth).CalcDiffTarget(ClrDF_raw[i],dymeanList)  
        
#         ClrDF_fromTargetSide={}
    
#         strartPos =0
#         strartPos_x =np.mean(ClrDF_rawXY[0]['Magenta_x'])
#         yTargetDF_all={}
#         xTargetDF_all={}
#         Target_image_pnl={}
    
#         for i in range(len(ImRoi.keys())):
            
#             # dymeanList=[]
        
#             # dymeanList=Circles(sInput+'\\'+Pnl+fileNME).CalcDiffTarget(ClrDF_raw[i],dymeanList)
        
#             # ClrDF_fromTarget[i],ClrDF_fromTargetS_goly[i],dymeanListdic[i],yTargetDF,strartPos = Circles(pth).calcDiffernceFromeTarget(ClrDF_raw[i],dymeanList,colorInUseName,strartPos);
            
#             ClrDF_fromTarget[i],ClrDF_fromTargetS_goly[i],dymeanListdic[i],yTargetDF,strartPos,strartPos_x,xTargetDF = Circles(pth).calcDiffernceFromeTargetXY(ClrDF_rawXY[i],dymeanList,colorInUseName,strartPos,strartPos_x,DistanceBetweenColumns[i]);
#             xTargetDF_all[i]=xTargetDF
#             yTargetDF_all[i]=yTargetDF
    
#             x[i]=np.mean(dymeanListdic[i])
#         ClrDF_fromTargetS_goly_allPanels[Pnl]=ClrDF_fromTargetS_goly
#         ClrDF_fromTarget_allPanels[Pnl]=ClrDF_fromTarget
#         ClrDF_fromTargetPnl[Pnl]=ClrDF_fromTargetSide
#         yTargetDF_allpanel[Pnl]=yTargetDF_all
#         xTargetDF_allpanel[Pnl]=xTargetDF_all
        
        
        
#         C2Cmat=pd.DataFrame()
#         MaxMinColor=pd.DataFrame()
#         for i in range(len(ImRoi.keys())):
#             C2Cmat[i],MaxMinColor[i] = Circles(sInput+'\\'+Pnl+fileNME).calcC2C('Magenta', ClrDF_fromTargetS_goly_allPanels[Pnl][i]);
            
#         # C2Cmat_allPanels= pd.concat([C2Cmat_allPanels, C2Cmat])  
#         C2Cmat_allPanels[Pnl]=   C2Cmat
#         MaxMinColor_allPanels[Pnl]= MaxMinColor
#         RefC01='Cyan'
#         Color_Black_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC01,colorInUseName,len(ImRoi.keys()))
#         # # Cyan_Black_Sgoly={};
#         # # RefCl='Cyan'
#         # # for i in range(3):
#         # #     Cyan_Black_Sgoly[i]=pd.DataFrame()
#         # #     for col in ClrDF_fromTargetS_goly[i].columns:
#         # #         if RefCl == col:
#         # #             continue;
#         # #         Cyan_Black_Sgoly[i] = pd.concat([Cyan_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)
#         # #     Cyan_Black_Sgoly_allPanels[i]=pd.concat([Cyan_Black_Sgoly_allPanels[i], Cyan_Black_Sgoly[i]])
        
#         RefC02='Magenta'
#         Green_Black_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC02,colorInUseName,len(ImRoi.keys()))
#         Green_Black_NO_FILTER_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTarget,RefC02,colorInUseName,len(ImRoi.keys()))

#         RefC03='Orange'
#         if RefC03 in colorInUseName:
#             col1_col2_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC03,colorInUseName,len(ImRoi.keys()))
    
#         RefC04='Yellow'
#         col2_col3_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC04,colorInUseName,len(ImRoi.keys()))
        
#         RefC05='Black'
#         col3_col4_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC05,colorInUseName,len(ImRoi.keys()))
        
#         RefC06='Blue'
#         if RefC06 in colorInUseName:
#             col4_col5_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC06,colorInUseName,len(ImRoi.keys()))
    
#         RefC07='Green'
#         if RefC07 in colorInUseName:
#             col5_col6_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC07,colorInUseName,len(ImRoi.keys()))
    
#         # # RefCl='Magenta'
#         # RefCl11='Magenta'
    
#         # Color_Black_Sgoly_allPanels=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,Color_Black_Sgoly_allPanels,RefCl11,colorInUseName,len(ImRoi.keys()))
       
        
#         continuesPoints=continuesPoints+pointsForPanel
#         C2Cmat_allPanels_continues= pd.concat([C2Cmat_allPanels_continues, C2Cmat[:-PointsToIgnore]])  

#         indexPanelNameDic[continuesPoints] = Pnl
        
#         print(Pnl)
        
#     except:
#         continue

# ######################################################################################

# ######################################################################################    

# ClrDF_fromTarget_allPanels=DeleteCommonTredLineFronData(ClrDF_fromTarget_allPanels)


# ClrDF_fromTargetS_goly_allPanels=DeleteCommonTredLineFronData(ClrDF_fromTargetS_goly_allPanels)


# C2Cmat_allPanels_continues= C2Cmat_allPanels_continues.reset_index(drop=True);
    
# ClrDF_fromTargetS_goly_allPanels_continues = createContinues(ClrDF_fromTargetS_goly_allPanels_continues,ClrDF_fromTargetS_goly_allPanels,len(ImRoi.keys()),PointsToIgnore);

# ClrDF_fromTarget_allPanels_continues = createContinues(ClrDF_fromTarget_allPanels_continues,ClrDF_fromTarget_allPanels,len(ImRoi.keys()),PointsToIgnore);

# RefC01='Cyan'
# RefCl=RefC01
# if RefC01 in colorInUseName:
#     Color_Black_Sgoly_allPanels_continues = createContinues(Color_Black_Sgoly_allPanels_continues,Color_Black_Sgoly_allPanels,len(ImRoi.keys()),PointsToIgnore);

# RefC02='Magenta'

# RefCl=RefC02
# if RefC02 in colorInUseName:
#     Green_Black_Sgoly_allPanels_continues = createContinues(Green_Black_Sgoly_allPanels_continues,Green_Black_Sgoly_allPanels,len(ImRoi.keys()),PointsToIgnore);

# RefC03='Orange'
# if RefC03 in colorInUseName:
#     col1_col2_Sgoly_allPanels_continues = createContinues(col1_col2_Sgoly_allPanels_continues,col1_col2_Sgoly_allPanels,len(ImRoi.keys()),PointsToIgnore);

# RefC05='Black'
# RefCl=RefC05

# if RefC05 in colorInUseName:
#     col3_col4_Sgoly_allPanels_continues = createContinues(col3_col4_Sgoly_allPanels_continues,col3_col4_Sgoly_allPanels,len(ImRoi.keys()),PointsToIgnore);



# RefC04='Yellow'
# if RefC04 in colorInUseName:
#     col2_col3_Sgoly_allPanels_continues = createContinues(col2_col3_Sgoly_allPanels_continues,col2_col3_Sgoly_allPanels,len(ImRoi.keys()),PointsToIgnore);
  
# RefC06='Blue'
# if RefC06 in colorInUseName:
#     col4_col5_Sgoly_allPanels_continues = createContinues(col4_col5_Sgoly_allPanels_continues,col4_col5_Sgoly_allPanels,len(ImRoi.keys()),PointsToIgnore);
# RefC07='Green'
# if RefC07 in colorInUseName:
#     col5_col6_Sgoly_allPanels_continues = createContinues(col5_col6_Sgoly_allPanels_continues,col5_col6_Sgoly_allPanels,len(ImRoi.keys()),PointsToIgnore);




# Side2Side=ClrDF_fromTargetS_goly_allPanels_continues[0]-ClrDF_fromTargetS_goly_allPanels_continues[len(ClrDF_fromTargetS_goly_allPanels_continues)-1]


# ##################################################################

# db= Side2Side;
# # RefCl='Cyan'

# PlotTitle='Side_2_Side'

# fileName=PlotTitle+'.html'

# ##################################################################
# ymax=100

# fig = go.Figure()

# for c in db.columns:  
   
#     lineColor=c
        
#     if lineColor=='Yellow':
#         lineColor='gold';
        
#     fig.add_trace(go.Scatter(y=list(db[c]*pixSize), line=dict(color=lineColor, dash='solid')  ,name=c))
        

 
# fig= Plot_Panel_number(fig,ymax,indexPanelNameDic)
 

# titleColor=c.split('-')[0]
# if titleColor == 'Cyan':
#      titleColor = '#008B8B';
 
# if titleColor == 'Yellow':
#      titleColor = 'gold'; 
 
# fig.update_layout(title={
#       'text': PlotTitle,
#       'font': {'color': titleColor}
#   })
#   #fig_back.update_layout(title='ImagePlacement_Left-Back')
  
  
# fig.update_layout(
#       hoverlabel=dict(
#           namelength=-1
#       )
#   )
  
#   # datetime object containing current date and time
 
# plot(fig,auto_play=True,filename=fileName)  
#   # plot(fig)  
 
#   #plot(fig_back,filename="AQM-Back.html")  
# fig.show()


# ##################################################################

# db= C2Cmat_allPanels_continues;
# PlotTitle='C2C'
# fileName=PlotTitle+'.html'
# figC2C_multiPanel= PlotSingle_Basic_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100)
# ##########################################################################################
# ##########################################################################################
# ##########################################################################################
# ##########################################################################################
# ##########################################################################################
# ##########################################################################################
# db= Color_Black_Sgoly_allPanels_continues;
# RefC01='Cyan'
# RefCl=RefC01
# if RefC01 in colorInUseName:
#     PlotTitle='2 color diff - '+RefCl +' Vs color'
    
#     fileName=PlotTitle+'.html'
#     figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,sectionNumber2Show)

# ##########################################################################################
# db= Green_Black_Sgoly_allPanels_continues;
# RefC02='Magenta'

# RefCl=RefC02
# if RefC02 in colorInUseName:


#     PlotTitle='2 color diff -'+RefCl +' Vs color'
    
#     fileName=PlotTitle+'.html'
#     figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,sectionNumber2Show)
# ##########################################################################################
# ##########################################################################################
# db= col1_col2_Sgoly_allPanels_continues;
# RefC03='Orange'
# if RefC03 in colorInUseName:
#     RefCl=RefC03
    
#     PlotTitle='2 color diff - '+RefCl +' Vs color'
    
#     fileName=PlotTitle+'.html'
#     figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,sectionNumber2Show)

# ##########################################################################################
# db= col2_col3_Sgoly_allPanels_continues;

# RefC04='Yellow'
# RefCl=RefC04

# if RefC04 in colorInUseName:
#     PlotTitle='2 color diff -'+RefCl +' Vs color'
    
#     fileName=PlotTitle+'.html'
#     figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,sectionNumber2Show)


# ##########################################################################################
# db= col3_col4_Sgoly_allPanels_continues;
# RefC05='Black'
# RefCl=RefC05

# if RefC05 in colorInUseName:

#     PlotTitle='2 color diff - '+RefCl +' Vs color'
    
#     fileName=PlotTitle+'.html'
#     figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,sectionNumber2Show)

# ##########################################################################################
# db= col4_col5_Sgoly_allPanels_continues;

# RefC06='Blue'
# if RefC06 in colorInUseName:

#     RefCl=RefC06
    
    
#     PlotTitle='2 color diff -'+RefCl +' Vs color'
    
#     fileName=PlotTitle+'.html'
#     figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,sectionNumber2Show)


# ##########################################################################################
# db= col5_col6_Sgoly_allPanels_continues;

# RefC07='Green'
# if RefC07 in colorInUseName:

#     RefCl=RefC07
    
    
#     PlotTitle='2 color diff -'+RefCl +' Vs color'
    
#     fileName=PlotTitle+'.html'
#     figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,sectionNumber2Show)


# ##########################################################################################
# ##########################################################################################
# ##########################################################################################
# #############################################################################################
# ##########################################################################################

# db= ClrDF_fromTarget_allPanels_continues;
# PlotTitle='Single color from Target-NO FILTER'

# fileName=PlotTitle+'.html'
# figCyanVsClr_multiPanel_colorFromTarget= PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,sectionNumber2Show,0)
# ##########################################################################################
# ##########################################################################################
# #############################################################################################
# ##########################################################################################

# db= ClrDF_fromTargetS_goly_allPanels_continues;
# PlotTitle='Single color from Target'

# fileName=PlotTitle+'.html'
# figCyanVsClr_multiPanel_colorFromTarget= PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,sectionNumber2Show,figCyanVsClr_multiPanel_colorFromTarget)



# # #############################################################################################
# ##Table
# Color_vs_Sgoly_allPanels=Color_Black_Sgoly_allPanels
# statisticalData_mean,statisticalData_std,statisticalData_95p=CalcStatistics_perPanel(Color_vs_Sgoly_allPanels)

# TableTitle='cyan to other colors'
# FileName= 'SummarizeTable_regTempBTD.html'
# plotTable(statisticalData_mean,statisticalData_std,statisticalData_95p,TableTitle,FileName)



# ########################################################################################
# ########################################################################################
# ########################################################################################
# ############################-----------#################################################
# ############################--DEBUG----#################################################
# ###########################-----------##################################################
# ########################################################################################
# ########################################################################################
# ########################################################################################
# # Define the input string


# Color_Black_Sgoly_allPanelsDiverst_byPnum={}
# for i in range(1,12):
#     Color_Black_Sgoly_allPanelsDiverst_byPnum[i]=[]

# for flat in Color_Black_Sgoly_allPanels.keys():
#     flatNum = int(flat.split('-')[2])
#     panel= flatNum % 11
#     Color_Black_Sgoly_allPanelsDiverst_byPnum[panel+1].append(Color_Black_Sgoly_allPanels[flat])
    
# for  panel in Color_Black_Sgoly_allPanelsDiverst_byPnum.keys():   
#     db=Color_Black_Sgoly_allPanelsDiverst_byPnum[panel]   
#     PlotTitle='Cyan vs colors for panel '+str(panel)
#     fileName='Cyan vs colors for panel_'+str(panel)+'.html'
#     fig=PlotSingle_for_a_panel(db,PlotTitle,fileName,sectionNumber2Show,-PointsToIgnore)
    


# # clrToUse=colorInUseName
# # clrToUse=['Yellow','Magenta','Orange']
# # clrToUse=['Cyan','Black']


# # pnelToShow=sInputListSORTED[2:30]

# # NumOfSecToShow=NumOfSec


# # PlotRegistrationMapForPnls(ClrDF_fromTargetS_goly_allPanels,colorInUseName,clrToUseInJOB,pnelToShow,NumOfSecToShow,sInput,fileNME,scaling_factor,offset_factor)





# # path_FnameClrDF=sInput+'\ClrDF_fromTarget_allPanels.pkl'
# # with open(path_FnameClrDF, 'wb') as f:
# #     pickle.dump(ClrDF_fromTarget_allPanels, f)
    
    
# # path_FnameClrDF=sInput+'\ClrDF_rawXY_allPanels.pkl'
# # with open(path_FnameClrDF, 'wb') as f:
# #     pickle.dump(ClrDF_rawXY_allPanels, f)
    
# # path_FnameClrDF=sInput+'\ClrDF_fromTargetS_goly_allPanels.pkl'
# # with open(path_FnameClrDF, 'wb') as f:
# #     pickle.dump(ClrDF_fromTargetS_goly_allPanels, f)

 



# # Pnl='887-0-121'
# # sc=3
# # AreaTH=7


# # # gray=gray_pnl[Pnl][sc]


# # # gray=gray_all
# # # circle_image = np.ones_like(gray)*220
# # Cross_image = np.ones_like(gray)*220

# # # Iterate over the contours and find the center of each circle
# # for clr in yTargetDF.columns:
# #     for yCOORD,xCOORD in zip(ClrDF_rawXY_allPanels[Pnl][sc][clr],ClrDF_rawXY_allPanels[Pnl][sc][clr+'_x']):

# #         center_xINT = round(xCOORD-StartRoiCoed[sc])
# #         center_yINT = round(yCOORD)
        
 
# #         # Draw a circle at the center of the contour
# #         # cv2.circle(circle_image, (center_xINT, center_yINT), 5, (CircleLvl, CircleLvl, CircleLvl), -1)
# #         # Draw horizontal line (cross)
# #         cv2.line(Cross_image, (center_xINT-10, center_yINT), (center_xINT+10, center_yINT), (CircleLvl, CircleLvl, CircleLvl), 1)
        
# #         # Draw vertical line (cross)
# #         cv2.line(Cross_image, (center_xINT, center_yINT-10), (center_xINT, center_yINT+10),(CircleLvl, CircleLvl, CircleLvl), 1)

        

# # adjusted_image = np.where(gray_pnl[Pnl][sc] > 220, 220, np.where(gray_pnl[Pnl][sc] < 50, 70, gray_pnl[Pnl][sc]))

# # dIm=(Cross_image+gray_pnl[Pnl][sc])

# # dIm[dIm<0]=0

# # plt.figure(1)
# # plt.imshow(gray)



# # plt.figure(2)
# # plt.imshow(circle_image)


# # plt.figure(23)
# # plt.imshow(ImRoi[sc])

# # plt.figure(20)
# # plt.imshow(dIm)

# # clr='Yellow'
# # plt.figure(11)
# # plt.plot(np.diff(ClrDF_rawXY_allPanels[Pnl][sc][clr]))


# # ######################################################################################
# # clrInUse=['Cyan','Black','Magenta','Yellow']
# # # clrInUse=['Cyan','Black','Magenta']
# # # clrInUse=['Magenta','Yellow','Blue','Orange','Cyan','Green','Black']

# # # clrInUse=['Yellow','Orange','Magenta']

# # C2Cmat_allPanels={}
# # MaxMinColor_allPanels={}
# # for Pnl in sInputListSORTED: 
# #         C2Cmat=pd.DataFrame()
# #         MaxMinColor=pd.DataFrame()
# #         for i in range(len(ImRoi.keys())):
# #             C2Cmat[i],MaxMinColor[i] = Circles(sInput+'\\'+Pnl+fileNME).calcC2C_v1('Magenta', ClrDF_fromTargetS_goly_allPanels[Pnl][i][clrInUse]);
# #             # C2Cmat[i],MaxMinColor[i] = Circles(sInput+'\\'+Pnl+fileNME).calcC2C('Magenta', ClrDF_fromTargetS_goly_allPanels[Pnl][i][clrInUse]);
            
# #         # C2Cmat_allPanels= pd.concat([C2Cmat_allPanels, C2Cmat])  
# #         C2Cmat_allPanels[Pnl]=   C2Cmat
# #         MaxMinColor_allPanels[Pnl]= MaxMinColor




# # for Pnl in ClrDF_fromTarget_allPanels.keys():
# #     plt.figure(Pnl)
    
# #     # for sec in range(1,18,3):
# #     for sec in range(NumOfSec):
    
# #         l=list(MaxMinColor_allPanels[Pnl][sec])
# #         result = find_change_index(l)
# #         ll=list(C2Cmat_allPanels[Pnl][sec])
# #         mirrored_result = mirror_signal_along_x_axis(ll)
# #         secPos=np.mean(ClrDF_rawXY_allPanels[Pnl][sec]['Magenta_x'])
# #         # mirrored_result=list(C2Cmat_allPanels[Pnl][0]-1)
# #         ll1=list(np.asarray(ll)+secPos)
# #         mirrored_result1=list(np.asarray(mirrored_result)+secPos)
    
# #         rnge=list(range(len(ll)))
# #         i=0
# #         clr1=l[i][0]
# #         clr2=l[i][1]
# #         for res in result:
# #             clr1=l[i][0]
# #             clr2=l[i][1]
# #             if clr1 == 'Yellow':
# #                 clr1='Gold'
# #             if clr2 == 'Yellow':
# #                 clr2='Gold'    
# #             # plt.plot(rnge[i:res],ll[i:res],color=clr1,linewidth=2)
# #             # plt.plot(rnge[i:res],mirrored_result[i:res],color=clr2,linewidth=2)
            
# #             plt.plot(ll1[i:res],rnge[i:res],color=clr1,linewidth=2)
# #             plt.plot(mirrored_result1[i:res],rnge[i:res],color=clr2,linewidth=2)
# #             i=res-1
        
# #         plt.plot(ll1[i:],rnge[i:],color=clr1,linewidth=2)
# #         plt.plot(mirrored_result1[i:],rnge[i:],color=clr2,linewidth=2)
# # ####################################################################################
# # ####################################################################################
# # ####################################################################################
           
# # ClrDF_raw={}
# # ClrDF_rawXY={}

# # current_line_sortedDIC={}


# # circle_image_all={}
# # gray_all={}

# # for i in range(len(ImRoi.keys())):
    
# #     circle_centers,gray,circle_image,edges=Circles(pth).find_circles(ImRoi[i],CircleArea);
    
    
# #     circle_image_all[i]=circle_image
    
# #     gray_all[i]=gray
    
# #     current_line_sorted=Circles(pth).SortCircle_Coord(circle_centers);
    
# #     if len(current_line_sorted.keys()) < 7:
# #         current_line_sorted=Circles(pth).SortCircle_Coord_y(circle_centers)

    
# #     if len(current_line_sorted.keys()) > 7:
# #           keysOverLimit=[key for key in  current_line_sorted.keys() if key>6]
# #           for key in keysOverLimit:
# #               del current_line_sorted[key]           
# #     # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
    
    
# #     current_line_sorted,cntMissingCircles=Circles(pth).AddMissingCircles_withPolyFIT(current_line_sorted);

    
     
# #     current_line_sorted = Circles(pth).filterCircles_WithPol(current_line_sorted)
    
# #     current_line_sorted,cntMissingCircles=Circles(pth).AddMissingCircles_withPolyFIT(current_line_sorted);
    
# #     current_line_sorted=Circles(pth).Find_MIssDetected_Circles(current_line_sorted);

# #     current_line_sorted = Circles(pth).filterCircles_WithPol(current_line_sorted)


    
# #     # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
    
    
    
    
# #     # if len(current_line_sorted.keys()) > 7:
# #     #     del current_line_sorted[7]
    
# #     # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
    
# #     # current_line_sorted = self.filterCircles(current_line_sorted)
    
# #     # current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);

    
# #     ClrDF=pd.DataFrame()
# #     ClrDF_x=pd.DataFrame()

# #     tmp_df=pd.DataFrame()
# #     for Key,Value in ColorDic.items():
# #         tmp_df = pd.DataFrame({Value: Circles(pth).CreatColorList(Key,current_line_sorted)})
# #         ClrDF = pd.concat([ClrDF, tmp_df], axis=1)
# #         ClrDF_x = pd.concat([ClrDF_x, tmp_df], axis=1)
# #         tmp_df_x = pd.DataFrame({Value+'_x': Circles(pth).CreatColorList_x(Key,current_line_sorted,StartRoiCoed[i])})
# #         ClrDF_x = pd.concat([ClrDF_x, tmp_df_x], axis=1)

        
        
    
     
# #     current_line_sortedDIC[i] = current_line_sorted
# #     ClrDF_raw[i]=ClrDF
# #     ClrDF_rawXY[i]=ClrDF_x

# # result={}
# # result['gray_all']=gray_all
# # result['circle_image_all']=circle_image_all
# # result['edges']=edges
# # result['ClrDF_raw']=ClrDF_raw
# # result['ClrDF_rawXY']=ClrDF_rawXY
# # result['current_line_sortedDIC']=current_line_sortedDIC
# # result['StartRoiCoed']=StartRoiCoed

# ###################################################################################
# ###################################################################################
