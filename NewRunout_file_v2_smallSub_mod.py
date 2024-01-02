# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:50:54 2023

@author: Ireneg
"""

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


# import plotly.io as pio
# pio.renderers
# pio.renderers.default='browser'

# %matplotlib
############################################################################
global ColorDic,pixSize,MaxWaveWindow,sideDic,CircleArea,DistBtwCrcle,UseTarget,DistanceBetweenColumns,NumOfSec,CircleLvl

SatndardDistanceBetweenColumns= 406.5 #404.9626381019503;
LargeDistanceBetweenColumns= 1431.057469809592;
UseTarget =1 # False- uses the average distance between circles in pixels, True- uses DistBtwCrcle*AQMscale = (18 x 0.9832)

DistBtwCrcle=18
ScaleY=1#0.9965001812608202
AQMscale=0.9832
CircleArea=15
MaxWaveWindow=11
S_g_Degree=1
ColorDic={0:'Magenta',1:'Yellow',2:'Blue',3:'Orange',4:'Cyan',5:'Green',6:'Black'}
pixSize = 84.6666 # [um]
sideDic={0:'Left Side',1:'Middle',2:'Right Side'}

ColorDicNum={'Magenta':0,'Yellow':1,'Blue':2,'Orange':3,'Cyan':4,'Green':5,'Black':6}

numberOfPoints=439#288 #439

# DistanceBetweenColumns={i:SatndardDistanceBetweenColumns*1 for i in range(2,17)}
# DistanceBetweenColumns[0]=0

# DistanceBetweenColumns[1]=LargeDistanceBetweenColumns*1
# DistanceBetweenColumns[17]=LargeDistanceBetweenColumns*1

DistanceBetweenColumns={i:SatndardDistanceBetweenColumns*1 for i in range(1,16)}
DistanceBetweenColumns[0]=0

# DistanceBetweenColumns[1]=LargeDistanceBetweenColumns*1
# DistanceBetweenColumns[17]=LargeDistanceBetweenColumns*1


NumOfSec=16
CircleLvl=70

x_Cut_coord=[550,8220]

############################################################################



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
   
        return ImRoi,StartRoiCoed




    
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
   
        return ImRoi,StartRoiCoed
    
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
            
            if len(current_line_sorted[k])>numberOfPoints:
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
            indices = np.argwhere(abs(np.diff(np.array(y_values)))<(dymean/2))
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
        for i in ClrDF_rawSide.index:
            for col in ClrDF_rawSide.columns:
                if col == refClr:
                    continue;
                l.append(ClrDF_rawSide[refClr][i]-ClrDF_rawSide[col][i])
                
            C2Cmat.append((max(l)-min(l))*pixSize)
            l=[]

        return C2Cmat
    
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


def PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,ymax,sectionNumber):
    
   fig = go.Figure()
   

   
   for i in sectionNumber:
       for c in db[i].columns:  
           lineColor = c
                
           if lineColor=='Yellow':
                lineColor='gold';
                
           fig.add_trace(go.Scatter(y=list(db[i][c]*pixSize),line=dict(color=lineColor) , name=c+' section '+str(i)))
       
   
    
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


def Plot_Panel_number(fig,ymax,indexPanelNameDic):
    
    for key, value in indexPanelNameDic.items():
         fig.add_trace(go.Scatter(x=[key], y=[ymax],
                                 marker=dict(color="green", size=10),
                                 mode="markers",
                                 text=value,
                                 # font_size=18,
                                 hoverinfo='text'))
         
         fig.data[len(fig.data)-1].showlegend = False
         fig.add_vline(x=key, line_width=0.5, line_dash="dash", line_color="green")
 
    return fig





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



##############################################################################################################
##############################################################################################################
##############################################################################################################
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
sInput = filedialog.askdirectory()
os.chdir(sInput)
# os.chdir(r'D:\BTDencoder\B1')


sInputList=[folder for folder in os.listdir(sInput) if os.path.isdir(os.path.join(sInput, folder)) and ('-' in folder)]


pnl = sInputList[0]

listLastDig=[int(ll.split('-')[2])   for ll in sInputList]


sorted_indices_list = sorted_indices(listLastDig)

sInputListSORTED=[]

for ll in  sorted_indices_list:
    sInputListSORTED.append(sInputList[ll])
    
Pnl = sInputListSORTED[0]


# colorInUseName=['Magenta','Yellow','Blue','Orange','Cyan','Green','Black']


colorInUseName=['Magenta','Yellow','Cyan','Black']
colorInUseNum=[ColorDicNum[itm] for itm in colorInUseName]

# sInputListL=[sInputList[0]]

# sInput = r'D:\MainProj\Undefined[22][29-06-2023 16-48-59]'

# pnl = '\\394-0-3'

# imInp_Orig = cv2.imread(r'E:\s8\for irene\12-0-11\\FullImage.bmp')

# plt.figure()
# plt.imshow(imInp_Orig)

# sInputList=['394-0-1','394-0-2','394-0-3','394-0-4','394-0-5','394-0-6']
# imInp_Orig = cv2.imread(sInput+'\\'+Pnl+fileNME)


# plt.figure()
# plt.imshow(imInp_Orig)


# imInp_Orig = cv2.imread(Circles(r'D:\B8\new file\95-0-18\FullImage.bmp').pthF)

# plt.figure()
# plt.imshow(imInp_Orig)

# ImRoi= Circles(r'D:\B8\new file\95-0-16\FullImage.bmp').loadImage();





fileNME='\\FullImage.bmp'
# ImRoi,StartRoiCoed= Circles(sInput+'\\'+pnl+fileNME).loadImage();

ClrDF_fromTargetPnl={}
C2Cmat_allPanels=pd.DataFrame()
ClrDF_fromTargetS_goly_allPanels={}
ClrDF_fromTarget_allPanels={}

Color_Black_Sgoly_allPanels={}
Green_Black_Sgoly_allPanels={};
col1_col2_Sgoly_allPanels={};
col2_col3_Sgoly_allPanels={};
col3_col4_Sgoly_allPanels={};
col4_col5_Sgoly_allPanels={};
col5_col6_Sgoly_allPanels={};


ClrDF_fromTargetS_goly_allPanels_continues={}
ClrDF_fromTarget_allPanels_continues={}

Color_Black_Sgoly_allPanels_continues={}
Green_Black_Sgoly_allPanels_continues={};
col1_col2_Sgoly_allPanels_continues={};
col2_col3_Sgoly_allPanels_continues={};
col3_col4_Sgoly_allPanels_continues={};
col4_col5_Sgoly_allPanels_continues={};
col5_col6_Sgoly_allPanels_continues={};

for i in range(NumOfSec):
    ClrDF_fromTargetS_goly_allPanels_continues[i]=pd.DataFrame();
    ClrDF_fromTarget_allPanels_continues[i]=pd.DataFrame();
    Color_Black_Sgoly_allPanels_continues[i]=pd.DataFrame();
    Green_Black_Sgoly_allPanels_continues[i]=pd.DataFrame();
    col1_col2_Sgoly_allPanels_continues[i]=pd.DataFrame();
    col2_col3_Sgoly_allPanels_continues[i]=pd.DataFrame();
    col3_col4_Sgoly_allPanels_continues[i]=pd.DataFrame();
    col4_col5_Sgoly_allPanels_continues[i]=pd.DataFrame();
    col5_col6_Sgoly_allPanels_continues[i]=pd.DataFrame();

indexPanelNameDic={}




# ImRoi= Circles(sInput+'\\'+pnl+fileNME).loadImage();
Clr1='Cyan'


       
# ImgClr,gray,circle_image,edges,ClrDF_raw = Circles(sInput).CalcIntegralError(ImRoi, Clr1)
# dymeanList=[]
# for Pnl in sInputListSORTED:

#     ImRoi= Circles(sInput+'\\'+Pnl+fileNME).loadImage();
    
#     gray,circle_image,edges,ClrDF_raw = Circles(sInput+'\\'+Pnl+fileNME).CalcorColorMat(ImRoi)
    
    
#     # C2CmatOP_side = Circles(sInput).calcC2C('Magenta', ClrDF_raw[0]);
    
#     ClrDF_fromTargetS_goly={}
#     ClrDF_fromTarget={}
#     dymeanListdic={}
#     x={}
#     # dymeanList=[]
#     for i in range(3):
#       dymeanList=Circles(sInput+'\\'+Pnl+fileNME).CalcDiffTarget(ClrDF_raw[i],dymeanList)  

# plt.figure('ImRoi')
# plt.imshow(ImRoi)


# for Pnl in sInputListL:
ClrDF_rawXY_allPanels={}
yTargetDF_allpanel={}
xTargetDF_allpanel={}
calcTarget = True

circle_image_pnl={}
gray_pnl={}
Target_image_pnl={}
for Pnl in sInputListSORTED:
    
    try:
        pth=sInput+'\\'+Pnl+fileNME
    
        if NumOfSec == 18:
            ImRoi,StartRoiCoed= Circles(pth).loadImage();
        else:
            ImRoi,StartRoiCoed= Circles(pth).loadImage_SmallSubstrate();
        


        # pth=r'D:\B8\new file\95-0-18\FullImage.bmp'
        
        # ImRoi,StartRoiCoed= Circles(pth).loadImage();
    
        gray_all,circle_image_all,edges,ClrDF_raw, ClrDF_rawXY,current_line_sortedDIC,StartRoiCoed = Circles(pth).CalcorColorMat(ImRoi,StartRoiCoed)
        
        circle_image_pnl[Pnl]=circle_image_all
        gray_pnl[Pnl]=gray_all
        
        
        
        ClrDF_rawXY_allPanels[Pnl]=ClrDF_rawXY
        # C2CmatOP_side = Circles(sInput).calcC2C('Magenta', ClrDF_raw[0]);
        
        ClrDF_fromTargetS_goly={}
        ClrDF_fromTarget={}
        dymeanListdic={}
        x={}
        dymeanList=[]
        for i in range(len(ImRoi.keys())):
          dymeanList=Circles(pth).CalcDiffTarget(ClrDF_raw[i],dymeanList)  
        
        ClrDF_fromTargetSide={}
    
        strartPos =0
        strartPos_x =np.mean(ClrDF_rawXY[0]['Magenta_x'])
        yTargetDF_all={}
        xTargetDF_all={}
        Target_image_pnl={}
    
        for i in range(len(ImRoi.keys())):
            
            # dymeanList=[]
        
            # dymeanList=Circles(sInput+'\\'+Pnl+fileNME).CalcDiffTarget(ClrDF_raw[i],dymeanList)
        
            # ClrDF_fromTarget[i],ClrDF_fromTargetS_goly[i],dymeanListdic[i],yTargetDF,strartPos = Circles(pth).calcDiffernceFromeTarget(ClrDF_raw[i],dymeanList,colorInUseName,strartPos);
            
            ClrDF_fromTarget[i],ClrDF_fromTargetS_goly[i],dymeanListdic[i],yTargetDF,strartPos,strartPos_x,xTargetDF = Circles(pth).calcDiffernceFromeTargetXY(ClrDF_rawXY[i],dymeanList,colorInUseName,strartPos,strartPos_x,DistanceBetweenColumns[i]);
            xTargetDF_all[i]=xTargetDF
            yTargetDF_all[i]=yTargetDF
    
            x[i]=np.mean(dymeanListdic[i])
        ClrDF_fromTargetS_goly_allPanels[Pnl]=ClrDF_fromTargetS_goly
        ClrDF_fromTarget_allPanels[Pnl]=ClrDF_fromTarget
        ClrDF_fromTargetPnl[Pnl]=ClrDF_fromTargetSide
        yTargetDF_allpanel[Pnl]=yTargetDF_all
        xTargetDF_allpanel[Pnl]=xTargetDF_all
        
        
        
        C2Cmat=pd.DataFrame()
        for i in range(len(ImRoi.keys())):
            C2Cmat[i] = Circles(sInput+'\\'+Pnl+fileNME).calcC2C('Magenta', ClrDF_fromTargetS_goly_allPanels[Pnl][i]);
            
        C2Cmat_allPanels= pd.concat([C2Cmat_allPanels, C2Cmat])  
        
        RefC01='Cyan'
        Color_Black_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC01,colorInUseName,len(ImRoi.keys()))
        # # Cyan_Black_Sgoly={};
        # # RefCl='Cyan'
        # # for i in range(3):
        # #     Cyan_Black_Sgoly[i]=pd.DataFrame()
        # #     for col in ClrDF_fromTargetS_goly[i].columns:
        # #         if RefCl == col:
        # #             continue;
        # #         Cyan_Black_Sgoly[i] = pd.concat([Cyan_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)
        # #     Cyan_Black_Sgoly_allPanels[i]=pd.concat([Cyan_Black_Sgoly_allPanels[i], Cyan_Black_Sgoly[i]])
        
        RefC02='Magenta'
        Green_Black_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC02,colorInUseName,len(ImRoi.keys()))
        
        
        
        # RefC03='Orange'
        # col1_col2_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC03,colorInUseName,len(ImRoi.keys()))
    
        RefC04='Yellow'
        col2_col3_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC04,colorInUseName,len(ImRoi.keys()))
        
        RefC05='Black'
        col3_col4_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC05,colorInUseName,len(ImRoi.keys()))
        
        # RefC06='Blue'
        # col4_col5_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC06,colorInUseName,len(ImRoi.keys()))
    
        # RefC07='Green'
        # col5_col6_Sgoly_allPanels[Pnl]=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,RefC07,colorInUseName,len(ImRoi.keys()))
    
        # # RefCl='Magenta'
        # RefCl11='Magenta'
    
        # Color_Black_Sgoly_allPanels=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,Color_Black_Sgoly_allPanels,RefCl11,colorInUseName,len(ImRoi.keys()))
       
        
        
        
        indexPanelNameDic[len(C2Cmat_allPanels.iloc[:, 0])-1] = Pnl
        
        print(Pnl)
        
    except:
        continue

#######################################################################################

key= list(ClrDF_rawXY_allPanels.keys())[0]
cols=ClrDF_rawXY_allPanels[key][0].columns
colList=[]

for i in range(len(ClrDF_rawXY_allPanels[key][0][cols[0]])):
    colList.append('x'+str(i))
    colList.append('y'+str(i))

colList_full=['Flat','Section','Color']+colList
df = pd.DataFrame(columns=colList_full)

for key in ClrDF_rawXY_allPanels.keys():
    
    for sc in ClrDF_rawXY_allPanels[key].keys():
        ll=[]
        for color in colorInUseName:
            l=[]
            for i in range(len(ClrDF_rawXY_allPanels[key][sc][color])):
                l.append(ClrDF_rawXY_allPanels[key][sc][color][i])
                l.append(ClrDF_rawXY_allPanels[key][sc][color+'_x'][i])

            
            ll=[key,sc+1,color]+l
            new_row_df = pd.DataFrame([ll], columns=colList_full)
    
            # Concatenate the existing DataFrame with the new row DataFrame
            df = pd.concat([df, new_row_df], ignore_index=True)
    
    
    
df.to_csv('DigitalRunOut.csv',  index=False)

######################################################################################    
   
C2Cmat_allPanels= C2Cmat_allPanels.reset_index(drop=True)    

for i in range(len(ImRoi.keys())):
    for pnl in ClrDF_fromTargetS_goly_allPanels.keys():
        ClrDF_fromTargetS_goly_allPanels_continues[i]=pd.concat([ClrDF_fromTargetS_goly_allPanels_continues[i],ClrDF_fromTargetS_goly_allPanels[pnl][i]]).reset_index(drop=True) 
        ClrDF_fromTarget_allPanels_continues[i]=pd.concat([ClrDF_fromTarget_allPanels_continues[i],ClrDF_fromTarget_allPanels[pnl][i]]).reset_index(drop=True) 
        Color_Black_Sgoly_allPanels_continues[i]=pd.concat([Color_Black_Sgoly_allPanels_continues[i],Color_Black_Sgoly_allPanels[pnl][i]]).reset_index(drop=True) 
        Green_Black_Sgoly_allPanels_continues[i]=pd.concat([Green_Black_Sgoly_allPanels_continues[i],Green_Black_Sgoly_allPanels[pnl][i]]).reset_index(drop=True) 

        # col1_col2_Sgoly_allPanels_continues[i]=pd.concat([col1_col2_Sgoly_allPanels_continues[i],col1_col2_Sgoly_allPanels[pnl][i]]).reset_index(drop=True) 
        col2_col3_Sgoly_allPanels_continues[i]=pd.concat([col2_col3_Sgoly_allPanels_continues[i],col2_col3_Sgoly_allPanels[pnl][i]]).reset_index(drop=True) 
        col3_col4_Sgoly_allPanels_continues[i]=pd.concat([col3_col4_Sgoly_allPanels_continues[i],col3_col4_Sgoly_allPanels[pnl][i]]).reset_index(drop=True) 
        # col4_col5_Sgoly_allPanels_continues[i]=pd.concat([col4_col5_Sgoly_allPanels_continues[i],col4_col5_Sgoly_allPanels[pnl][i]]).reset_index(drop=True) 
        # col5_col6_Sgoly_allPanels_continues[i]=pd.concat([col5_col6_Sgoly_allPanels_continues[i],col5_col6_Sgoly_allPanels[pnl][i]]).reset_index(drop=True) 



        # Color_Black_Sgoly_allPanels[i]=Color_Black_Sgoly_allPanels[i].reset_index(drop=True)
        # Cyan_Black_Sgoly_allPanels[i]=Cyan_Black_Sgoly_allPanels[i].reset_index(drop=True) 
        # ClrDF_fromTargetS_goly_allPanels[i]=ClrDF_fromTargetS_goly_allPanels[i].reset_index(drop=True) 
#### Create Images
targetImage_Pnl={}
ClrDF_rawXY_image_Pnl={}
colorList_y=[col for col in ClrDF_rawXY[0].columns if not('_x' in col)]
colorList_x=[col for col in ClrDF_rawXY[0].columns if ('_x' in col)]

for Pnl in  sInputListSORTED:
    
    targetImage_all={}
    ClrDF_rawXY_all={}
    for i in range(len(yTargetDF_allpanel[Pnl])):
        
        targetImage_all[i]=Circles(pth).CreateCicles(xTargetDF_allpanel[Pnl][i]-StartRoiCoed[i],yTargetDF_allpanel[Pnl][i],gray_pnl[Pnl][i])
        ClrDF_rawXY_all[i]=Circles(pth).CreateCicles(ClrDF_rawXY_allPanels[Pnl][i][colorList_x]-StartRoiCoed[i],ClrDF_rawXY_allPanels[Pnl][i][colorList_y],gray_pnl[Pnl][i])

        
    targetImage_Pnl[Pnl]=targetImage_all
    ClrDF_rawXY_image_Pnl[Pnl]=ClrDF_rawXY_all
    print('image-'+Pnl)

plt.figure(0)
plt.imshow( ClrDF_rawXY_image_Pnl[Pnl][sc])


plt.figure(1)
plt.imshow( gray_pnl[Pnl][sc])


# plt.figure(2)
# plt.imshow( targetImage_Pnl[Pnl][0]-ClrDF_rawXY_image_Pnl[Pnl][0])


# plt.figure(3)
# plt.imshow( ClrDF_rawXY_image_Pnl[Pnl][0]-circle_image_pnl[Pnl][0])


Pnl='74-0-33'
sc=0

adjusted_image = np.where(gray_pnl[Pnl][sc] > 220, 220, np.where(gray_pnl[Pnl][sc] < 50, 70, gray_pnl[Pnl][sc]))

dIm=(circle_image_pnl[Pnl][sc].astype(np.int32)-adjusted_image.astype(np.int32))

dIm2=(ClrDF_rawXY_image_Pnl[Pnl][sc].astype(np.int32)-adjusted_image.astype(np.int32))

dIm3=(targetImage_Pnl[Pnl][sc].astype(np.int32)-adjusted_image.astype(np.int32))

dIm4=(targetImage_Pnl[Pnl][sc].astype(np.int32)-ClrDF_rawXY_image_Pnl[Pnl][sc].astype(np.int32))


dIm[dIm<0]=0
dIm2[dIm2<0]=0
dIm3[dIm3<0]=0
dIm4[dIm4<0]=0

# #### Check the Circles

# # dIm= adjusted_image-circle_image

# plt.figure(0)
# plt.plot(ClrDF_fromTarget_allPanels[Pnl][6]['Yellow'])

# plt.figure(0)
# plt.plot(np.diff(ClrDF_rawXY_allPanels[Pnl][sc]['Magenta']))
# ## maxdImInt=np.max(dImInt)
# ## mindImInt=np.min(dImInt)

# ## nominaldImInt=(dImInt-mindImInt)/(maxdImInt-mindImInt)*255



plt.figure(10)
plt.imshow(gray_pnl[Pnl][sc] )

plt.figure(101)
plt.imshow(ClrDF_rawXY_allPanels[Pnl][sc] )

plt.figure(1)
plt.imshow(dIm)

plt.figure(11)
plt.imshow(dIm2)

plt.figure(111)
plt.imshow(dIm3)

plt.figure(110)
plt.imshow(dIm4)

# plt.figure(2)
# plt.imshow(adjusted_image)

plt.figure(31)
plt.imshow(targetImage_Pnl[Pnl][sc])

plt.figure(3)
plt.imshow(ClrDF_rawXY_image_Pnl[Pnl][sc])



path_FnameClrDF=sInput+'\ClrDF_fromTarget_allPanels.pkl'
with open(path_FnameClrDF, 'wb') as f:
    pickle.dump(ClrDF_fromTarget_allPanels, f)
    
    
path_FnameClrDF=sInput+'\ClrDF_rawXY_allPanels.pkl'
with open(path_FnameClrDF, 'wb') as f:
    pickle.dump(ClrDF_rawXY_allPanels, f)
    
path_FnameClrDF=sInput+'\ClrDF_fromTargetS_goly_allPanels.pkl'
with open(path_FnameClrDF, 'wb') as f:
    pickle.dump(ClrDF_fromTargetS_goly_allPanels, f)

path_FnameClrDF=sInput+'\\targetImage_Pnl.pkl'
with open(path_FnameClrDF, 'wb') as f:
    pickle.dump(targetImage_Pnl, f)
    
path_FnameClrDF=sInput+'\\ClrDF_rawXY_image_Pnl.pkl'
with open(path_FnameClrDF, 'wb') as f:
    pickle.dump(ClrDF_rawXY_image_Pnl, f)

path_FnameClrDF=sInput+'\\gray_pnl.pkl'
with open(path_FnameClrDF, 'wb') as f:
    pickle.dump(gray_pnl, f)
# path_ClrDF_fromTarget_allPanels=sInput+'\ClrDF_fromTarget_allPanels.pkl'
# path_indexPanelNameDic=sInput+'\indexPanelNameDic.pkl'
# with open(path_ClrDF_fromTarget_allPanels, 'wb') as f:
#     pickle.dump(ClrDF_fromTarget_allPanels, f)

# with open(path_indexPanelNameDic, 'wb') as f:
#     pickle.dump(indexPanelNameDic, f)

# path_ClrDF_fromTargetS_goly=sInput+'\ClrDF_fromTargetS_goly_allPanels.pkl'
# with open(path_ClrDF_fromTargetS_goly, 'wb') as f:
#     pickle.dump(ClrDF_fromTargetS_goly_allPanels, f)


imInp_Orig = cv2.imread(r'E:\s8\for irene\12-0-11\FullImage.bmp')


plt.figure(00)
plt.imshow(imInp_Orig)


Side2Side=ClrDF_fromTargetS_goly_allPanels_continues[0]-ClrDF_fromTargetS_goly_allPanels_continues[len(ClrDF_fromTargetS_goly_allPanels_continues)-1]

# file_path = r'D:\BTDencoder\B2\04092026_poxy\ClrDF_fromTargetS_goly_allPanels.pkl'

# # Open the file in binary read mode
# with open(file_path, 'rb') as file:
#     # Unpickle the object from the file
#     ClrDF_fromTargetS_goly_allPanels = pickle.load(file)


# file_path = r'D:\BTDencoder\B2\04092026_poxy\indexPanelNameDic.pkl'

# # Open the file in binary read mode
# with open(file_path, 'rb') as file:
#     # Unpickle the object from the file
#     indexPanelNameDic = pickle.load(file)

# Cyan_Black_Sgoly={};
# RefCl='Cyan'
# for i in range(3):
#     Cyan_Black_Sgoly[i]=pd.DataFrame()
#     for col in ClrDF_fromTargetS_goly[i].columns:
#         if RefCl == col:
#             continue;
#         Cyan_Black_Sgoly[i] = pd.concat([Cyan_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)


# Green_Black_Sgoly={};
# RefCl='Green'
# for i in range(3):
#     Green_Black_Sgoly[i]=pd.DataFrame()
#     for col in ClrDF_fromTargetS_goly[i].columns:
#         if RefCl == col:
#             continue;
#         Green_Black_Sgoly[i] = pd.concat([Green_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)

        

##########################################################################################


# plt.figure()
# plt.plot(np.diff(ClrDF_raw[2]['Green']))
# plt.plot(abs(np.array(dyT)))    

# dyT=[]
   
# for i in range(1,len(yTargetDF['Green'])):
#     dyT.append(yTargetDF['Green'][i-1]-yTargetDF['Green'][i])
    
    
# plt.figure()
# plt.plot(dyT)  


# plt.figure()
# plt.plot((ClrDF_raw[0]['Green']-yTargetDF['Green'])*pixSize)
# # circle_centers,gray,circle_image,edges =  Circles(sInput).find_circles(ImRoi[2],30)

# # plt.figure(3)
# # plt.plot((ClrDF_fromTargetS_goly[0]['Cyan']-ClrDF_fromTargetS_goly[0]['Black'])*pixSize)

# current_line_sorted= Circles(sInput).SortCircle_Coord(circle_centers)

# current_line_sorted,cntMissingCircles= Circles(sInput).AddMissingCircles(current_line_sorted)

# ClrDF_fromTarget,ClrDF_fromTargetS_goly,dymeanList,yTargetDF=Circles(sInput).calcDiffernceFromeTarget(ClrDF_raw[2])
##################################################################

db= Side2Side;
# RefCl='Cyan'

PlotTitle='Side_2_Side'

fileName=PlotTitle+'.html'

##################################################################
ymax=100

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


##################################################################

db= C2Cmat_allPanels;
PlotTitle='C2C'
fileName=PlotTitle+'.html'
figC2C_multiPanel= PlotSingle_Basic_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100)
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
db= Color_Black_Sgoly_allPanels_continues;
# RefCl='Cyan'
RefCl=RefC01

PlotTitle='2 color diff - '+RefCl +' Vs color'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

##########################################################################################
db= Green_Black_Sgoly_allPanels_continues;

# RefCl='Magenta'
RefCl=RefC02


PlotTitle='2 color diff -'+RefCl +' Vs color'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])


##########################################################################################
db= col1_col2_Sgoly_allPanels_continues;
# RefCl='Cyan'
RefCl=RefC03

PlotTitle='2 color diff - '+RefCl +' Vs color'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

##########################################################################################
db= col2_col3_Sgoly_allPanels_continues;

# RefCl='Magenta'
RefCl=RefC04


PlotTitle='2 color diff -'+RefCl +' Vs color'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])


##########################################################################################
db= col3_col4_Sgoly_allPanels_continues;
# RefCl='Cyan'
RefCl=RefC05

PlotTitle='2 color diff - '+RefCl +' Vs color'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

##########################################################################################
db= col4_col5_Sgoly_allPanels_continues;

# RefCl='Magenta'
RefCl=RefC06


PlotTitle='2 color diff -'+RefCl +' Vs color'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])


##########################################################################################
db= col5_col6_Sgoly_allPanels_continues;

# RefCl='Magenta'
RefCl=RefC07


PlotTitle='2 color diff -'+RefCl +' Vs color'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])


##########################################################################################
##########################################################################################
##########################################################################################
db= ClrDF_fromTargetS_goly_allPanels_continues;
PlotTitle='Single color from Target'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_colorFromTarget= PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

#############################################################################################
##########################################################################################

db= ClrDF_fromTarget_allPanels_continues;
PlotTitle='Single color from Target-NO FILTER'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_colorFromTarget= PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])




#############################################################################################


ClrDF_rawSide=ClrDF_rawXY[i]

DistanceBetweenColumns_side=DistanceBetweenColumns[i]


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
    ClrDF_fromTargetS_goly[col]=savgol5_filter((pd.Series(yTarget)- ClrDF_rawSide[col]), MaxWaveWindow, S_g_Degree)
    yTargetDF=pd.concat((yTargetDF,pd.Series(yTarget).rename(col)),axis=1)
    yTarget=[]
    xTargetDF=pd.concat((xTargetDF,pd.Series(xTarget).rename(col+'_x')),axis=1)
    xTarget=[]
    if j < len(colorInUseName)-1:
        # strartPos_x = DistBtwCrcle/AQMscale +strartPos_x
        strartPos_x = DistBtwCrcle/ScaleY +strartPos_x



