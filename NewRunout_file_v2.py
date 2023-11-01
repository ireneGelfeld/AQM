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


# import plotly.io as pio
# pio.renderers
# pio.renderers.default='browser'

# %matplotlib
############################################################################
global ColorDic,pixSize,MaxWaveWindow,sideDic,CircleArea,DistBtwCrcle,UseTarget,DistanceBetweenColumns

SatndardDistanceBetweenColumns= 404.9626381019503;
LargeDistanceBetweenColumns= 1431.057469809592;
UseTarget =1 # False- uses the average distance between circles in pixels, True- uses DistBtwCrcle*AQMscale = (18 x 0.9832)

DistBtwCrcle=18
ScaleY=0.9965001812608202
AQMscale=0.9832
CircleArea=15
MaxWaveWindow=11
S_g_Degree=1
ColorDic={0:'Magenta',1:'Yellow',2:'Blue',3:'Orange',4:'Cyan',5:'Green',6:'Black'}
pixSize = 84.6666 # [um]
sideDic={0:'Left Side',1:'Middle',2:'Right Side'}

ColorDicNum={'Magenta':0,'Yellow':1,'Blue':2,'Orange':3,'Cyan':4,'Green':5,'Black':6}


DistanceBetweenColumns={i:SatndardDistanceBetweenColumns*1 for i in range(2,17)}
DistanceBetweenColumns[0]=0

DistanceBetweenColumns[1]=LargeDistanceBetweenColumns*1
DistanceBetweenColumns[17]=LargeDistanceBetweenColumns*1


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
    
    def loadImage(self):
    
        imInp_Orig = cv2.imread(self.pthF)
    
        Roi={}
        
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


        for i in range(18):
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
                cv2.circle(circle_image, (center_xINT, center_yINT), 5, (50, 50, 50), -1)
                
                
        return circle_centers,gray,circle_image,edges
    
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
            
            while len(y) != 439:
        
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
       
           
       ImgClr={}
       ClrDF_raw={}
       ClrDF_rawXY={}

       current_line_sortedDIC={}

       for i in range(len(ImRoi.keys())):
           
           circle_centers,gray,circle_image,edges=self.find_circles(ImRoi[i],15);

           
           current_line_sorted=self.SortCircle_Coord(circle_centers);
           
           if len(current_line_sorted.keys()) > 7:
               del current_line_sorted[7]
           
           current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
           
           current_line_sorted = self.filterCircles(current_line_sorted)
           
           current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);

           
           ClrDF=pd.DataFrame()
           ClrDF_x=pd.DataFrame()

           tmp_df=pd.DataFrame()
           tmp_df_xy=pd.DataFrame()
           for Key,Value in ColorDic.items():
               tmp_df = pd.DataFrame({Value: self.CreatColorList(Key,current_line_sorted)})
               ClrDF = pd.concat([ClrDF, tmp_df], axis=1)
               ClrDF_x = pd.concat([ClrDF_x, tmp_df], axis=1)
               tmp_df_x = pd.DataFrame({Value+'_x': self.CreatColorList_x(Key,current_line_sorted,StartRoiCoed[i])})
               ClrDF_x = pd.concat([ClrDF_x, tmp_df_x], axis=1)

               
               
           
            
           current_line_sortedDIC[i] = current_line_sorted
           ClrDF_raw[i]=ClrDF
           ClrDF_rawXY[i]=ClrDF_x

           
       return gray,circle_image,edges,ClrDF_raw,ClrDF_rawXY,current_line_sortedDIC  
 
    

 
    
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
   
   dashSolidDot={0:'solid',1:'dash',2:'dot'}
   

   for i in range(3):
       db=dbwAll[i]
       
       for c in colorInUseName:  
           
           lineColor = c
               
           if lineColor=='Yellow':
               lineColor='gold';
           
           fig.add_trace(go.Scatter(x=x,y=list(db[c]),mode='lines',
                                    line=dict(color=lineColor, dash=dashSolidDot[i]),name=c+' ' +sideDic[i]))
           fig.data[len(fig.data)-1].visible = 'legendonly';
           fig.add_trace(go.Scatter(x=x,y=savgol_filter(db[c], MaxWaveWindow, S_g_Degree), 
                                    line=dict(color=lineColor, dash=dashSolidDot[i]) ,name='S.Goly ' + c+' ' +sideDic[i]))

       
   

   
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


def PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,ymax):
    
   fig = go.Figure()
   
   dashSolidDot={0:'solid',1:'dash',2:'dot'}
   PageSide={0:'Left',1:'Middle',2:'Right'}
   
   for i in range(3):
       for c in db[i].columns:  
           lineColor = c
                
           if lineColor=='Yellow':
                lineColor='gold';
                
           fig.add_trace(go.Scatter(y=list(db[i][c]*pixSize),line=dict(color=lineColor, dash=dashSolidDot[i]) , name=c+' '+PageSide[i]))
       
   
    
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

def PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,ymax):
    
   fig = go.Figure()
   
   dashSolidDot={0:'solid',1:'dash',2:'dot'}
   PageSide={0:'Left',1:'Middle',2:'Right'}
  
    
   for i in range(3):
        for c in db[i].columns:  
           
            lineColor=c.split('-')[1]
                
            if lineColor=='Yellow':
                lineColor='gold';
                
            fig.add_trace(go.Scatter(y=list(db[i][c]*pixSize), line=dict(color=lineColor, dash=dashSolidDot[i])  ,name=c+' '+PageSide[i]))
           
   
    
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


def Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,Color_Black_Sgoly_allPanels,RefCl,colorInUseName):
    
    Color_Black_Sgoly={};

    for i in range(3):
        Color_Black_Sgoly[i]=pd.DataFrame()
        for col in colorInUseName:
            if RefCl == col:
                continue;
            Color_Black_Sgoly[i] = pd.concat([Color_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)
        Color_Black_Sgoly_allPanels[i]=pd.concat([Color_Black_Sgoly_allPanels[i], Color_Black_Sgoly[i]])

    return Color_Black_Sgoly_allPanels



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


colorInUseName=['Magenta','Yellow','Blue','Orange','Cyan','Green','Black']


# colorInUseName=['Magenta','Yellow','Cyan','Black']
colorInUseNum=[ColorDicNum[itm] for itm in colorInUseName]

# sInputListL=[sInputList[0]]

# sInput = r'D:\MainProj\Undefined[22][29-06-2023 16-48-59]'

# pnl = '\\394-0-3'


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


C2Cmat_allPanels=pd.DataFrame()
Cyan_Black_Sgoly_allPanels={};
Green_Black_Sgoly_allPanels={};
ClrDF_fromTargetS_goly_allPanels={}
ClrDF_fromTarget_allPanels={}
Color_Black_Sgoly_allPanels={}

ClrDF_fromTargetPnl={}

for i in range(len(ImRoi.keys())):
    Cyan_Black_Sgoly_allPanels[i]=pd.DataFrame();
    Green_Black_Sgoly_allPanels[i]=pd.DataFrame();
    ClrDF_fromTargetS_goly_allPanels[i]=pd.DataFrame();
    ClrDF_fromTarget_allPanels[i]=pd.DataFrame();
    Color_Black_Sgoly_allPanels[i]=pd.DataFrame();

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

for Pnl in sInputListSORTED:

    # ImRoi,StartRoiCoed= Circles(sInput+'\\'+Pnl+fileNME).loadImage();
    pth=r'D:\B8\new file\95-0-18\FullImage.bmp'
    # pth=sInput+'\\'+Pnl+fileNME
    ImRoi,StartRoiCoed= Circles(pth).loadImage();

    gray,circle_image,edges,ClrDF_raw, ClrDF_rawXY,current_line_sortedDIC = Circles(pth).CalcorColorMat(ImRoi,StartRoiCoed)
    
    
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
    for i in range(len(ImRoi.keys())):
        
        # dymeanList=[]
    
        # dymeanList=Circles(sInput+'\\'+Pnl+fileNME).CalcDiffTarget(ClrDF_raw[i],dymeanList)
    
        # ClrDF_fromTarget[i],ClrDF_fromTargetS_goly[i],dymeanListdic[i],yTargetDF,strartPos = Circles(pth).calcDiffernceFromeTarget(ClrDF_raw[i],dymeanList,colorInUseName,strartPos);
        
        ClrDF_fromTarget[i],ClrDF_fromTargetS_goly[i],dymeanListdic[i],yTargetDF,strartPos,strartPos_x,xTargetDF = Circles(pth).calcDiffernceFromeTargetXY(ClrDF_rawXY[i],dymeanList,colorInUseName,strartPos,strartPos_x,DistanceBetweenColumns[i]);
        xTargetDF_all[i]=xTargetDF
        yTargetDF_all[i]=yTargetDF

        x[i]=np.mean(dymeanListdic[i])
        ClrDF_fromTargetS_goly_allPanels[i]=pd.concat([ClrDF_fromTargetS_goly_allPanels[i], ClrDF_fromTargetS_goly[i]])
        ClrDF_fromTarget_allPanels[i]=pd.concat([ClrDF_fromTarget_allPanels[i], ClrDF_fromTarget[i]])
        ClrDF_fromTargetSide[i]=ClrDF_fromTarget[i]
    C2Cmat=pd.DataFrame()
    ClrDF_fromTargetPnl[Pnl]=ClrDF_fromTargetSide
    for i in range(3):
        C2Cmat[sideDic[i]] = Circles(sInput+'\\'+Pnl+fileNME).calcC2C('Magenta', ClrDF_fromTargetS_goly[i]);
        
    C2Cmat_allPanels= pd.concat([C2Cmat_allPanels, C2Cmat])  
    
    RefC01='Blue'
    Cyan_Black_Sgoly_allPanels=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,Cyan_Black_Sgoly_allPanels,RefC01,colorInUseName)
    # Cyan_Black_Sgoly={};
    # RefCl='Cyan'
    # for i in range(3):
    #     Cyan_Black_Sgoly[i]=pd.DataFrame()
    #     for col in ClrDF_fromTargetS_goly[i].columns:
    #         if RefCl == col:
    #             continue;
    #         Cyan_Black_Sgoly[i] = pd.concat([Cyan_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)
    #     Cyan_Black_Sgoly_allPanels[i]=pd.concat([Cyan_Black_Sgoly_allPanels[i], Cyan_Black_Sgoly[i]])
    
    
    # RefCl='Magenta'
    RefCl11='Magenta'

    Color_Black_Sgoly_allPanels=Color_To_Black_4allPanel(ClrDF_fromTargetS_goly,Color_Black_Sgoly_allPanels,RefCl11,colorInUseName)
   
    
    
    
    indexPanelNameDic[len(C2Cmat_allPanels.iloc[:, 0])-1] = Pnl
    
    print(Pnl)




    
   
C2Cmat_allPanels= C2Cmat_allPanels.reset_index(drop=True)    


for i in range(3):
    Color_Black_Sgoly_allPanels[i]=Color_Black_Sgoly_allPanels[i].reset_index(drop=True)
    Cyan_Black_Sgoly_allPanels[i]=Cyan_Black_Sgoly_allPanels[i].reset_index(drop=True) 
    ClrDF_fromTargetS_goly_allPanels[i]=ClrDF_fromTargetS_goly_allPanels[i].reset_index(drop=True) 

# path_FnameClrDF=r'E:\B2_17082023\PCCimage\Production[250][17-08-2023 11-34-07]\ClrDF_fromTargetPnl.pkl'
# with open(path_FnameClrDF, 'wb') as f:
#     pickle.dump(ClrDF_fromTargetPnl, f)


path_ClrDF_fromTarget_allPanels=sInput+'\ClrDF_fromTarget_allPanels.pkl'
path_indexPanelNameDic=sInput+'\indexPanelNameDic.pkl'
with open(path_ClrDF_fromTarget_allPanels, 'wb') as f:
    pickle.dump(ClrDF_fromTarget_allPanels, f)

with open(path_indexPanelNameDic, 'wb') as f:
    pickle.dump(indexPanelNameDic, f)

path_ClrDF_fromTargetS_goly=sInput+'\ClrDF_fromTargetS_goly_allPanels.pkl'
with open(path_ClrDF_fromTargetS_goly, 'wb') as f:
    pickle.dump(ClrDF_fromTargetS_goly_allPanels, f)



Side2Side=ClrDF_fromTargetS_goly_allPanels[0]-ClrDF_fromTargetS_goly_allPanels[2]

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
db= Cyan_Black_Sgoly_allPanels;
# RefCl='Cyan'
RefCl=RefC01

PlotTitle='2 color diff - '+RefCl +' Vs color'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100)

##########################################################################################
db= Color_Black_Sgoly_allPanels;

# RefCl='Magenta'
RefCl=RefCl11


PlotTitle='2 color diff -'+RefCl +' Vs color'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100)

##########################################################################################
##########################################################################################
##########################################################################################
db= ClrDF_fromTargetS_goly_allPanels;
PlotTitle='Single color from Target'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_colorFromTarget= PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100)

#############################################################################################
##########################################################################################

db= ClrDF_fromTarget_allPanels;
PlotTitle='Single color from Target-NO FILTER'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_colorFromTarget= PlotSingle_BasicVsTraget_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100)




#############################################################################################







###########################  CumSum
# CumSumClrDiff_fromTargetPnl={}

# for Pnl in sInputListSORTED:
#   CumSumClrDiff_fromTargetSide={}
  
#   for i in range(3):      
#       x,CumSumClrDiff_fromTargetSide[i]= CalcCumError(ClrDF_fromTargetPnl[Pnl][i],200)
  
#   CumSumClrDiff_fromTargetPnl[Pnl] = CumSumClrDiff_fromTargetSide
  
# #############################################################################################
# #############################################################################################
# #############################################################################################
# for n in range(len(CumSumClrDiff_fromTargetPnl.keys())):
#     db= CumSumClrDiff_fromTargetPnl[sInputListSORTED[n]][0];
#     PlotTitle='CumSum for '+sInputListSORTED[n]+' '+sideDic[0]
#     fileName=PlotTitle+'.html'
    
    
#     CumSumClrDiff_fig=PlotSingle_Basic(x,db,PlotTitle,fileName)
    


####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################


# clrFomTarget=ClrDF_fromTargetPnl[Pnl][i]
# LngthLimit=200

# cumSmPnl=pd.DataFrame() 
# integral_value=pd.DataFrame() 

# for col in clrFomTarget.columns:
#     y= clrFomTarget[col][:LngthLimit]*pixSize
    
#     x=np.array(range(LngthLimit))*DistBtwCrcle*AQMscale*pixSize*1e-3;
#     coefficients = np.polyfit(x, y, deg=1)
 
#     # Get the slope and intercept of the linear line
#     slope = coefficients[0]
#     intercept = coefficients[1]
 
#     # # Print the equation of the line
#     # print(f"Equation of the line: y = {slope:.2f}x + {intercept:.2f}")
 
#     # Predict y-values for the original x-values
#     y_predicted = np.polyval(coefficients, x)
#     cumSmPnl[col]=list(np.cumsum(y_predicted-y))
    
#     x_vals=np.array(range(len(y_predicted-y)))
#     y_vals=y_predicted-y
#     intCalc=[]
#     for j in range(len(y_vals)-1):
#         intCalc.append(np.trapz(y_vals[:j+1], x_vals[:j+1]))
#     integral_value[col] = intCalc



# circle_imageLineSorted = np.ones_like(gray)*220

# k=2

# for center in current_line_sorted[k]:
#     center_xINT=int(center[0])
#     center_yINT=int(center[1])

#     cv2.circle(circle_imageLineSorted, (center_xINT, center_yINT), 5, (50, 50, 50), -1)




# adjusted_image = np.where(gray > 220, 220, np.where(gray < 50, 50, gray))

# dIm=circle_image-adjusted_image


# plt.figure(1)
# plt.imshow(dIm)



# plt.figure(2)
# plt.imshow(adjusted_image)

# plt.figure(3)
# plt.imshow(circle_image)


# plt.figure(4)
# plt.imshow(circle_imageLineSorted)

# circle_image = np.ones_like(gray)*220

# for k in current_line_sorted.keys():
#     for center in current_line_sorted[k]:
#        # x_values = [int(center[0]) for center in current_line_sorted[k]]
   
#        # y_values = [int(center[1]) for center in current_line_sorted[k]]

 
#         # Draw a circle at the center of the contour
#        cv2.circle(circle_image, (int(center[0]), int(center[1])), 5, (50, 50, 50), -1)
                

# ##################################################################################

# gray,circle_image,edges,ClrDF_raw = Circles(sInput+'\\'+Pnl+fileNME).CalcorColorMat(ImRoi)
# ############


# ## Plot differance from Target

# # db= ClrDF_fromTargetOP_side;
# # PageSide='OP-Side'
# # PlotTitle='Color differance from Traget '+PageSide
# # fileName=PlotTitle+'.html'
# # figMagentaVsClrs_Op_Side= PlotSingle(db,PlotTitle,fileName, db.columns, PageSide,0)


# ## Plot differance from Target S.Goly
# plt.figure(1)
# plt.imshow(imInp_Orig)

# db= ClrDF_fromTarget[0];
# PageSide='OP-Side'
# PlotTitle='Color differance from Traget '+PageSide+' '+Pnl
# fileName=PlotTitle+'.html'
# figTargetOP= PlotSingleWstarvitsky(x[0]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, db.columns, PageSide,0)



# db= ClrDF_fromTarget[1];
# PageSide='Middle'
# PlotTitle='Color differance from Traget '+PageSide+' '+Pnl
# fileName=PlotTitle+'.html'
# figTargetMiddle= PlotSingleWstarvitsky(x[1]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, db.columns, PageSide,0)

# ClrDF_raw[2]['Green']
# yTargetDF['Green']

# db= ClrDF_fromTarget[2];
# PageSide='DR-Side'
# PlotTitle='Color differance from Traget '+PageSide+' '+Pnl
# fileName=PlotTitle+'.html'
# figTargetDR= PlotSingleWstarvitsky(x[2]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, db.columns, PageSide,0)

# plt.Figure()
# plt.plot(ClrDF_fromTarget[2]['Green'])
# # xMean=np.mean([x[0],x[1],x[2]])
# plt.Figure()
# plt.plot(np.diff(ClrDF_raw[2]['Green']))
# ## Plot C2C 

# db= C2Cmat;
# PlotTitle='C2C of Panel '+' '+Pnl
# fileName=PlotTitle+'.html'
# C2C_All_Side= PlotSingle_Basic(xMean*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName)


# ##################################################################


# dbwAll= ClrDF_fromTarget;
# PlotTitle='Color differance from Traget for all sides '+' '+Pnl
# fileName=PlotTitle+'.html'
# figTargets_all_Side=PlotSingle_Wstarvitsky_allInOneGraph(xMean*84.666*1e-3*np.array(range(439)),dbwAll,PlotTitle,fileName)





# ########################################################
# ##########################################################################################
# ##########################################################################################
# ##########################################################################################


# ColList = Cyan_Black_Sgoly[0].columns
# RefCl='Cyan'
# x=[17.8,17.8,17.8]
# ###############################################            
# db= Cyan_Black_Sgoly[0]*pixSize;
# PageSide='OP-Side'
# PlotTitle='Diff: '+RefCl+' Vs Colors '+PageSide+' '+Pnl
# fileName=PageSide+' '+RefCl+'VsColors.html'
# figDiffVsClrs_Op_Side= PlotSingle_wAverage(x[0]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, ColList, PageSide,1,'Offset')
# ###############################################
# db= Cyan_Black_Sgoly[1]*pixSize;
# PageSide='Mid-Side'
# PlotTitle='Diff: '+RefCl+' Vs Colors '+PageSide+' '+Pnl
# fileName=PageSide+' '+Clr1+'VsColors.html'
# figDiffVsClrs_Mid_Side= PlotSingle_wAverage(x[1]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, ColList, PageSide,1,'Offset')

# ###############################################
# db= Cyan_Black_Sgoly[2]*pixSize;
# PageSide='Dr-Side'
# PlotTitle='Diff: '+RefCl+' Vs Colors '+PageSide+' '+Pnl
# fileName=PageSide+' '+Clr1+'VsColors.html'
# figDiffVsClrs_Dr_Side= PlotSingle_wAverage(x[2]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, ColList, PageSide,1,'Offset')

# ###############################################


# ColList = Green_Black_Sgoly[0].columns
# RefCl='Green'
# x=[17.8,17.8,17.8]
# ###############################################            
# db= Green_Black_Sgoly[0]*pixSize;
# PageSide='OP-Side'
# PlotTitle='Diff: '+RefCl+' Vs Colors '+PageSide+' '+Pnl
# fileName=PageSide+' '+RefCl+'VsColors.html'
# figDiffVsClrs_Op_Side= PlotSingle_wAverage(x[0]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, ColList, PageSide,1,'Offset')
# ###############################################
# db= Green_Black_Sgoly[1]*pixSize;
# PageSide='Mid-Side'
# PlotTitle='Diff: '+RefCl+' Vs Colors '+PageSide+' '+Pnl
# fileName=PageSide+' '+Clr1+'VsColors.html'
# figDiffVsClrs_Mid_Side= PlotSingle_wAverage(x[1]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, ColList, PageSide,1,'Offset')

# ###############################################
# db= Green_Black_Sgoly[2]*pixSize;
# PageSide='Dr-Side'
# PlotTitle='Diff: '+RefCl+' Vs Colors '+PageSide+' '+Pnl
# fileName=PageSide+' '+Clr1+'VsColors.html'
# figDiffVsClrs_Dr_Side= PlotSingle_wAverage(x[2]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, ColList, PageSide,1,'Offset')
  
# ##################################################################
# ##################################################################
# ##################################################################
# ##################################################################
# RefCl= 'Cyan'
# db= Cyan_Black_Sgoly[0]*pixSize;
# PageSide='OP-Side'
# PlotTitle='FFT '+RefCl+' Vs Colors '+PageSide+' '+Pnl
# fileName=PageSide+' '+RefCl+'FFT.html'
# FFT_Op_Side= FFT_Plot(db,PlotTitle,fileName, (7830/17.7), PageSide)
##################################################################
# ImgClr={}
# ClrDF_raw={}

# for i in range(3):
    
#     circle_centers,gray,circle_image,edges=Circles(sInput+'\\'+Pnl+fileNME).find_circles(ImRoi[i],15);

    
#     current_line_sorted=Circles(sInput+'\\'+Pnl+fileNME).SortCircle_Coord(circle_centers);
    
#     if len(current_line_sorted.keys()) > 7:
#         del current_line_sorted[7]
    
#     current_line_sorted,cntMissingCircles=Circles(sInput+'\\'+Pnl+fileNME).AddMissingCircles(current_line_sorted);
    
#     current_line_sorted = Circles(sInput+'\\'+Pnl+fileNME).filterCircles(current_line_sorted)
    
#     current_line_sorted,cntMissingCircles=Circles(sInput+'\\'+Pnl+fileNME).AddMissingCircles(current_line_sorted);

    
#     ClrDF=pd.DataFrame()
#     tmp_df=pd.DataFrame()
#     for Key,Value in ColorDic.items():
#         tmp_df = pd.DataFrame({Value: Circles(sInput+'\\'+Pnl+fileNME).CreatColorList(Key,current_line_sorted)})
#         ClrDF = pd.concat([ClrDF, tmp_df], axis=1)

# ####################################################################

# circles_sorted1 = sorted(circle_centers, key=lambda x: x[0])

# x_values = [center[0] for center in circles_sorted1]

# y_values = [center[1] for center in circles_sorted1]


# # plt.figure(1)
# # plt.plot(x_values,'x')

# current_line = {}

# k=0
# l=[]

# for i in range(len(x_values) - 1):
  
#     diff = x_values[i + 1] - x_values[i]
#     l.append((x_values[i],y_values[i]))
  
#     if abs(diff) > 5:  # Adjust the threshold as needed
#         if len(l)<10:
#             l=[] 
#             continue;
#         current_line[k]=l
#         l=[]        
#         k=k+1;    

# l.append((x_values[len(x_values)-1],y_values[len(x_values)-1]))

# current_line[k]=l

# current_line_sorted={}
# for k,l in current_line.items():
#     l_sorted = sorted(l, key=lambda x: x[1])
#     current_line_sorted[k]=l_sorted

    
# ##################################################################
# RefCl='Green'

# db= Green_Black_Sgoly_allPanels[0]*pixSize;
# PageSide='OP-Side'
# PlotTitle='FFT '+RefCl+' Vs Colors '+PageSide+' '+Pnl
# fileName=PageSide+' '+RefCl+'FFT.html'
# FFT_Op_Side= FFT_Plot(db,PlotTitle,fileName,1/((17*84.666*1e-3*0.9832)*1e-3), PageSide,'[1/m]')


RefCl='Targe'

db= ClrDF_fromTarget_allPanels[0]*pixSize;
PageSide='left-Side'
PlotTitle='FFT '+RefCl
fileName=PageSide+' '+RefCl+'FFT.html'
FFT_TrgtM_Side= FFT_Plot(db,PlotTitle,fileName,1/((18*84.666*1e-6*0.9832)), PageSide,'[1/m]',colorInUseName)


RefCl='Targe- Hz'

db= ClrDF_fromTarget_allPanels[0]*pixSize;
PageSide='left-Side'
PlotTitle='FFT '+RefCl
fileName=PageSide+' '+RefCl+'FFT.html'
FFT_TrgHz_Side= FFT_Plot(db,PlotTitle,fileName,439/((7549/12480)*0.504), PageSide,'[Hz]',colorInUseName)


# DEBUG find_circles
# adjusted_image = np.where(gray > 220, 220, np.where(gray < 50, 50, gray))



# dIm=circle_image-adjusted_image

# # result_circle_image = calculate_average_2d_array(circle_image)

# # print('circle_image-adjusted_image: ',calculate_average_2d_array(dIm))
# # print('adjusted_image: ',calculate_average_2d_array(adjusted_image))
# # print('circle_image: ',calculate_average_2d_array(circle_image))


# plt.figure('dIm2')
# plt.imshow(dIm)


plt.figure('ImRoi2')
plt.imshow(ImRoi[i])

# plt.figure('adjusted_image2')
# plt.imshow(adjusted_image)

# plt.figure('circle_image2')
# plt.imshow(circle_image)

###########################################################

        


    
###############################################
###############################################
###############################################
# db= ImgClr;
# PlotTitle='Integral Error: '+Clr1+' Vs Colors'
# fileName=Clr1+'VsColors.html'

# figMagentaVsClrs= Plot3subPlots(db,PlotTitle,fileName)
###############################################
###############################################

# ln_list=[len(ImgClr[0].columns),len(ImgClr[1].columns),len(ImgClr[2].columns)]
# min_index = ln_list.index(min(ln_list))
# ColList = ImgClr[min_index].columns

# ###############################################            
# db= ImgClr[0];
# PageSide='OP-Side'
# PlotTitle='Integral Error: '+Clr1+' Vs Colors '+PageSide
# fileName=PageSide+' '+Clr1+'VsColors.html'
# figMagentaVsClrs_Op_Side= PlotSingle(db,PlotTitle,fileName, ColList, PageSide,1)
# ###############################################

# db= ImgClr[1];
# PageSide='Mid-Side'
# PlotTitle='Integral Error: '+Clr1+' Vs Colors '+PageSide
# fileName=PageSide+' '+Clr1+'VsColors.html'
# figMagentaVsClrs_Mid_Side= PlotSingle(db,PlotTitle,fileName, ColList, PageSide,1)

# ###############################################

# db= ImgClr[2];
# PageSide='Dr-Side'
# PlotTitle='Integral Error: '+Clr1+' Vs Colors '+PageSide
# fileName=PageSide+' '+Clr1+'VsColors.html'
# figMagentaVsClrs_Dr_Side= PlotSingle(db,PlotTitle,fileName, ColList, PageSide,1)

# ###############################################
# ###############################################

# ln_list=[len(ImgClr[0].columns),len(ImgClr[1].columns),len(ImgClr[2].columns)]
# min_index = ln_list.index(min(ln_list))
# ColList = ImgClr[min_index].columns

# ###############################################            
# db= ImgClr[0];
# PageSide='OP-Side'
# PlotTitle='Integral Error: '+Clr1+' Vs Colors '+PageSide
# fileName=PageSide+' '+Clr1+'VsColors.html'
# figMagentaVsClrs_Op_Side= PlotSingleWstarvitsky(x[0]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, ColList, PageSide,1)
# ###############################################

# db= ImgClr[1];
# PageSide='Mid-Side'
# PlotTitle='Integral Error: '+Clr1+' Vs Colors '+PageSide
# fileName=PageSide+' '+Clr1+'VsColors.html'
# figMagentaVsClrs_Mid_Side= PlotSingleWstarvitsky(x[0]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, ColList, PageSide,1)

# ###############################################

# db= ImgClr[2];
# PageSide='Dr-Side'
# PlotTitle='Integral Error: '+Clr1+' Vs Colors '+PageSide
# fileName=PageSide+' '+Clr1+'VsColors.html'
# figMagentaVsClrs_Dr_Side= PlotSingleWstarvitsky(x[0]*84.666*1e-3*np.array(range(439)),db,PlotTitle,fileName, ColList, PageSide,1)

###############################################



##############################################################################################################

# # image = cv2.imread(image_path)

# # Create a blank image of the same size as the original image
# circle_image = np.ones_like(image)*255

# # plt.Figure()
# # plt.imshow(circle_image)


# # Draw the circles on the circle image
# for (x, y) in circle_centers:
#     # print(x,y)
#     cv2.circle(circle_image, (x, y), 5, (0, 0, 255), -1)

# # Display the circle image
# cv2.imshow("Circle Image", circle_image)
# cv2.waitKey(0)

#############################################################################################################


dxList=[]
for i in range(len(ClrDF_rawXY.keys())-1):
    dx = abs(np.mean(ClrDF_rawXY[i]['Black_x'])-np.mean(ClrDF_rawXY[i+1]['Magenta_x']))
    dxList.append(dx)




colX= list(xTargetDF_all[0].columns)
dxColor=[]
# k=1
for k in range(len(ClrDF_rawXY.keys())):
    for i in range(len(colX)-1):
        dx = abs(np.mean(ClrDF_rawXY[k][colX[i]])-np.mean(ClrDF_rawXY[k][colX[i+1]]))
        dxColor.append(dx)


plt.figure(1)
plt.plot(dxColor)

np.mean(dxColor)

small=np.mean(dxList[1:16])
large=np.mean([dxList[0],dxList[16]])


        
ClrDF_TargetXY={}

for i in range(len(ClrDF_rawXY.keys())):
    targetDF=pd.DataFrame()
    for col in yTargetDF_all[i].columns:
        targetDF=pd.concat((targetDF,yTargetDF_all[i][col].rename(col)),axis=1)
        targetDF=pd.concat((targetDF,xTargetDF_all[i][col+'_x'].rename(col+'_x')),axis=1)
    ClrDF_TargetXY[i]=targetDF

DeltaTarget_result={}
for i in range(len(ClrDF_TargetXY.keys())):
    DeltaTarget_result[i]=  ClrDF_TargetXY[i]-  ClrDF_rawXY[i]




TmagentaYDic={}
TmagentaXDic={}

DeltaTarget_resultMagentaYDic={}
DeltaTarget_resultMagentaXDic={}


for color in colorInUseName:
    colorX=color+'_x'

    for i in range(len(ClrDF_rawXY.keys())):
        TmagentaY=pd.concat([TmagentaY,yTargetDF_all[i][color]],axis=1).rename(columns={color:color+str(i)})
        TmagentaX=pd.concat([TmagentaX,xTargetDF_all[i][colorX]],axis=1).rename(columns={colorX:colorX+str(i)})
        DeltaTarget_resultMagentaX=pd.concat([DeltaTarget_resultMagentaX,(xTargetDF_all[i][colorX]-ClrDF_rawXY[i][colorX])],axis=1).rename(columns={colorX:colorX+str(i)})
        DeltaTarget_resultMagentaY=pd.concat([DeltaTarget_resultMagentaY,(yTargetDF_all[i][color]-ClrDF_rawXY[i][color])],axis=1).rename(columns={color:color+str(i)})
    

    TmagentaYDic[color]=TmagentaY
    TmagentaXDic[color]=TmagentaX

    DeltaTarget_resultMagentaYDic[color]=DeltaTarget_resultMagentaY
    DeltaTarget_resultMagentaXDic[color]=DeltaTarget_resultMagentaX
    
    TmagentaY=pd.DataFrame()
    TmagentaX=pd.DataFrame()

    DeltaTarget_resultMagentaY=pd.DataFrame()
    DeltaTarget_resultMagentaX=pd.DataFrame()
    

col='Magenta'
plt.figure('all collors '+col)



colX=col+'_x'


# plt.figure(col)
    

# Meshgrid 
# x = np.array(xTargetDF_all[17].iloc[:200,:])
# y = np.array(yTargetDF_all[17].iloc[:200,:])
x = np.array(TmagentaXDic[col].iloc[:200,:])
y = np.array(TmagentaYDic[col].iloc[:200,:])


# X, Y = np.meshgrid(x, y)

# Directional vectors 
# u = np.array(DeltaTarget_result[17][list(xTargetDF_all[0].columns)].iloc[:200,:])
# v = np.zeros((200, 18))
u = np.array(DeltaTarget_resultMagentaXDic[col].iloc[:200,:])
# u =np.zeros((200, 18))


# v = np.array(DeltaTarget_resultMagentaYDic[col].iloc[:200,:])
v = np.zeros((200, 18))

# z = np.sqrt(np.square(u)+np.square(v))

if col== 'Yellow':
    col ='gold'

# arrow_scale = 0.02  # Adjust the size of the dots
# for i in range(len(x)):
#     for j in range(len(y)):
#         plt.plot(X[i, j] + arrow_scale * U[i, j], Y[i, j] + arrow_scale * V[i, j], 'o', color=col)

# # Plotting Vector Field with QUIVER 
plt.quiver(x, y, u, v, color=col, headaxislength=4, headlength=5, headwidth=3) 
plt.title('Vector Field') 

# Setting x, y boundary limits 
# plt.xlim(-7, 7) 
# plt.ylim(-7, 7) 

# Show plot with grid 
plt.grid() 
plt.show() 




plt.figure()
plt.plot(DeltaTarget_result[0]['Magenta_x']-DeltaTarget_result[0]['Green_x'])



##################################################################################


plt.figure('all collors1')


col='Magenta'
colX=col+'_x'
TmagentaY=pd.DataFrame()
TmagentaX=pd.DataFrame()

DeltaTarget_resultMagentaY=pd.DataFrame()
DeltaTarget_resultMagentaX=pd.DataFrame()

for i in range(len(ClrDF_rawXY.keys())):
    TmagentaY=pd.concat([TmagentaY,yTargetDF_all[i][col]],axis=1).rename(columns={col:col+str(i)})
    TmagentaX=pd.concat([TmagentaX,xTargetDF_all[i][colX]],axis=1).rename(columns={colX:colX+str(i)})
    DeltaTarget_resultMagentaX=pd.concat([DeltaTarget_resultMagentaX,(xTargetDF_all[i][colX]-ClrDF_rawXY[i][colX])],axis=1).rename(columns={colX:colX+str(i)})
    DeltaTarget_resultMagentaY=pd.concat([DeltaTarget_resultMagentaY,(yTargetDF_all[i][col]-ClrDF_rawXY[i][col])],axis=1).rename(columns={col:col+str(i)})

plt.figure(col)
    

# Meshgrid 
# x = np.array(xTargetDF_all[0].iloc[:,:])
# y = np.array(yTargetDF_all[0].iloc[:,:])
x = np.array(TmagentaX.iloc[:200,:])
y = np.array(TmagentaY.iloc[:200,:])

# X, Y = np.meshgrid(x, y)

# Directional vectors 
# u = np.array(DeltaTarget_result[0][list(xTargetDF_all[0].columns)].iloc[:,:])
# v = np.array(DeltaTarget_result[0][list(yTargetDF_all[0].columns)].iloc[:,:])
# u = np.array(DeltaTarget_resultMagentaX.iloc[:200,:])
u =np.zeros((200, 18))




v = np.array(DeltaTarget_resultMagentaY.iloc[:200,:])
# v = np.zeros((200, 18))


if col== 'Yellow':
    col ='gold'

# arrow_scale = 0.02  # Adjust the size of the dots
# for i in range(len(x)):
#     for j in range(len(y)):
#         plt.plot(X[i, j] + arrow_scale * U[i, j], Y[i, j] + arrow_scale * V[i, j], 'o', color=col)

# # Plotting Vector Field with QUIVER 
# plt.quiver(x, y, u, v, color=col, headaxislength=4, headlength=5, headwidth=3) 
# plt.title('Vector Field') 
# Depict illustration 
# plt.figure(figsize=(10, 10)) 
plt.quiver(x, y, u, v, color=col, headaxislength=4, headlength=5, headwidth=3) 
plt.title('Vector Field') 

# Setting x, y boundary limits 
# plt.xlim(-7, 7) 
# plt.ylim(-7, 7) 

# Show plot with grid 
plt.grid() 
plt.show() 



# imInp_Orig = cv2.imread(r'D:\B8\new file\95-0-18\FullImage.bmp')
# plt.Figure()
# plt.imshow(imInp_Orig)
########################################################################

ZallCols={}

colList=list(yTargetDF_all[0].columns)

for col in list(yTargetDF_all[0].columns):

    colX=col+'_x'
    
    TmagentaY=pd.DataFrame()
    TmagentaX=pd.DataFrame()
    
    DeltaTarget_resultMagentaY=pd.DataFrame()
    DeltaTarget_resultMagentaX=pd.DataFrame()
    
    for i in range(len(ClrDF_rawXY.keys())):
        TmagentaY=pd.concat([TmagentaY,yTargetDF_all[i][col]],axis=1).rename(columns={col:col+str(i)})
        TmagentaX=pd.concat([TmagentaX,xTargetDF_all[i][colX]],axis=1).rename(columns={colX:colX+str(i)})
        DeltaTarget_resultMagentaX=pd.concat([DeltaTarget_resultMagentaX,(xTargetDF_all[i][colX]-ClrDF_rawXY[i][colX])],axis=1).rename(columns={colX:colX+str(i)})
        DeltaTarget_resultMagentaY=pd.concat([DeltaTarget_resultMagentaY,(yTargetDF_all[i][col]-ClrDF_rawXY[i][col])],axis=1).rename(columns={col:col+str(i)})

    u = np.array(DeltaTarget_resultMagentaX.iloc[:200,:])
    # u =np.zeros((200, 18))


    v = np.array(DeltaTarget_resultMagentaY.iloc[:200,:])
    # v = np.zeros((200, 18))

    ZallCols[col] = np.sqrt(np.square(u)+np.square(v))
    
    
arr=  np.zeros((200, 18))

for i in range(18):
    for j in range(200):
        tmp= [ZallCols[col][j,i] for col in colList]
        arr[j,i]= np.max(tmp)-np.min(tmp)




# plt.figure('sns')

# colors = sns.color_palette("YlOrRd", as_cmap=True)

# sns.heatmap(arr*pixSize, annot=False, cmap=colors, fmt='.2f')

# # Set labels for the axes
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')

# # Show the heatmap
# plt.show()









