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



# import plotly.io as pio
# pio.renderers
# pio.renderers.default='browser'

# %matplotlib
############################################################################
global ColorDic,pixSize,MaxWaveWindow,sideDic,CircleArea


CircleArea=30
MaxWaveWindow=11
S_g_Degree=1
ColorDic={0:'Magenta',1:'Yellow',2:'Blue',3:'Orange',4:'Cyan',5:'Green',6:'Black'}
pixSize = 84.6666 # [um]
sideDic={0:'Left Side',1:'Middle',2:'Right Side'}


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
    
    
    def loadImage(self):
        
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
 
    def CalcorColorMat(self,ImRoi):
       
           
       ImgClr={}
       ClrDF_raw={}

       for i in range(3):
           
           circle_centers,gray,circle_image,edges=self.find_circles(ImRoi[i],15);

           
           current_line_sorted=self.SortCircle_Coord(circle_centers);
           
           if len(current_line_sorted.keys()) > 7:
               del current_line_sorted[7]
           
           current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
           
           current_line_sorted = self.filterCircles(current_line_sorted)
           
           current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);

           
           ClrDF=pd.DataFrame()
           tmp_df=pd.DataFrame()
           for Key,Value in ColorDic.items():
               tmp_df = pd.DataFrame({Value: self.CreatColorList(Key,current_line_sorted)})
               ClrDF = pd.concat([ClrDF, tmp_df], axis=1)
           
           ClrDF_raw[i]=ClrDF

           
       return gray,circle_image,edges,ClrDF_raw   
 
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
    
    def calcDiffernceFromeTarget(self,ClrDF_rawSide):
        
        yTarget=[]

        ClrDF_fromTarget=pd.DataFrame()
        ClrDF_fromTargetS_goly=pd.DataFrame()

        strartPos=ClrDF_rawSide['Magenta'][0]
        
        dymeanList=[]
        yTargetDF=pd.DataFrame()

        for col in ClrDF_rawSide.columns:
            dymean= np.mean(np.diff(ClrDF_rawSide[col])[:200])
            dymeanList.append(dymean)
            yTarget = [i * dymean + strartPos for i in range(len(ClrDF_rawSide[col]))]
            ClrDF_fromTarget[col]=(pd.Series(yTarget)- ClrDF_rawSide[col])*pixSize
            ClrDF_fromTargetS_goly[col]=savgol_filter((pd.Series(yTarget)- ClrDF_rawSide[col]), MaxWaveWindow, S_g_Degree)
            yTargetDF=pd.concat((yTargetDF,pd.Series(yTarget).rename(col)),axis=1)
            yTarget=[]

        return ClrDF_fromTarget,ClrDF_fromTargetS_goly,dymeanList,yTargetDF

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

def PlotSingle_Wstarvitsky_allInOneGraph(x,dbwAll,PlotTitle,fileName):
    
   fig = go.Figure()
   
   dashSolidDot={0:'solid',1:'dash',2:'dot'}
   

   for i in range(3):
       db=dbwAll[i]
       
       for c in db.columns:  
           
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
            
       fig.add_trace(go.Scatter(x=x,y=list(db[c]), name=c))
       
   
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


def FFT_Plot(db,PlotTitle,fileName, Fs, PageSide):






    N = len(db.index)
    
 
    p1=pd.DataFrame()
    T = 1/(Fs);
    x = np.linspace(0.0, N*T, N)
    y =pd.DataFrame();
    # Yfft=scipy.fftpack.fft(y)
    for col in db.columns:
        y[col]= np.fft.fft(db[col],axis=0)
        p1[col]=(1/(Fs*N))*pow(abs(y[col][:N//2]),2);
        
        p1[col][1:-1]=2*p1[col][1:-1];
  
    
    
    
    
    xf = np.linspace(0.0, Fs/2, int(N/2))
    
    
    fig = go.Figure()
    
    # rnge=[3,6,7]
    
    # db=ImagePlacement_Rightpp
    
    for col in p1.columns:
        
        lineColor=col.split('-')[1]
        fig.add_trace(
        
        go.Scatter(x=list(xf),y=list(10*np.log10(p1[col])),line_color=lineColor,name=col+' ' +PageSide))
    
    
    
    fig.update_layout( xaxis=dict(
            title='Freq [Hz]',
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
##############################################################################################################
##############################################################################################################
##############################################################################################################
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
sInput = filedialog.askdirectory()
os.chdir(sInput)


sInputList=[folder for folder in os.listdir(sInput) if os.path.isdir(os.path.join(sInput, folder)) and ('-' in folder)]


pnl = sInputList[0]

listLastDig=[int(ll.split('-')[2])   for ll in sInputList]


sorted_indices_list = sorted_indices(listLastDig)

sInputListSORTED=[]

for ll in  sorted_indices_list:
    sInputListSORTED.append(sInputList[ll])
    
Pnl = sInputListSORTED[3]

# sInputListL=[sInputList[0]]

# sInput = r'D:\MainProj\Undefined[22][29-06-2023 16-48-59]'

# pnl = '\\394-0-3'


# sInputList=['394-0-1','394-0-2','394-0-3','394-0-4','394-0-5','394-0-6']


fileNME='\\FullImage.bmp'


C2Cmat_allPanels=pd.DataFrame()
Cyan_Black_Sgoly_allPanels={};
Green_Black_Sgoly_allPanels={};
ClrDF_fromTargetS_goly_allPanels={}

for i in range(3):
    Cyan_Black_Sgoly_allPanels[i]=pd.DataFrame();
    Green_Black_Sgoly_allPanels[i]=pd.DataFrame();
    ClrDF_fromTargetS_goly_allPanels[i]=pd.DataFrame();

indexPanelNameDic={}

ImRoi= Circles(sInput+'\\'+pnl+fileNME).loadImage();
Clr1='Cyan'

# ImgClr,gray,circle_image,edges,ClrDF_raw = Circles(sInput).CalcIntegralError(ImRoi, Clr1)

# for Pnl in sInputListL:

for Pnl in sInputListSORTED:

    ImRoi= Circles(sInput+'\\'+Pnl+fileNME).loadImage();
    
    gray,circle_image,edges,ClrDF_raw = Circles(sInput+'\\'+Pnl+fileNME).CalcorColorMat(ImRoi)
    
    
    # C2CmatOP_side = Circles(sInput).calcC2C('Magenta', ClrDF_raw[0]);
    
    ClrDF_fromTargetS_goly={}
    ClrDF_fromTarget={}
    dymeanListdic={}
    x={}
    for i in range(3):
    
        ClrDF_fromTarget[i],ClrDF_fromTargetS_goly[i],dymeanListdic[i],yTargetDF = Circles(sInput+'\\'+Pnl+fileNME).calcDiffernceFromeTarget(ClrDF_raw[i]);
        x[i]=np.mean(dymeanListdic[i])
        ClrDF_fromTargetS_goly_allPanels[i]=pd.concat([ClrDF_fromTargetS_goly_allPanels[i], ClrDF_fromTargetS_goly[i]])
   
    C2Cmat=pd.DataFrame()
    
    for i in range(3):
        C2Cmat[sideDic[i]] = Circles(sInput+'\\'+Pnl+fileNME).calcC2C('Magenta', ClrDF_fromTargetS_goly[i]);
        
    C2Cmat_allPanels= pd.concat([C2Cmat_allPanels, C2Cmat])    
    Cyan_Black_Sgoly={};
    RefCl='Cyan'
    for i in range(3):
        Cyan_Black_Sgoly[i]=pd.DataFrame()
        for col in ClrDF_fromTargetS_goly[i].columns:
            if RefCl == col:
                continue;
            Cyan_Black_Sgoly[i] = pd.concat([Cyan_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)
        Cyan_Black_Sgoly_allPanels[i]=pd.concat([Cyan_Black_Sgoly_allPanels[i], Cyan_Black_Sgoly[i]])
    
    
    Green_Black_Sgoly={};
    RefCl='Green'
    for i in range(3):
        Green_Black_Sgoly[i]=pd.DataFrame()
        for col in ClrDF_fromTargetS_goly[i].columns:
            if RefCl == col:
                continue;
            Green_Black_Sgoly[i] = pd.concat([Green_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)
        Green_Black_Sgoly_allPanels[i]=pd.concat([Green_Black_Sgoly_allPanels[i], Green_Black_Sgoly[i]])

    
    
    
    indexPanelNameDic[len(C2Cmat_allPanels.iloc[:, 0])-1] = Pnl
    
    print(Pnl)
    
   
C2Cmat_allPanels= C2Cmat_allPanels.reset_index(drop=True)    


for i in range(3):
    Green_Black_Sgoly_allPanels[i]=Green_Black_Sgoly_allPanels[i].reset_index(drop=True)
    Cyan_Black_Sgoly_allPanels[i]=Cyan_Black_Sgoly_allPanels[i].reset_index(drop=True) 
    ClrDF_fromTargetS_goly_allPanels[i]=ClrDF_fromTargetS_goly_allPanels[i].reset_index(drop=True) 

Cyan_Black_Sgoly={};
RefCl='Cyan'
for i in range(3):
    Cyan_Black_Sgoly[i]=pd.DataFrame()
    for col in ClrDF_fromTargetS_goly[i].columns:
        if RefCl == col:
            continue;
        Cyan_Black_Sgoly[i] = pd.concat([Cyan_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)


Green_Black_Sgoly={};
RefCl='Green'
for i in range(3):
    Green_Black_Sgoly[i]=pd.DataFrame()
    for col in ClrDF_fromTargetS_goly[i].columns:
        if RefCl == col:
            continue;
        Green_Black_Sgoly[i] = pd.concat([Green_Black_Sgoly[i], (ClrDF_fromTargetS_goly[i][RefCl]-ClrDF_fromTargetS_goly[i][col]).rename(RefCl+'-'+col)], axis=1)

        

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

db= C2Cmat_allPanels;
PlotTitle='C2C'
fileName=PlotTitle+'.html'
figC2C_multiPanel= PlotSingle_Basic_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100)
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
db= Cyan_Black_Sgoly_allPanels;
PlotTitle='2 color diff -Cyan Vs color'

fileName=PlotTitle+'.html'
figCyanVsClr_multiPanel_OP= PlotSingle_DiffFromRefclr_multiPanel(db,PlotTitle,fileName,indexPanelNameDic,100)

##########################################################################################
db= Green_Black_Sgoly_allPanels;
PlotTitle='2 color diff - Green Vs color'

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

#############################################################################################
# yTarget=[]

# ClrDF_fromTarget=pd.DataFrame()
# ClrDF_fromTargetS_goly=pd.DataFrame()

# strartPos=ClrDF_rawSide['Magenta'][0]

# dymeanList=[]
# yTargetDF=pd.DataFrame()

# for col in ClrDF_rawSide.columns:
#     dymean= np.mean(np.diff(ClrDF_rawSide[col])[:200])
#     dymeanList.append(dymean)
#     yTarget = [i * dymean + strartPos for i in range(len(ClrDF_rawSide[col]))]
#     ClrDF_fromTarget[col]=(pd.Series(yTarget)- ClrDF_rawSide[col])*pixSize
#     ClrDF_fromTargetS_goly[col]=savgol_filter((pd.Series(yTarget)- ClrDF_rawSide[col]), MaxWaveWindow, S_g_Degree)
#     yTargetDF=pd.concat([yTargetDF,pd.Series(yTarget).rename(col)])    
#     yTarget=[]



# adjusted_image = np.where(gray > 220, 220, np.where(gray < 50, 50, gray))

# dIm=circle_image-adjusted_image


# plt.figure(1)
# plt.imshow(dIm)



# plt.figure(2)
# plt.imshow(adjusted_image)

# plt.figure(3)
# plt.imshow(circle_image)




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

    
# ##################################################################
# RefCl='Green'

# db= Green_Black_Sgoly[0]*pixSize;
# PageSide='OP-Side'
# PlotTitle='FFT '+RefCl+' Vs Colors '+PageSide+' '+Pnl
# fileName=PageSide+' '+RefCl+'FFT.html'
# FFT_Op_Side= FFT_Plot(db,PlotTitle,fileName, (7830/17.7), PageSide)

# DEBUG find_circles
# adjusted_image = np.where(gray > 220, 220, np.where(gray < 50, 50, gray))



# dIm=circle_image-adjusted_image

# # result_circle_image = calculate_average_2d_array(circle_image)

# print('circle_image-adjusted_image: ',calculate_average_2d_array(dIm))
# print('adjusted_image: ',calculate_average_2d_array(adjusted_image))
# print('circle_image: ',calculate_average_2d_array(circle_image))


# plt.figure(1)
# plt.imshow(dIm)


# plt.figure(1)
# plt.imshow(ImRoi[2])

# # plt.figure(2)
# # plt.imshow(adjusted_image)

# plt.figure(3)
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






