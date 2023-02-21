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


class plotPlotly(CIScurveFromImage):
   def __init__(self,ImageGL,plotTitle,fileName,RecDimX, RecDimY):
      super().__init__(ImageGL)

      self.plotTitle=plotTitle;
      self.fileName=fileName;
      self.RecDimX=RecDimX
      self.RecDimY=RecDimY
      


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




















from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog

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

figCIScalc=plotPlotly(ImageGL,plotTitle,fileName,RecDimX, RecDimY).PlotCIS();

# root = Tk()
# root.withdraw()
T_Lum = simpledialog.askstring("Input", "Enter T_Lum value:", parent=root)
RawData=pd.DataFrame();

max_val,max_index = CIScurveFromImage(ImageGL).AplyFilters(float(T_Lum)+0.01, RecDimX, RecDimY)

RawData['Value']=list(max_index)

RawData.to_csv(pth4save+'RawData.csv',header=None)


# T_Lum=0.7
# I2 = ImageGL
# I2 = cv2.convertScaleAbs(I2)
# bw = cv2.threshold(I2, 255*T_Lum, 255, cv2.THRESH_BINARY)[1]
# BWdfill = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 100)))
# diff_bw = np.diff(BWdfill, axis=0)



# # diff_bw = np.diff(bw, axis=0)



# max_val=[]
# max_index=[]

# for i in range(diff_bw.shape[1]):
#     max_val.append(np.amax(diff_bw[:,i]))
#     max_index.append(np.argmax(diff_bw[:,i]))
   

plt.figure()
plt.plot(max_index)
plt.title(" T_Lum value: "+T_Lum)
         
       

        
# os.chdir(r'D:\B2\CIS_New')
# img = cv2.imread('82.bmp')



# # Crop the first upper corner
# cropped_img1 = img[:int(img.shape[0]/20), :int(img.shape[1]/20)]

# # Crop the second upper corner
# cropped_img2 = img[:int(img.shape[0]/20), int(img.shape[1])-int(img.shape[1]/20):int(img.shape[1])]

# # Resize the cropped images to 500 x 500
# # cropped_img1 = cv2.resize(cropped_img1, (500, 500))
# # cropped_img2 = cv2.resize(cropped_img2, (500, 500))

# # Get the user's selected coordinates from each cropped image
# point1 = cv2.selectROI("LEFT- choose above the upper strip line and press SPACEBAR", cropped_img1)
# point2 = cv2.selectROI("RIGHT- choose the center of the strip and press SPACEBAR", cropped_img2)



# # Convert the coordinates from the cropped images to the original image
# point01 = (point1[0], point1[1])
# point02 = (int(img.shape[1])-(int(cropped_img2.shape[1])-point2[0]), point2[1])

# print("Selected point 1 in the original image:", point01)
# print("Selected point 2 in the original image:", point02)

# cv2.destroyAllWindows()

# x1,y1=point01
# x2,y2=point02

# cropped_img = img[y1:y2, :int(img.shape[1])]


# # cv2.namedWindow("Cropped Image", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("Cropped Image", 500, 500)
# if not os.path.exists("Cropped images"):
#     os.makedirs("Cropped images")

# cv2.imwrite("Cropped images/1.bmp", cropped_img)            
            
            
            
            
            
            
