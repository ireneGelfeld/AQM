# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:15:47 2023

@author: Ireneg
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
import os
import pickle



# import plotly.io as pio
# pio.renderers
# pio.renderers.default='browser'

# %matplotlib
############################################################################
global ColorDic,CircleArea,ColorDicNum


CircleArea=15

ColorDic={0:'Magenta',1:'Yellow',2:'Blue',3:'Orange',4:'Cyan',5:'Green',6:'Black'}


ColorDicNum={'Magenta':0,'Yellow':1,'Blue':2,'Orange':3,'Cyan':4,'Green':5,'Black':6}




############################################################################



def sorted_indices(arr):
    # Enumerate the list to keep track of original indices
    enumerated_list = list(enumerate(arr))
    # Sort the enumerated list based on the values
    sorted_enumerated_list = sorted(enumerated_list, key=lambda x: x[1])
    # Extract and return the sorted indices
    sorted_indices_list = [index for index, _ in sorted_enumerated_list]
    return sorted_indices_list



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
    

    def CheckValidity(self,current_line_sorted):
        
        
        valid=True

        if len(current_line_sorted.keys())!=7:
            valid=False

        for key in current_line_sorted.keys():
            x = self.CreatColorList_x(key,current_line_sorted,0)
            if len(x)<400 or len(x)>460:
                valid=False
            
        return valid

 
    def CalcorColorMat(self,ImRoi,StartRoiCoed):
       
           
       ClrDF_raw={}
       ClrDF_rawXY={}

       current_line_sortedDIC={}
       k=0

       for i in range(len(ImRoi.keys())):
           
           
            circle_centers,gray,circle_image,edges=self.find_circles(ImRoi[i],CircleArea);
            current_line_sorted=self.SortCircle_Coord(circle_centers);

            if len(current_line_sorted.keys()) > 7:
                del current_line_sorted[7]
                
                
            if not self.CheckValidity(current_line_sorted):
                continue
        
        
        

            
            current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
            
            current_line_sorted = self.filterCircles(current_line_sorted)
            
            current_line_sorted,cntMissingCircles=self.AddMissingCircles(current_line_sorted);
 
            
            ClrDF=pd.DataFrame()
            ClrDF_x=pd.DataFrame()
 
            tmp_df=pd.DataFrame()
            tmp_df_x=pd.DataFrame()
            for Key,Value in ColorDic.items():
                tmp_df = pd.DataFrame({Value: self.CreatColorList(Key,current_line_sorted)})
                ClrDF = pd.concat([ClrDF, tmp_df], axis=1)
                ClrDF_x = pd.concat([ClrDF_x, tmp_df], axis=1)
                tmp_df_x = pd.DataFrame({Value+'_x': self.CreatColorList_x(Key,current_line_sorted,StartRoiCoed[i])})
                ClrDF_x = pd.concat([ClrDF_x, tmp_df_x], axis=1)
  
               
           
            
            current_line_sortedDIC[k] = current_line_sorted
            ClrDF_raw[k]=ClrDF
            ClrDF_rawXY[k]=ClrDF_x
            k=k+1

           
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

            
            

        
        

        
        
        
        
        
        
        
        
        
        
        
        
#############################################################################
import sys

if len(sys.argv) < 2:
    print("Usage: python script.py <folder_name>")
    sys.exit(1)

folder_name = sys.argv[1]

print("Folder name:", folder_name)

sInput=folder_name

os.chdir(sInput)
# os.chdir(r'D:\BTDencoder\B1')


sInputList=[folder for folder in os.listdir(sInput) if os.path.isdir(os.path.join(sInput, folder)) and ('-' in folder)]

# os.chdir(r'D:\BTDencoder\B1')

listLastDig=[int(ll.split('-')[2])   for ll in sInputList]


sorted_indices_list = sorted_indices(listLastDig)

sInputListSORTED=[]

for ll in  sorted_indices_list:
    sInputListSORTED.append(sInputList[ll])

fileNME='\\FullImage.bmp'


ClrDF_rawXYdic={}

for Pnl in sInputListSORTED:

    ImRoi,StartRoiCoed= Circles(sInput+'\\'+Pnl+fileNME).loadImage();
    
    gray,circle_image,edges,ClrDF_raw, ClrDF_rawXY,current_line_sortedDIC= Circles(sInput+'\\'+Pnl+fileNME).CalcorColorMat(ImRoi,StartRoiCoed)

    ClrDF_rawXYdic[Pnl]=ClrDF_rawXY
    
    print(sInput+'\\'+Pnl+fileNME)





path_ClrDF_rawXYdic=sInput+'\ClrDF_rawXYdic.pkl'
with open(path_ClrDF_rawXYdic, 'wb') as f:
    pickle.dump(ClrDF_rawXYdic, f)


# file_path =path_ClrDF_rawXYdic

# # Open the file in binary read mode
# with open(file_path, 'rb') as file:
#     # Unpickle the object from the file
#     ClrDF_rawXYdicLoaded = pickle.load(file)

# ImRoi,StartRoiCoed= Circles(pth).loadImage();

# gray,circle_image,edges,ClrDF_raw, ClrDF_rawXY18,current_line_sortedDIC = Circles(pth).CalcorColorMat(ImRoi,StartRoiCoed)
        
        
        



# ClrDF_raw={}
# ClrDF_rawXY={}

# current_line_sortedDIC={}
# k=0

# for i in range(len(ImRoi.keys())):

    
#     circle_centers,gray,circle_image,edges=Circles(sInput+'\\'+Pnl+fileNME).find_circles(ImRoi[i],15);
#     print('i='+str(i))
    
    
#     current_line_sorted=Circles(sInput+'\\'+Pnl+fileNME).SortCircle_Coord(circle_centers);
    
    
#     if not Circles(sInput+'\\'+Pnl+fileNME).CheckValidity(current_line_sorted):
#         continue



#     if len(current_line_sorted.keys()) > 7:
#         del current_line_sorted[7]
    
#     current_line_sorted,cntMissingCircles=Circles(sInput+'\\'+Pnl+fileNME).AddMissingCircles(current_line_sorted);
    
#     current_line_sorted =Circles(sInput+'\\'+Pnl+fileNME).filterCircles(current_line_sorted)
    
#     current_line_sorted,cntMissingCircles=Circles(sInput+'\\'+Pnl+fileNME).AddMissingCircles(current_line_sorted);
 
    
#     ClrDF=pd.DataFrame()
#     ClrDF_x=pd.DataFrame()
 
#     tmp_df=pd.DataFrame()
#     tmp_df_xy=pd.DataFrame()
#     for Key,Value in ColorDic.items():
#         tmp_df = pd.DataFrame({Value: Circles(sInput+'\\'+Pnl+fileNME).CreatColorList(Key,current_line_sorted)})
#         ClrDF = pd.concat([ClrDF, tmp_df], axis=1)
#         ClrDF_x = pd.concat([ClrDF_x, tmp_df], axis=1)
#         tmp_df_x = pd.DataFrame({Value+'_x': Circles(sInput+'\\'+Pnl+fileNME).CreatColorList_x(Key,current_line_sorted,StartRoiCoed[i])})
#         ClrDF_x = pd.concat([ClrDF_x, tmp_df_x], axis=1)

   
        
    
     
#     current_line_sortedDIC[k] = current_line_sorted
#     ClrDF_raw[k]=ClrDF
#     ClrDF_rawXY[k]=ClrDF_x
#     print(k)
#     k=k+1

    





# adjusted_image = np.where(gray > 220, 220, np.where(gray < 50, 50, gray))

# dIm=circle_image-adjusted_image


# plt.figure(1)
# plt.imshow(dIm)



# plt.figure(2)
# plt.imshow(adjusted_image)

# plt.figure(3)
# plt.imshow(circle_image)







# plt.figure(0)
# plt.plot(x)

        