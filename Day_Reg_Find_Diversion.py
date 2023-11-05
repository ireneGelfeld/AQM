# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:30:46 2021

@author: Ireneg
"""
# from IPython import get_ipython
# get_ipython().magic('reset -sf')

# import plotly.io as pio
# pio.renderers
# pio.renderers.default='browser'
##############################################################################
global thresholdCorr,NumOfConsElements;

thresholdCorr = 30 # In [um]- The threshold between Consecutive corrections
NumOfConsElements=4 # Number of consecutive corrections with a difference above thresholdCorr




##############################################################################
import os


import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import scipy.io
from datetime import datetime
import glob
from zipfile import ZipFile 
from pathlib import Path
from collections import OrderedDict
from io import BytesIO
import zipfile 

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.subplots import make_subplots
import re
from scipy.signal import savgol_filter


def remove_decimal_numbers(string):
    pattern = r'\d+\.\d+'
    result = re.sub(pattern, '', string)
    return result


def add_zero_to_timestamp(timestamp):
    date, time = timestamp.split(' ')
    time_components = time.split(':')
    corrected_time_components = []

    for component in time_components:
        if len(component) == 1:
            corrected_time_components.append('0' + component)
        else:
            corrected_time_components.append(component)

    corrected_time = ':'.join(corrected_time_components)
    corrected_timestamp = f"{date} {corrected_time}"
    return corrected_timestamp

def Create_Blanket_ReplacementList(BlanketRep):
    column_name = 'ok'
    filter_value = 'ok'

    # Create a Series filtering out the specified value
    BalnketRepTime = BlanketRep.loc[BlanketRep[column_name] != filter_value, column_name].reset_index(drop=True)
    BlanketRepList=[]
    for itm in BalnketRepTime:
        if '$' in itm:
            doubleRep=itm.split('$')
            for doubleRepItm in doubleRep:
                doubleRepItm=remove_decimal_numbers(doubleRepItm)[:-1]
                doubleRepItm=add_zero_to_timestamp(doubleRepItm)
                newString=doubleRepItm.replace('/','-').replace(':','-')
                BlanketRepList.append('BlanketReplacment '+newString+'    ')
        else:
                itm=remove_decimal_numbers(itm)[:-1]
                itm=add_zero_to_timestamp(itm)
                newString=itm.replace('/','-').replace(':','-')
                BlanketRepList.append('BlanketReplacment '+newString+'    ')

    return BlanketRepList


class PreapareDataExtractZip():
    def __init__(self, pthF=None):
        ''' constructor ''' 
        self.pthF = pthF
        
    def UnzipFilesAndSaveToFolderList(self):
        ZpFile = [f for f in glob.glob("**/*.zip", recursive=True)]
        folder = []
        # specifying the zip file name 
        for n in range(len(ZpFile)):
            Fname=ZpFile[n]
            folder.append(Fname[0:len(ZpFile[n])-4])
            if not os.path.isdir(Fname[0:len(ZpFile[n])-4]):
                with ZipFile(ZpFile[n], 'r') as zip: 
                   zip.extractall(Fname[0:len(ZpFile[n])-4])
        return folder;
    
    def ExtractFilesFromZip(self):        
        os.chdir(self.pthF);
        folder=self.UnzipFilesAndSaveToFolderList();
        return folder;

class PreapareData():
    def __init__(self, pthF=None):
        ''' constructor ''' 
        self.pthF = pthF
        
    def ZipFilesFolderList(self):
        folder = [f for f in glob.glob("**/*.zip", recursive=True)]
        return folder;
    
    def ExtractFilesFromZip(self):        
        os.chdir(self.pthF);
        folder=self.ZipFilesFolderList();
        return folder;


class CorrectionAndDiversion():
    def __init__(self, pthF,fldrs,side):
        ''' constructor ''' 
        self.pthF = pthF;
        self.fldrs = fldrs;
        self.side = side;
   
        

    def GetFileFromZip(self,zip_file_path,subdir_name_in_zip,file_name_in_zip):
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_path_in_zip = subdir_name_in_zip + "/" + file_name_in_zip
            with zip_ref.open(file_path_in_zip) as file:
                # read the contents of the file into memory
                file_content = file.read()
                
                # convert the file content to a pandas dataframe
                df = pd.read_csv(BytesIO(file_content))
        return  df;          
        
 
    
    def ReadImagePlacmentAndCorrectionData(self):
        
        
        self.correctionDic={}
        fcorr='PanelCorrection.csv'

        for f in self.fldrs:
            PanelCorrection=pd.DataFrame()
      
            try:
               # os.chdir(pthCrr);
               # PanelCorrection=pd.read_csv(fcorr)
               
               zip_file_path = self.pthF+'/'+f
               subdir_name_in_zip = self.side+'/CorrectionOperators';
               file_name_in_zip = fcorr;
               
               PanelCorrection=self.GetFileFromZip(zip_file_path,subdir_name_in_zip,file_name_in_zip);
               self.correctionDic[f]=PanelCorrection
               
            except:
                1
      
            
    
    def CalcDivergincePerColor(self):
        
        self.diverdingJobs={}
        for key,value in self.correctionDic.items():
            colorList= value['Color'].unique();
            divertingcolors={}
            for color in colorList:
                df=value[value['Color']==color].reset_index(drop=True)
                CorrValue={}
                CorrValue[0]=df['C2C_X Correction'][0]
                divergins=[]
                for index in df.index[1:]:
                    
                   
                    if index-1 in CorrValue.keys():
                        if  (df['C2C_X Correction'][index] >= 0) == (CorrValue[index-1] >= 0):
                            if (abs(df['C2C_X Correction'][index]) -abs( CorrValue[index-1]))> thresholdCorr:
                                if len(CorrValue.keys())<NumOfConsElements:
                                    CorrValue[index]=df['C2C_X Correction'][index]
                                    continue;
                           
                                    
                    if   len(CorrValue.keys())>NumOfConsElements-1:
                        divergins.append(index-1)
                    CorrValue={}
                    CorrValue[index]=df['C2C_X Correction'][index]
                    
                divertingcolors[color]=divergins
                
            self.diverdingJobs[key]=divertingcolors
            
    def saveDivergingJobs(self):
        
        os.chdir(self.pthF)
        
        df = pd.DataFrame(self.diverdingJobs).T
        
        df.to_csv('DivergingJobs.csv')
        
        return df
                

#################################################



      
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()

pthF = filedialog.askdirectory()


pthF=pthF+'/';



folder=PreapareData(pthF).ExtractFilesFromZip();   


fldrs=folder
side =  'Front'
  
    
correctionAndDiversionFRONT  = CorrectionAndDiversion(pthF,fldrs,side)

    
correctionAndDiversionFRONT.ReadImagePlacmentAndCorrectionData()
   
correctionAndDiversionFRONT.CalcDivergincePerColor()

dfFront=     correctionAndDiversionFRONT.saveDivergingJobs()
    
    
    

correctionDic=correctionAndDiversionFRONT.correctionDic


######################







