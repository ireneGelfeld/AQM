# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:00:16 2022

@author: Ireneg
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

# import plotly.io as pio
# pio.renderers
# pio.renderers.default='browser'
#####################################Params #############################################################
#########################################################################################################
Panel = 11;
ColorForDisplay = 'Cyan'


#########################################################################################################
#########################################################################################################
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

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.subplots import make_subplots
from io import BytesIO
import zipfile 

class PreapareData():
    def __init__(self, pthF=None):
        ''' constructor ''' 
        self.pthF = pthF
        
    def UnzipFilesAndSaveToFolderList(self):
        folder = [f for f in glob.glob("**/*.zip", recursive=True)]
 
        return folder;
    
    def ExtractFilesFromZip(self):        
        os.chdir(self.pthF);
        folderZip=self.UnzipFilesAndSaveToFolderList();
        folderWaveCalibration=[]
        for f in folderZip:
            if 'Wave' in f:
                if not '\\' in f:
                    folderWaveCalibration.append(f);     
        return folderWaveCalibration;
    

class PreapareDataOLD():
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
        folderZip=self.UnzipFilesAndSaveToFolderList();
        folderSub=next(os.walk(self.pthF))[1];
        folder= list(set(folderZip + folderSub))
        folderWaveCalibration=[]
        for f in folder:
            if 'Wave' in f:
                folderWaveCalibration.append(f);     
        return folderWaveCalibration;

class CalcWave:
    def  __init__(self,pathF,folder,side,Panel): 
        self.pathF= pathF;
        self.folder = folder;
        self.side = side;
        self.Panel= Panel;
    
    def GetFileFromZip(self,zip_file_path,subdir_name_in_zip,file_name_in_zip):
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_path_in_zip = subdir_name_in_zip + "/" + file_name_in_zip
            with zip_ref.open(file_path_in_zip) as file:
                # read the contents of the file into memory
                file_content = file.read()
                
                # convert the file content to a pandas dataframe
                RawData = pd.read_csv(BytesIO(file_content),skiprows=1,header = None)
                Hder = pd.read_csv(BytesIO(file_content),index_col=0, nrows=0).columns.tolist() 

                
        return  RawData,Hder;     
    
    
    def LoadRawData(self,f):

        zip_file_path = self.pathF+f
        subdir_name_in_zip = self.side+'/'+'CorrectionOperators';
        file_name_in_zip = 'WavePrintDirectionCorrection.csv';        
       
        RawData,Hder= self.GetFileFromZip(zip_file_path, subdir_name_in_zip, file_name_in_zip)

        return  RawData,Hder;
    
    def LoadRawDataOLD(self,f):
        RawData=pd.read_csv(self.pathF+f+'/'+self.side+'/'+'CorrectionOperators/WavePrintDirectionCorrection.csv',skiprows=1,header = None);
        Hder = pd.read_csv(self.pathF+f+'/'+self.side+'/'+'CorrectionOperators/WavePrintDirectionCorrection.csv', index_col=0, nrows=0).columns.tolist()   

        return  RawData,Hder;
    
    
    def CreatBeforeAfterDFforSpecificPanel(self,Colors,RawData,ColorDic,BeforAfterCorrection):
        BeforAfterCorrByColor=pd.DataFrame();          
        for i,clr in enumerate(Colors):
            BeforAfterCorr=RawData[RawData[3] == BeforAfterCorrection][RawData[2] == self.Panel][RawData[1]== i+1].reset_index(drop=True);
            D = BeforAfterCorr.iloc[0,4:].reset_index(drop=True);
            BeforAfterCorrByColor=pd.concat([BeforAfterCorrByColor,D.rename(ColorDic[i+1])],axis=1)
        return BeforAfterCorrByColor;
    
    def OrgnazeDataByColorAndCorrectionState(self,f):
         RawData,Hder = self.LoadRawData(f);
         Colors=RawData.iloc[:,1].unique().tolist()
         ColorDic={1:'Cyan',2:'Magenta',3:'Yellow',4:'Black',5:'Orange',7:'Green',6:'Blue'}
         BarDic={4:'Cyan',6:'Magenta',8:'Yellow',2:'Black',7:'Orange',5:'Green',3:'Blue'}
         
         BeforCorrByColor =  self.CreatBeforeAfterDFforSpecificPanel(Colors,RawData,ColorDic,' Before')
         AfterCorrByColor =  self.CreatBeforeAfterDFforSpecificPanel(Colors,RawData,ColorDic,' After')
         CorrectionCorrByColor =  self.CreatBeforeAfterDFforSpecificPanel(Colors,RawData,ColorDic,'Correction')
         
         return ColorDic,BarDic,BeforCorrByColor,AfterCorrByColor,CorrectionCorrByColor;
    
    def SortJobsByTime(self,ColmnList):
        JobNmeDic={}
        for c in ColmnList:
            cc=c[:-4]
            jobNme=cc.split(' ')
            tme=jobNme[len(jobNme)-1].split('-')
            dte=jobNme[len(jobNme)-2].split('-')
            # datetime(year, month, day, hour, minute, second, microsecond)
            JobNmeDic[datetime(int(dte[len(dte)-1]), int(dte[len(dte)-2]), int(dte[len(dte)-3]),int(tme[len(tme)-3]), int(tme[len(tme)-2]), int(tme[len(tme)-1]))]=c;
        JobNmeSORTED = OrderedDict(sorted(JobNmeDic.items()))    
        return JobNmeSORTED;
    
    def CreateDFwithAllColorAndWaveCorrection(self):
        CorrectionCorrByColorAllWave=pd.DataFrame();
        JobNmeSORTED= self.SortJobsByTime(self.folder)
        for f in list(JobNmeSORTED.values()):
            try:
              ColorDic,BarDic,BeforCorrByColor,AfterCorrByColor,CorrectionCorrByColor = self.OrgnazeDataByColorAndCorrectionState(f);
              colNmf={};
              for c in CorrectionCorrByColor.columns:
                  colNmf[c]=f+' '+c;
                 
              CorrectionCorrByColor = CorrectionCorrByColor.rename(columns=colNmf);
              CorrectionCorrByColorAllWave=pd.concat([CorrectionCorrByColorAllWave,CorrectionCorrByColor],axis=1);
            except:
                  continue;
        return  JobNmeSORTED,CorrectionCorrByColorAllWave;

#################################################################################################################
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthF = filedialog.askdirectory()


pthF=pthF+'/';
side='Front';

folderWaveCalibrationFront=PreapareData(pthF).ExtractFilesFromZip()

JobNmeSORTEDFront,CorrectionCorrByColorAllWaveFront = CalcWave(pthF,folderWaveCalibrationFront,side,Panel).CreateDFwithAllColorAndWaveCorrection();



try:
    side='Back';

    folderWaveCalibrationBack=PreapareData(pthF).ExtractFilesFromZip()
    
    
    JobNmeSORTEDBack,CorrectionCorrByColorAllWaveBack = CalcWave(pthF,folderWaveCalibrationBack,side,Panel).CreateDFwithAllColorAndWaveCorrection();
except:
    1

os.chdir(pthF)

#########################PLOT########################################

fig = go.Figure()

db=CorrectionCorrByColorAllWaveFront



col=(list(db.columns))
rnge=range(len(col))

for i in rnge:
# for i in rnge:
    clr=col[i].split(' ')[len(col[i].split(' '))-1]
    fig.add_trace(go.Scatter(y=list(db[col[i]]),line_color=clr,
                name=col[i]))
    if not clr == ColorForDisplay:
         fig.data[i].visible = 'legendonly';
        


    
            


fig.update_layout(title='WAVE CALIBRATION Front')


fig.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)

now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig,auto_play=True,filename="WAVE CALIBRATION Front"+ dt_string +".html")  
#plot(fig_back,filename="AQM-Back.html")  
fig.show()



try:
    fig1 = go.Figure()

    db=CorrectionCorrByColorAllWaveBack
    
    
    
    col=(list(db.columns))
    rnge=range(len(col))
    
    for i in rnge:
    # for i in rnge:
        clr=col[i].split(' ')[len(col[i].split(' '))-1]
        fig1.add_trace(go.Scatter(y=list(db[col[i]]),line_color=clr,
                    name=col[i]))
        if not clr == ColorForDisplay:
             fig1.data[i].visible = 'legendonly';
            
    
    
        
                
    
    
    fig1.update_layout(title='WAVE CALIBRATION Back')
    
    
    fig1.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    plot(fig1,auto_play=True,filename="WAVE CALIBRATION Back"+ dt_string +".html")  
    #plot(fig_back,filename="AQM-Back.html")  
    fig1.show()

except:
    1



#################################################################################


# CorrectionCorrByColorAllWave=pd.DataFrame();
# JobNmeSORTED= CalcWave(pthF,folderWaveCalibrationFront,side,Panel).SortJobsByTime(CalcWave(pthF,folderWaveCalibrationFront,side,Panel).folder)
# for f in list(JobNmeSORTED.values()):
#     try:
#       ColorDic,BarDic,BeforCorrByColor,AfterCorrByColor,CorrectionCorrByColor = CalcWave(pthF,folderWaveCalibrationFront,side,Panel).OrgnazeDataByColorAndCorrectionState(f);
#       colNmf={};
#       for c in CorrectionCorrByColor.columns:
#           colNmf[c]=f+' '+c;
         
#       CorrectionCorrByColor = CorrectionCorrByColor.rename(columns=colNmf);
#       CorrectionCorrByColorAllWave=pd.concat([CorrectionCorrByColorAllWave,CorrectionCorrByColor],axis=1);
#     except:
#           continue;