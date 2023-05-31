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
            if 'WaveCalibration' in f:
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
            if 'WaveCalibration' in f:
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
    
    
    def CreateDFwithAllColorAndWaveCorrection_Dic(self):
        # CorrectionCorrByColorAllWave=pd.DataFrame();
        JobNmeSORTED= self.SortJobsByTime(self.folder)
        WaveCorrJobDIC={}
        for f in list(JobNmeSORTED.values()):
            try:
              ColorDic,BarDic,BeforCorrByColor,AfterCorrByColor,CorrectionCorrByColor = self.OrgnazeDataByColorAndCorrectionState(f);
              # colNmf={};
              # for c in CorrectionCorrByColor.columns:
              #     colNmf[c]=f+' '+c;
                 
              # CorrectionCorrByColor = CorrectionCorrByColor.rename(columns=colNmf);
              WaveCorrJobDIC[f]=CorrectionCorrByColor
              # CorrectionCorrByColorAllWave=pd.concat([CorrectionCorrByColorAllWave,CorrectionCorrByColor],axis=1);
            except:
                  continue;
        return  JobNmeSORTED,WaveCorrJobDIC;
    
    
def rotate_line(x, y, theta):
    # Convert theta to radians
    theta_rad = np.radians(theta)
    
    # Create rotation matrix
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                [np.sin(theta_rad), np.cos(theta_rad)]])
    
    # Create a 2D vector from the original line coordinates
    original_vector = np.array([x, y])
    
    # Perform the rotation
    rotated_vector = np.dot(rotation_matrix, original_vector)
    
    # Extract the rotated line coordinates
    rotated_x = rotated_vector[0]
    rotated_y = rotated_vector[1]
    
    return rotated_x, rotated_y

    
class PlotPlotly():
    def __init__(self, pthF, side):
        self.pthF = pthF;
        self.side = side;
            
    def PlotRotateWave(self,clr,CorrectionCorrByColorAllWavedic, plotTitle,fileName):
        
        fig = go.Figure()
        keys=CorrectionCorrByColorAllWavedic.keys()
        Wave2Compare=CorrectionCorrByColorAllWavedic[keys[0]][clr]
        Wave2fixed=CorrectionCorrByColorAllWavedic[keys[1]][clr]

        # Add traces, one for each slider step
        # fig.add_trace(
        #     go.Scatter(x=list(self.xdb),y=list(self.ydb),line_color='red' ,
        #                 name='raw Data'))

        # fig.add_trace(
        #     go.Scatter(x=list(self.xdb),y=self.tlt,line_color='blue' ,
        #                 name='Tilt '+'Slope(x1000)='+"{0:.3f}".format(self.z[0]*1000)))
        fig.add_trace(
            go.Scatter(y=list(Wave2Compare), line_color=clr,
                       name=keys[0]+' wave 2 compare'))

        # fig.add_trace(
        #     go.Scatter(y=list(db[ColorForDisplay]),line_color=ColorForDisplay , line=dict(dash='dash'),
        #                 name=ColorForDisplay+'_After'), row=2, col=1)

        ##### Fiter Vs Befor ####
        for theta in range(35):
            rotated_x, rotated_y = rotate_line(x, Wave2fixed, theta)
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color=clr),
                    name=keys[1]+" wave rotation = " + str(theta),
                    y=rotated_y))

        # Make 10th trace visible
        fig.data[1].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": plotTitle + str(i)}],  # layout attribute
            )

            if i < len(fig.data):
                # Toggle i'th trace to "visible"
                step["args"][0]["visible"][i] = True

            # step["args"][0]["visible"][0] = True
            # step["args"][0]["visible"][1] = True

            steps.append(step)

        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Theta:"},
            pad={"t": 100},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders
        )

        fig.show()

        plot(fig, filename=fileName)

        return fig 

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

JobNmeSORTEDFront,CorrectionCorrByColorAllWaveFrontdic = CalcWave(pthF,folderWaveCalibrationFront,side,Panel).CreateDFwithAllColorAndWaveCorrection_Dic();


try:
    side='Back';

    folderWaveCalibrationBack=PreapareData(pthF).ExtractFilesFromZip()
    
    
    JobNmeSORTEDBack,CorrectionCorrByColorAllWaveBack = CalcWave(pthF,folderWaveCalibrationBack,side,Panel).CreateDFwithAllColorAndWaveCorrection();

    JobNmeSORTEDBack,CorrectionCorrByColorAllWaveBackdic = CalcWave(pthF,folderWaveCalibrationBack,side,Panel).CreateDFwithAllColorAndWaveCorrection_Dic();


except:
    1

os.chdir(pthF)
#########################PLOT########################################


#########################PLOT########################################

plt.figure()
plt.plot(CorrectionCorrByColorAllWaveFrontdic['QCS WaveCalibration_78 Archive 11-05-2023 20-55-41.zip']['Black'])

lineExmp=list[CorrectionCorrByColorAllWaveFrontdic['QCS WaveCalibration_78 Archive 11-05-2023 20-55-41.zip']['Black']]

x=list(range(len(lineExmp)))

thetaEng=list(np.asarray(range(1,11,1))*0.1)

rotated_x, rotated_y=rotate_line(x, lineExmp, 90)

plt.figure()
plt.plot(lineExmp)
for i in thetaEng:
    rotated_x, rotated_y=rotate_line(x, lineExmp, i)
    plt.plot(rotated_x,rotated_y)
    # plt.plot(rotated_y)



plt.figure()
# plt.plot(lineExmp)
for i in range(2,60,2):
    rotated_x, rotated_y=rotate_line(x, lineExmp, i)
    plt.plot(rotated_x)


plt.figure()
# plt.plot(lineExmp)
for i in range(2,35,2):
    rotated_x, rotated_y=rotate_line(x, lineExmp, i)
    # plt.plot(rotated_x,rotated_y)
    plt.plot(rotated_y,color='b')
plt.plot(rotated_y,color='r')


plt.figure()
plt.plot(lineExmp)
rotated_x, rotated_y=rotate_line(x, lineExmp, 60)
plt.plot(rotated_y)


theta=10
x=list(range(len(lineExmp)))
y=lineExmp


theta_rad = np.radians(theta)

# Create rotation matrix
rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                            [np.sin(theta_rad), np.cos(theta_rad)]])

# Create a 2D vector from the original line coordinates
original_vector = np.array([x, y])

# Perform the rotation
rotated_vector = np.dot(rotation_matrix, original_vector)

# Extract the rotated line coordinates
rotated_x = rotated_vector[0]
rotated_y = rotated_vector[1]

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
