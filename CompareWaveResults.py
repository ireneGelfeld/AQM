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
global StartCycle4Avr,Panel,ColorForDisplay,Cycle2Display

Panel = 6;
ColorForDisplay = 'Cyan'
StartCycle4Avr = 2; # Start averaging for all plots defult = 2
Cycle2Display = 2

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


class PreapareData():
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
    def  __init__(self,pthF,folder,side,Panel): 
        self.pthF= pthF;
        self.folder = folder;
        self.side = side;
        self.Panel= Panel;
        

    
    def LoadRawData_CorrOP(self,f):
        RawData=pd.read_csv(self.pthF+f+'/'+self.side+'/'+'CorrectionOperators/WavePrintDirectionCorrection.csv',skiprows=1,header = None);
        Hder = pd.read_csv(self.pthF+f+'/'+self.side+'/'+'CorrectionOperators/WavePrintDirectionCorrection.csv', index_col=0, nrows=0).columns.tolist()   

        return  RawData,Hder;
    
    
    def CreatBeforeAfterDFforSpecificPanel(self,Colors,RawData,ColorDic,BeforAfterCorrection):
        BeforAfterCorrByColor=pd.DataFrame();          
        for i,clr in enumerate(Colors):
            BeforAfterCorr=RawData[RawData[3] == BeforAfterCorrection][RawData[2] == self.Panel][RawData[1]== i+1].reset_index(drop=True);
            D = BeforAfterCorr.iloc[0,4:].reset_index(drop=True);
            BeforAfterCorrByColor=pd.concat([BeforAfterCorrByColor,D.rename(ColorDic[i+1])],axis=1)
        return BeforAfterCorrByColor;
    
    def OrgnazeDataByColorAndCorrectionState(self,f):
         RawData,Hder = self.LoadRawData_CorrOP(f);
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
            try:
                RawData,Hder = self.LoadRawData_CorrOP(c);
                jobNme=c.split(' ')
                tme=jobNme[len(jobNme)-1].split('-')
                dte=jobNme[len(jobNme)-2].split('-')
                # datetime(year, month, day, hour, minute, second, microsecond)
                JobNmeDic[datetime(int(dte[len(dte)-1]), int(dte[len(dte)-2]), int(dte[len(dte)-3]),int(tme[len(tme)-3]), int(tme[len(tme)-2]), int(tme[len(tme)-1]))]=c;
            except:
                continue;
        JobNmeSORTED = OrderedDict(sorted(JobNmeDic.items()))    
        return JobNmeSORTED;
    
    def CreateDFwithAllColorAndWaveCorrection(self):
        CorrectionCorrByColorAllWave=pd.DataFrame();
        BeforeCorrByColorAllWave=pd.DataFrame();
        JobNmeSORTED= self.SortJobsByTime(self.folder)
        for f in list(JobNmeSORTED.values()):
            try:
              ColorDic,BarDic,BeforCorrByColor,AfterCorrByColor,CorrectionCorrByColor = self.OrgnazeDataByColorAndCorrectionState(f);
              colNmf={};
              for c in CorrectionCorrByColor.columns:
                  colNmf[c]=f+' '+c;
                 
              CorrectionCorrByColor = CorrectionCorrByColor.rename(columns=colNmf);
              BeforCorrByColor = BeforCorrByColor.rename(columns=colNmf);
              
              CorrectionCorrByColorAllWave=pd.concat([CorrectionCorrByColorAllWave,CorrectionCorrByColor],axis=1);
              BeforeCorrByColorAllWave=pd.concat([BeforeCorrByColorAllWave,BeforCorrByColor],axis=1);

              # CorrectionCorrByColorAllWave=pd.concat([CorrectionCorrByColorAllWave,BeforCorrByColor],axis=1);

            except:
                  continue;
        return  JobNmeSORTED,BeforeCorrByColorAllWave;
    



class CalcWaveFromRawData(CalcWave):
    def  __init__(self, pthF,folder,side,Panel): 
        super().__init__(pthF,folder,side,Panel)
 
        

    
    def LoadRawData(self,f):
        RawData=pd.read_csv(self.pthF+f+'/'+self.side+'/'+'RawResults/WavePrintDirection.csv');

        return  RawData;
    
    def getColors(self,f):
        RawData= self.LoadRawData(f);
        ColorList=RawData.iloc[:,7].unique().tolist();
        return ColorList
    
    def getNumberOfFlats(self,f):
        RawData= self.LoadRawData(f);
        FlatList=RawData.iloc[:,1].unique().tolist();
        return FlatList
    
    def FilterRawData(self,ColorForDisplay,f):
        RawData= self.LoadRawData(f);
        
        # DataSec=RawData[RawData['Overall Status']=='Success'].reset_index(drop=True);
        
        DataSec=RawData;


        DataSecPrintDirc=DataSec[DataSec['Direction Type ']=='Print Direction']
        
        DataSecPrintDircPanel=DataSecPrintDirc[DataSecPrintDirc['Panel Id']==self.Panel]
        
        DataSecPrintDircPanelColor=DataSecPrintDircPanel[DataSecPrintDircPanel[' Seperation']==ColorForDisplay].reset_index(drop=True)
        
        col=list(DataSecPrintDircPanelColor.columns)
        
        cutCols=col[12:396]
        
        DataSecPrintDircPanelColorCUT=DataSecPrintDircPanelColor[cutCols];
        LocatorIndex= DataSec['Locator Index'][int(len(DataSec['Locator Index'])/2)];
        
        return LocatorIndex,DataSecPrintDircPanelColorCUT,cutCols;
    
    
    def ArrangeRawDataForAnalize(self,ColorForDisplay,f):
       
        LocatorIndex,DataSecPrintDircPanelColorCUT,cutCols=self.FilterRawData(ColorForDisplay,f);
        WaveRaw=pd.DataFrame();

        for i in range(len(DataSecPrintDircPanelColorCUT[cutCols[0]])):
            l=list(DataSecPrintDircPanelColorCUT.loc[i,:])
            tmpDF=pd.DataFrame();
            while (1):
                tmp=next((j for j, x in enumerate(l) if not isinstance(x, float)), 'DONE');
                if  tmp == 'DONE':
                    break;
                else:
                   if  l[tmp].replace('.', '', 1).replace('-', '').isdigit():
                       l[tmp]=float(l[tmp]);
                   else: 
                       if l[tmp] == 'NotFound':
                           if isinstance(l[tmp+1], float):
                               l[tmp]=l[tmp+1]
                           else:
                               if l[tmp+1].replace('.', '', 1).replace('-', '').isdigit():
                                  l[tmp]=float(l[tmp+1])
                               else:
                                  break;
            if not tmp == 'DONE':
               WaveRaw=pd.concat([WaveRaw,pd.DataFrame(l[0:tmp-1])],axis=1).rename(columns={0:i+1}) 
        return  WaveRaw;  

  
    def CreateDicOfWaveRawData(self,f):
        ColorList=self.getColors(f);
        WaveRawDataDic={};
        for ColorForDisplay in ColorList:  
            tmp=self.ArrangeRawDataForAnalize(ColorForDisplay,f);
            tmp=pd.concat([tmp,tmp.loc[:,StartCycle4Avr:].mean(axis=1)],axis=1).rename(columns={0:'Mean'})
            WaveRawDataDic[ColorForDisplay]=tmp;
        return WaveRawDataDic;
    
    def CompareWaveDataToEarliestWave(self):    
               
        JobNmeSORTED,CorrectionCorrByColorAllWave= self.CreateDFwithAllColorAndWaveCorrection();
        JobList=list(JobNmeSORTED.values());
        WaveRawData_sub_FirstCorr={}
        for JobName in JobList[1:]:
            WaveRawDataDic=self.CreateDicOfWaveRawData(JobName);
            ColorList=self.getColors(JobName);
            for clr in ColorList:
                for col in WaveRawDataDic[clr].columns:
                    # StpageAVR=np.mean([WaveRawDataDic[clr][col][0],WaveRawDataDic[clr][col][len(WaveRawDataDic[clr][col])-1]])
                    WaveRawDataDic[clr][col]=WaveRawDataDic[clr][col]+CorrectionCorrByColorAllWave[JobList[0]+' '+clr][:len(WaveRawDataDic[clr][col])];
                    StpageAVR=np.mean([WaveRawDataDic[clr][col][0],WaveRawDataDic[clr][col][len(WaveRawDataDic[clr][col])-1]])
                    WaveRawDataDic[clr][col]=WaveRawDataDic[clr][col]-StpageAVR;

            WaveRawData_sub_FirstCorr[JobName]=WaveRawDataDic;
       
        return WaveRawData_sub_FirstCorr,ColorList,JobNmeSORTED;
    
    def CalcC2CPerCycle(self):
        
        WaveRawData_sub_FirstCorr,ColorList,JobNmeSORTED= self.CompareWaveDataToEarliestWave();
        C2C_FromWaveDifferance={}
        for jb in list(WaveRawData_sub_FirstCorr.keys()):
            Df=pd.DataFrame();
            for col in WaveRawData_sub_FirstCorr[jb][ColorList[0]].columns:
                if col == 'Mean':
                    continue;
                tmp=[]    
                for i in range(len(WaveRawData_sub_FirstCorr[jb][ColorList[0]][1])):
                    l=[]
                    for clr in ColorList:
                        l.append(WaveRawData_sub_FirstCorr[jb][clr][col][i])
                    tmp.append(np.max(l)-np.min(l));
                Df=pd.concat([Df,pd.Series(tmp)],axis=1).rename(columns={0:col})
                
            C2C_FromWaveDifferance[jb]=Df        
        
        return C2C_FromWaveDifferance;
               

class PlotGraphPlotly:       
    def  __init__(self,db,PlotTitle,fileName,ColorList): 
        self.db=db;
        self.PlotTitle= PlotTitle;
        self.fileName = fileName;
        self.ColorList=ColorList;
 
    def PlotWaveCalibration(self):
        
        fig = go.Figure()
        col=(list(self.db.columns))
        rnge=range(len(col))
        
        for i in rnge:
        # for i in rnge:
            clr=col[i].split(' ')[len(col[i].split(' '))-1]
            fig.add_trace(go.Scatter(y=list(db[col[i]]),line_color=clr,
                        name=col[i]))
            if not clr == ColorForDisplay:
                 fig.data[i].visible = 'legendonly';
                
        
        fig.update_layout(title=PlotTitle)
        
        
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        
  
        plot(fig,auto_play=True,filename=fileName+ ".html")  
        fig.show()
        return fig;
    
    def PlotWaveDataVSEarliestWave(self):
  
        fig = go.Figure()
        
        
        for jb in self.db.keys():
        # for i in rnge:
            for clr in self.ColorList:
                try: 
                    for col in self.db[jb][clr].columns:
                        if col=='Mean':
                            continue;
                        lineColor = clr;    
                        if lineColor=='Yellow':
                            lineColor='gold';  
                        fig.add_trace(go.Scatter(y=list(self.db[jb][clr][col]),line_color=lineColor,
                                    name=jb+' '+clr+' '+' Cycle='+str(col)))
                        # if not clr == ColorForDisplay and not col == Cycle2Display:
                        if  not col == Cycle2Display:

                             fig.data[len(fig.data)-1].visible = 'legendonly';
                except:
                    continue;
                
        
        fig.update_layout(title=PlotTitle)
        
        
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        
  
        plot(fig,auto_play=True,filename=fileName+ ".html")  
        fig.show()
        return fig;
     
    def PlotWaveDataVSEarliestWave(self):
  
        fig = go.Figure()
        
        
        for jb in self.db.keys():
        # for i in rnge:
            for clr in self.ColorList:
                try: 
                    for col in self.db[jb][clr].columns:
                        if col=='Mean':
                            continue;
                        lineColor = clr;    
                        if lineColor=='Yellow':
                            lineColor='gold';  
                        fig.add_trace(go.Scatter(y=list(self.db[jb][clr][col]),line_color=lineColor,
                                    name=jb+' '+clr+' '+' Cycle='+str(col)))
                        # if not clr == ColorForDisplay and not col == Cycle2Display:
                        if  not col == Cycle2Display:

                             fig.data[len(fig.data)-1].visible = 'legendonly';
                except:
                    continue;
                
        
        fig.update_layout(title=PlotTitle)
        
        
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        
  
        plot(fig,auto_play=True,filename=fileName+ ".html")  
        fig.show()
        return fig;     


    def PlotC2CDiff(self):
      
            fig = go.Figure()
            
            
            for jb in self.db.keys():
            # for i in rnge:
                
                    try: 
                        for col in self.db[jb].columns:
                         
                            fig.add_trace(go.Scatter(y=list(self.db[jb][col]),
                                        name=jb+' C2C '+' Cycle='+str(col)))
                            # if not clr == ColorForDisplay and not col == Cycle2Display:
                            if  not col == Cycle2Display:
    
                                 fig.data[len(fig.data)-1].visible = 'legendonly';
                    except:
                        continue;
                    
            
            fig.update_layout(title=PlotTitle)
            
            
            fig.update_layout(
                hoverlabel=dict(
                    namelength=-1
                )
            )
            
      
            plot(fig,auto_play=True,filename=fileName+ ".html")  
            fig.show()
            return fig;  
           



#################################################################################################################
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthF = filedialog.askdirectory()


pthF=pthF+'/';
side='Front';

folderWaveCalibrationFront=PreapareData(pthF).ExtractFilesFromZip()


# JobNmeSORTEDFront,CorrectionCorrByColorAllWaveFront = CalcWave(pthF,folderWaveCalibrationFront,side,Panel).CreateDFwithAllColorAndWaveCorrection();


# try:
#     side='Back';

#     folderWaveCalibrationBack=PreapareData(pthF).ExtractFilesFromZip()
    
    
#     JobNmeSORTEDBack,CorrectionCorrByColorAllWaveBack = CalcWave(pthF,folderWaveCalibrationBack,side,Panel).CreateDFwithAllColorAndWaveCorrection();
# except:
#     1

os.chdir(pthF)

side='Front';

WaveRawData_sub_FirstCorrFRONT,ColorList,JobNmeSORTED= CalcWaveFromRawData(pthF,folderWaveCalibrationFront,side,Panel).CompareWaveDataToEarliestWave()
C2C_FromWaveDifferanceFRONT= CalcWaveFromRawData(pthF,folderWaveCalibrationFront,side,Panel).CalcC2CPerCycle()

try:
   side='Back';
   WaveRawData_sub_FirstCorrBACK,ColorList,JobNmeSORTED= CalcWaveFromRawData(pthF,folderWaveCalibrationFront,side,Panel).CompareWaveDataToEarliestWave()
   C2C_FromWaveDifferanceBACK= CalcWaveFromRawData(pthF,folderWaveCalibrationFront,side,Panel).CalcC2CPerCycle()

except:
    1

#######################################################
# JobNmeSORTED,BeforeCorrByColorAllWave
# CorrectionCorrByColorAllWave=pd.DataFrame();
# BeforeCorrByColorAllWave=pd.DataFrame();
# JobNmeSORTED= CalcWaveFromRawData(pthF,folderWaveCalibrationFront,side,Panel).SortJobsByTime(CalcWaveFromRawData(pthF,folderWaveCalibrationFront,side,Panel).folder)
# for f in list(JobNmeSORTED.values()):
#     try:
#       ColorDic,BarDic,BeforCorrByColor,AfterCorrByColor,CorrectionCorrByColor = CalcWaveFromRawData(pthF,folderWaveCalibrationFront,side,Panel).OrgnazeDataByColorAndCorrectionState(f);
#       colNmf={};
#       for c in CorrectionCorrByColor.columns:
#           colNmf[c]=f+' '+c;
         
#       CorrectionCorrByColor = CorrectionCorrByColor.rename(columns=colNmf);
#       BeforCorrByColor = BeforCorrByColor.rename(columns=colNmf);
      
#       CorrectionCorrByColorAllWave=pd.concat([CorrectionCorrByColorAllWave,CorrectionCorrByColor],axis=1);
#       BeforeCorrByColorAllWave=pd.concat([BeforeCorrByColorAllWave,BeforCorrByColor],axis=1);

#       # CorrectionCorrByColorAllWave=pd.concat([CorrectionCorrByColorAllWave,BeforCorrByColor],axis=1);

#     except:
#           continue;


# JobNmeSORTED,CorrectionCorrByColorAllWave= CalcWaveFromRawData(pthF,folderWaveCalibrationFront,side,Panel).CreateDFwithAllColorAndWaveCorrection();
# JobList=list(JobNmeSORTED.values());
# WaveRawData_sub_FirstCorr={}
# for JobName in JobList[1:]:
#     WaveRawDataDic=CalcWaveFromRawData(pthF,folderWaveCalibrationFront,side,Panel).CreateDicOfWaveRawData(JobName);
#     ColorList=CalcWaveFromRawData(pthF,folderWaveCalibrationFront,side,Panel).getColors(JobName);
#     for clr in ColorList:
#         for col in WaveRawDataDic[clr].columns:
#             WaveRawDataDic[clr][col]=WaveRawDataDic[clr][col]+CorrectionCorrByColorAllWave[JobList[0]+' '+clr][:len(WaveRawDataDic[clr][col])];
#             StpageAVR=np.mean([WaveRawDataDic[clr][col][0],WaveRawDataDic[clr][col][len(WaveRawDataDic[clr][col])-1]])
#             WaveRawDataDic[clr][col]=WaveRawDataDic[clr][col]-StpageAVR;

#     WaveRawData_sub_FirstCorr[JobName]=WaveRawDataDic;







# plt.Figure()
# plt.plot(WaveRawDataDic[clr][col])

#########################PLOT########################################


side='Front'
PlotTitle=side+' Wave RawData SUB first Correction JOB: '+list(JobNmeSORTED.values())[0];
fileName=side+' Wave RawData SUB first Correction JOB ' +list(JobNmeSORTED.values())[0]+'.html';
figWaveDataSubAveragePerPanet=PlotGraphPlotly(WaveRawData_sub_FirstCorrFRONT,PlotTitle,fileName,ColorList).PlotWaveDataVSEarliestWave();
#####Back
try:
    side='Back'
    PlotTitle=side+' Wave RawData SUB first Correction JOB: '+list(JobNmeSORTED.values())[0];
    fileName=side+' Wave RawData SUB first Correction JOB ' +list(JobNmeSORTED.values())[0]+'.html';
    figWaveDataSubAveragePerPanet=PlotGraphPlotly(WaveRawData_sub_FirstCorrBACK,PlotTitle,fileName,ColorList).PlotWaveDataVSEarliestWave();

except:
    1

side='Front'
PlotTitle=side+' C2C Diffrence per Wave Vs First Wave Calibration: '+list(JobNmeSORTED.values())[0];
fileName=side+' C2C Diffrence per Wave Vs First Wave Calibration ' +list(JobNmeSORTED.values())[0]+'.html';
figC2Cdiff=PlotGraphPlotly(C2C_FromWaveDifferanceFRONT,PlotTitle,fileName,ColorList).PlotC2CDiff();
#####Back
try:
    side='Back'
    PlotTitle=side+' C2C Diffrence per Wave Vs First Wave Calibration: '+list(JobNmeSORTED.values())[0];
    fileName=side+' C2C Diffrence per Wave Vs First Wave Calibration ' +list(JobNmeSORTED.values())[0]+'.html';
    figC2Cdiff=PlotGraphPlotly(C2C_FromWaveDifferanceBACK,PlotTitle,fileName,ColorList).PlotC2CDiff();

except:
    1











