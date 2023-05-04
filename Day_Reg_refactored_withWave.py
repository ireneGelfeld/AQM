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
global DistBetweenSets,GlobalScale,PanelLengthInMM,JobLengthתcolor_combinations,FullColorList,JobLengthWave;

# For setting the min job length for Change Wave plot- this parameter should be used for setting the allowble moving avarage- for example set JobLengthWave= 100, MoveAveWave=20;
JobLengthWave=100;
MoveAveWave=30;
#For 252
MarkSetVersion=252

if MarkSetVersion==252:

    
    GlobalScale = 0.9983 # Drop3 simplex = 0.9976, Duplex = 0.9984 ,,,, Drop5 Simplex = 0.9953, Duplex = 0.9945 
    DistBetweenSets =  125686/GlobalScale; 
    
else:
#For 201
    GlobalScale = 0.9983
    DistBetweenSets =  102693; 
    

PanelLengthInMM = 650;
JobLength = 0;


#### Plots
I2Splot=0 # Plot I2S 
C2Cplot=1 # Plot C2C
ScalePlot=0 # Plot Scale
WaveChangePlot=1 # Plot Wave Change
c2cChangePlot=1 # Plot c2c Change


color_combinations = [    ['Black', 'Yellow'],
    ['Black', 'Cyan'],
    ['Black', 'Blue'],
    ['Black', 'Green'],
    ['Black', 'Orange'],
    ['Black', 'Magenta'],
    ['Yellow', 'Cyan'],
    ['Yellow', 'Blue'],
    ['Yellow', 'Green'],
    ['Yellow', 'Orange'],
    ['Yellow', 'Magenta'],
    ['Cyan', 'Blue'],
    ['Cyan', 'Green'],
    ['Cyan', 'Orange'],
    ['Cyan', 'Magenta'],
    ['Blue', 'Green'],
    ['Blue', 'Orange'],
    ['Blue', 'Magenta'],
    ['Green', 'Orange'],
    ['Green', 'Magenta'],
    ['Orange', 'Magenta']
]


FullColorList=['Black','Blue','Cyan','Green','Magenta','Orange','Yellow']


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


class DispImagePlacment():
    def __init__(self, pthF,fldrs,side,pageSide,JobLength):
        ''' constructor ''' 
        self.pthF = pthF;
        self.fldrs = fldrs;
        self.side = side;
        self.pageSide=pageSide;
        self.JobLength = JobLength;
        
    def CheckIfFileValid(self,f):
        vlid=False;
        dbtmp=pd.DataFrame();
        # pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
        
        zip_file_path = self.pthF+f
        subdir_name_in_zip = self.side+'/'+'RawResults';
        file_name_in_zip = 'ImagePlacement_Left.csv';
        
        try:
            dbtmp=self.GetFileFromZip(zip_file_path,subdir_name_in_zip,file_name_in_zip);
            if len(dbtmp['Flat Id'])>self.JobLength:
                vlid= True;
        except:
            vlid=False;
                        
        return vlid;
    def GetFileFromZip(self,zip_file_path,subdir_name_in_zip,file_name_in_zip):
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_path_in_zip = subdir_name_in_zip + "/" + file_name_in_zip
            with zip_ref.open(file_path_in_zip) as file:
                # read the contents of the file into memory
                file_content = file.read()
                
                # convert the file content to a pandas dataframe
                df = pd.read_csv(BytesIO(file_content))
        return  df;          
        
    def ExtractBasline(self,f):
        # magePlacement_res=pd.read_csv(self.pthF+f+'/'+self.side+'/'+'AnalysisResults/ImagePlacementAnalysis_'+self.side+'.csv',usecols=[5])
        zip_file_path = self.pthF+f
        subdir_name_in_zip = self.side+'/'+'AnalysisResults';
        file_name_in_zip = 'ImagePlacementAnalysis_'+self.side+'.csv';
        
        magePlacement_res=self.GetFileFromZip(zip_file_path,subdir_name_in_zip,file_name_in_zip);

        BaseLine=magePlacement_res['Expected X'][0];
        return BaseLine;
    
    def ReadImagePlacmentData(self):
        os.chdir(self.pthF);
        dbtmp=pd.DataFrame()
        ImagePlacement_pp=pd.DataFrame()
        fname='ImagePlacement_'+self.pageSide+'.csv';
        for f in self.fldrs:
            # pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults/';
            # # print(pthFf)
            # try:
            #    os.chdir(pthFf);
            # except:
            #    continue;
            
            if self.CheckIfFileValid(f):
                try:
                    BaseLine=self.ExtractBasline(f);
                    
                    zip_file_path = self.pthF+f
                    subdir_name_in_zip = self.side+'/'+'RawResults';
                    file_name_in_zip = fname;
                    
                    dbtmp=self.GetFileFromZip(zip_file_path,subdir_name_in_zip,file_name_in_zip);
                    
                    ImagePlacement_pp = pd.concat((ImagePlacement_pp, (dbtmp['T1->X'].rename(f)-BaseLine)), axis=1)
                except:
                    continue;
            
        return  ImagePlacement_pp;    
                

                       

class CalcC2C_AvrgOfAll(DispImagePlacment):
    def  __init__(self, pthF,fldrs,side,JobLength,PanelLengthInMM,pageSide): 
        super().__init__(pthF,fldrs,side,pageSide,JobLength)
        self.PanelLengthInMM = PanelLengthInMM;
  
    def LoadRawDataOLD(self,fname,f):
        RawData=pd.read_csv(self.pthF+f+'/'+self.side+'/'+'RawResults'+'/'+fname);
        l1=RawData.iloc[:,1].unique().tolist()
        RawDataSuccess=RawData[RawData['Registration Status']=='Success'].reset_index(drop=True)
        flatNumberFailed=(RawData[RawData['Registration Status']!='Success'].iloc[:,1].unique().tolist());
        return  RawDataSuccess,flatNumberFailed,l1;

    def GetFileFromZip(self,zip_file_path,subdir_name_in_zip,file_name_in_zip):
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_path_in_zip = subdir_name_in_zip + "/" + file_name_in_zip
            with zip_ref.open(file_path_in_zip) as file:
                # read the contents of the file into memory
                file_content = file.read()
                
                # convert the file content to a pandas dataframe
                df = pd.read_csv(BytesIO(file_content))
        return  df;     
    
    def LoadRawData(self,fname,f):
        
        zip_file_path = self.pthF+f
        subdir_name_in_zip = self.side+'/'+'RawResults';
        file_name_in_zip = fname;
        
        RawData=self.GetFileFromZip(zip_file_path, subdir_name_in_zip, file_name_in_zip);
        l1=RawData.iloc[:,1].unique().tolist()
        RawDataSuccess=RawData[RawData['Registration Status']=='Success'].reset_index(drop=True)
        flatNumberFailed=(RawData[RawData['Registration Status']!='Success'].iloc[:,1].unique().tolist());
        return  RawDataSuccess,flatNumberFailed,l1; 
    
    def CalcMeanByColorForAllJobs(self,fname):
        
        DataAllMeanColorSET1ToT = pd.DataFrame();
        DataAllMeanColorSET2ToT = pd.DataFrame();
        DataAllMeanColorSET3ToT = pd.DataFrame();
        
        
        lnToT={}
        colorDicOrg={};
        
        for f in self.fldrs:
            try:
                RawDataSuccess,flatNumberFailed,l1 = self.LoadRawData(fname,f);
                
                DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3,ln, colorDic= self.CalcMeanByColor(RawDataSuccess);
                
                
                for clr in colorDic.values():
                    if clr in list(colorDicOrg.values()): 
                        DataAllMeanColorSET1ToT['Set #1 X'][clr] = DataAllMeanColorSET1ToT['Set #1 X'][clr]  + DataAllMeanColorSET1['Set #1 X'][clr]*ln; 
                        DataAllMeanColorSET2ToT['Set #2 X'][clr] = DataAllMeanColorSET2ToT['Set #2 X'][clr]  + DataAllMeanColorSET2['Set #2 X'][clr]*ln;
                        DataAllMeanColorSET3ToT['Set #3 X'][clr] = DataAllMeanColorSET3ToT['Set #3 X'][clr]  + DataAllMeanColorSET3['Set #3 X'][clr]*ln;
                        lnToT[clr]= lnToT[clr] + ln;
                    else:
                        row = pd.Series({'Set #1 X':DataAllMeanColorSET1['Set #1 X'][clr]*ln},name=clr);
                        DataAllMeanColorSET1ToT=DataAllMeanColorSET1ToT.append(row);
                        row = pd.Series({'Set #2 X':DataAllMeanColorSET2['Set #2 X'][clr]*ln},name=clr);
                        DataAllMeanColorSET2ToT=DataAllMeanColorSET2ToT.append(row);
                        row = pd.Series({'Set #3 X':DataAllMeanColorSET3['Set #3 X'][clr]*ln},name=clr);
                        DataAllMeanColorSET3ToT=DataAllMeanColorSET3ToT.append(row);
                        colorDicOrg[len(colorDicOrg)]= clr;
                        lnToT[clr]=  ln;
                    
            except:
                 continue;
        if not len(DataAllMeanColorSET1['Set #1 X']):
            for  clr in colorDicOrg.values():
                row = pd.Series({'Set #1 X':0,'Ink\Sets':clr},name=clr);
                DataAllMeanColorSET1=DataAllMeanColorSET1.append(row);
                row = pd.Series({'Set #2 X':0,'Ink\Sets':clr},name=clr);
                DataAllMeanColorSET2=DataAllMeanColorSET2.append(row);
                row = pd.Series({'Set #3 X':0,'Ink\Sets':clr},name=clr);
                DataAllMeanColorSET3=DataAllMeanColorSET3.append(row);         
            
        for  clr in colorDicOrg.values():
           
            DataAllMeanColorSET1['Set #1 X'][clr] =DataAllMeanColorSET1ToT['Set #1 X'][clr]/lnToT[clr];
            DataAllMeanColorSET2['Set #2 X'][clr] =DataAllMeanColorSET2ToT['Set #2 X'][clr]/lnToT[clr];
            DataAllMeanColorSET3['Set #3 X'][clr] =DataAllMeanColorSET3ToT['Set #3 X'][clr]/lnToT[clr];
            
        MeregedDataAllMeanColor=pd.DataFrame();    
        
        MeregedDataAllMeanColor['Set #1 X']=DataAllMeanColorSET1['Set #1 X'];
        MeregedDataAllMeanColor['Set #2 X']=DataAllMeanColorSET2['Set #2 X'];
        MeregedDataAllMeanColor['Set #3 X']=DataAllMeanColorSET3['Set #3 X'];
        
        #MeregedDataAllMeanColor=MeregedDataAllMeanColor.rename(index=colorDicOrg);
        MeregedDataAllMeanColor=MeregedDataAllMeanColor.reset_index();
        MeregedDataAllMeanColor=MeregedDataAllMeanColor.rename(columns={'index':'Ink\Sets'});        
        Lname=fname.split('.');
        LLname=Lname[0].split('_');
        
        MeregedDataAllMeanColor.to_csv(self.pthF+'/'+'MeregedDataAllMeanColor_'+self.side+'_'+LLname[1]+'.csv',index=False);
        
        
        return  DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3,colorDicOrg;   
            
             
            
            
            
            
    
    def CalcMeanByColor(self,RawDataSuccess):
        
        RawDataSuccess=self.ConvertRowsToInt(RawDataSuccess)
        DataAllMeanColorSET1 = RawDataSuccess.groupby(['Ink\Sets'])['Set #1 X'].mean().reset_index()
        DataAllMeanColorSET2 = RawDataSuccess.groupby(['Ink\Sets'])['Set #2 X'].mean().reset_index()
        DataAllMeanColorSET3 = RawDataSuccess.groupby(['Ink\Sets'])['Set #3 X'].mean().reset_index()       
        colorDic={}
        for i,cl in enumerate(DataAllMeanColorSET1['Ink\Sets']):
            colorDic[i]=cl
        DataAllMeanColorSET1=DataAllMeanColorSET1.rename(index=colorDic)
        DataAllMeanColorSET2=DataAllMeanColorSET2.rename(index=colorDic)
        DataAllMeanColorSET3=DataAllMeanColorSET3.rename(index=colorDic)
        
        ln = len(RawDataSuccess['Set #1 X']);
        
        return DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3,ln,colorDic;
    
    def ConvertRowsToInt(self,RawDataSuccess):       
        RawDataSuccess['Set #1 X']  = RawDataSuccess['Set #1 X'].astype('int64');
        RawDataSuccess['Set #2 X']  = RawDataSuccess['Set #2 X'].astype('int64');
        RawDataSuccess['Set #3 X']  = RawDataSuccess['Set #3 X'].astype('int64');
        return RawDataSuccess;
    
    def LoadMeanColorPos(self):
       
        # RecPath= pthComp[0]+'/'+pthComp[1]+'/'+pthComp[2]+'/'+pthComp[3]+'/'+pthComp[4]+'/'+pthComp[5]+'/'+pthComp[6]+'/'+pthComp[7]+'/'+pthComp[8];
        MeregedDataAllMeanColor = pd.read_csv(self.pthF +'/'+'MeregedDataAllMeanColor_'+self.side+'_'+self.pageSide+'.csv')
        
        return MeregedDataAllMeanColor
    
    
    def LoadMeanColorPos_PickSide(self,pageSide):
       
        # RecPath= pthComp[0]+'/'+pthComp[1]+'/'+pthComp[2]+'/'+pthComp[3]+'/'+pthComp[4]+'/'+pthComp[5]+'/'+pthComp[6]+'/'+pthComp[7]+'/'+pthComp[8];
        MeregedDataAllMeanColor = pd.read_csv(self.pthF +'/'+'MeregedDataAllMeanColor_'+self.side+'_'+pageSide+'.csv')
        
        return MeregedDataAllMeanColor
    
    def f(self,XYS,ink,value):
        # geometry = {'X': {'Black':0,'Blue':2,'Cyan':1,'Green':2,'Magenta':0,'Orange':3,'Yellow':1},
        #             'Y': {'Black':1,'Blue':0,'Cyan':1,'Green':1,'Magenta':0,'Orange':0,'Yellow':0}}
        # geometry = {'X': {'Black':0,'Blue':2,'Cyan':1,'Green':2,'Magenta':0,'Orange':3,'Yellow':1},
        #             'Y': {'Black':0,'Blue':1,'Cyan':0,'Green':0,'Magenta':1,'Orange':1,'Yellow':1}}
        
        geometry = {'X': {'Black':-1,'Blue':1,'Cyan':0,'Green':1,'Magenta':-1,'Orange':2,'Yellow':0},
                    'Y': {'Black':0,'Blue':1,'Cyan':0,'Green':0,'Magenta':1,'Orange':1,'Yellow':1}}
        
        # geometry = {'X': {'Black':-1,'Cyan':0,'Yellow':0,'Magenta':-1},
        #             'Y': {'Black':0,'Cyan':0,'Yellow':-1,'Magenta':-1}}
        target_distance = {'X':144.4,'Y':64} 
        return value + (geometry[XYS][ink]) * target_distance[XYS] * (21.16666666 / GlobalScale * 0.9976)
    
    def get_key(self,my_dict,val):
        for key, value in my_dict.items():
             if val == value:
                 return key
    
                
    def CalcScaleFromTarget(self):
         
         
         
         MeregedDataAllMeanColor= self.LoadMeanColorPos();
         
         colorDic={}
         for i,cl in enumerate(MeregedDataAllMeanColor['Ink\Sets']):
             colorDic[i]=cl
         
         valueSet1= MeregedDataAllMeanColor['Set #1 X'][self.get_key(colorDic,'Cyan')]

         valueSet2= valueSet1+(DistBetweenSets/GlobalScale);

         valueSet3= valueSet2+(DistBetweenSets/GlobalScale);
         
         for key, value in colorDic.items():
             MeregedDataAllMeanColor['Set #1 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet1)
             MeregedDataAllMeanColor['Set #2 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet2)
             MeregedDataAllMeanColor['Set #3 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet3)

         
         DataAllMeanColorSET1=MeregedDataAllMeanColor[['Set #1 X','Ink\Sets']].rename(index=colorDic);
         DataAllMeanColorSET2=MeregedDataAllMeanColor[['Set #2 X','Ink\Sets']].rename(index=colorDic);
         DataAllMeanColorSET3=MeregedDataAllMeanColor[['Set #3 X','Ink\Sets']].rename(index=colorDic);
         
         RefSETloc=DataAllMeanColorSET1;
         RefSETloc.insert(1,'Set #2 X' ,list(DataAllMeanColorSET2['Set #2 X']))
         RefSETloc.insert(2,'Set #3 X' ,list(DataAllMeanColorSET3['Set #3 X']))
         RefSETloc=RefSETloc.drop(columns=['Ink\Sets'])
         
        ####### Calc Scale 
         x=[0,1,2];

         slp=[]

            
         for inx in DataAllMeanColorSET1.index:
             y=list(RefSETloc.iloc[self.get_key(colorDic,inx),:])
             z = np.polyfit(x, y, 1)
             p = np.poly1d(z)
             slp.append(list(p)[0])
             
         RefSETloc.insert(3,'Slop' ,slp)
         
         
         return DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3,colorDic,RefSETloc
     
    def DataForCalcSCALE_FromData(self,DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3,colorDic,RefSETloc,fname,f):    
        
         
         RawDataSuccess,flatNumberFailed,l1= self.LoadRawData(fname,f);
         
         RawDataSuccess=self.ConvertRowsToInt(RawDataSuccess);
        
         indexNumberFailed=[]
         col=[];
         [col.append(str(j)) for j in range(len(l1))];
         St1dataAllColors=pd.DataFrame(columns=col,index=colorDic.values());
         St2dataAllColors=pd.DataFrame(columns=col,index=colorDic.values());
         St3dataAllColors=pd.DataFrame(columns=col,index=colorDic.values());
         
         
         for j,l in enumerate(l1):
             if not(l in flatNumberFailed):
                 FlatIDdata=RawDataSuccess[RawDataSuccess['Flat Id']==l].reset_index();
                 
                 for i,x in enumerate(FlatIDdata['Set #1 X']):
                     St1dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x);
                 for i,x in enumerate(FlatIDdata['Set #2 X']):
                     St2dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x);
                 for i,x in enumerate(FlatIDdata['Set #3 X']):
                     St3dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x);
                 
             else:
                 indexNumberFailed.append(j)
                 if j>0:
                     St1dataAllColors[str(j)]=St1dataAllColors[str(j-1)]
                     St2dataAllColors[str(j)]=St2dataAllColors[str(j-1)]
                     St3dataAllColors[str(j)]=St3dataAllColors[str(j-1)]
                 else:
                     St1dataAllColors[str(j)]=0
                     St2dataAllColors[str(j)]=0
                     St3dataAllColors[str(j)]=0
         
       
         ## 3 Point
         ## Create Panel Set per color - Actual color position
         PanelColorSet={};
         ListColorDict={};

         for c in St1dataAllColors.columns:
              for inx in St1dataAllColors.index:
                  ListColorDict[inx]=[St1dataAllColors[c][inx],St2dataAllColors[c][inx],St3dataAllColors[c][inx]]
              PanelColorSet[c]=ListColorDict
              ListColorDict={}
         
            
         
            
         
          ## Calc Scale per Color 
         
         x=[0,1,2];

 
         
         Scale=St1dataAllColors;


         for c in St1dataAllColors.columns:
              for inx in St1dataAllColors.index:
                  y= PanelColorSet[c][inx]
                  z = np.polyfit(x, y, 1)
                  p = np.poly1d(z)
                  try:
                      Scale[c][inx]=(RefSETloc['Slop'][inx]/list(p)[0]-1)*self.PanelLengthInMM*1000
                  except:
                      continue;
         
            
         
            
         ## Calc Scale Max - Min        
         ScaleMaxMin=[]
         for c in Scale.columns:
             ScaleMaxMin.append(np.max(Scale[c])-np.min(Scale[c]));
             
         
         
         
         
         return St1dataAllColors,St2dataAllColors,St3dataAllColors,RefSETloc,Scale,ScaleMaxMin,colorDic
        
        
    def DataForCalcSCALE(self,fname,f):
        
         RawDataSuccess,flatNumberFailed,l1= self.LoadRawData(fname,f);
         
         RawDataSuccess=self.ConvertRowsToInt(RawDataSuccess);
         
         
         MeregedDataAllMeanColor= self.LoadMeanColorPos();
         
         colorDic={}
         for i,cl in enumerate(MeregedDataAllMeanColor['Ink\Sets']):
             colorDic[i]=cl
         
         valueSet1= MeregedDataAllMeanColor['Set #1 X'][self.get_key(colorDic,'Cyan')]

         valueSet2= valueSet1+(DistBetweenSets/GlobalScale);

         valueSet3= valueSet2+(DistBetweenSets/GlobalScale);
         
         for key, value in colorDic.items():
             MeregedDataAllMeanColor['Set #1 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet1)
             MeregedDataAllMeanColor['Set #2 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet2)
             MeregedDataAllMeanColor['Set #3 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet3)

         
         DataAllMeanColorSET1=MeregedDataAllMeanColor[['Set #1 X','Ink\Sets']].rename(index=colorDic);
         DataAllMeanColorSET2=MeregedDataAllMeanColor[['Set #2 X','Ink\Sets']].rename(index=colorDic);
         DataAllMeanColorSET3=MeregedDataAllMeanColor[['Set #3 X','Ink\Sets']].rename(index=colorDic);
         
         indexNumberFailed=[]
         col=[];
         [col.append(str(j)) for j in range(len(l1))];
         St1dataAllColors=pd.DataFrame(columns=col,index=colorDic.values());
         St2dataAllColors=pd.DataFrame(columns=col,index=colorDic.values());
         St3dataAllColors=pd.DataFrame(columns=col,index=colorDic.values());
         
         
         for j,l in enumerate(l1):
             if not(l in flatNumberFailed):
                 FlatIDdata=RawDataSuccess[RawDataSuccess['Flat Id']==l].reset_index();
                 
                 for i,x in enumerate(FlatIDdata['Set #1 X']):
                     St1dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x);
                 for i,x in enumerate(FlatIDdata['Set #2 X']):
                     St2dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x);
                 for i,x in enumerate(FlatIDdata['Set #3 X']):
                     St3dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x);
                 
             else:
                 indexNumberFailed.append(j)
                 if j>0:
                     St1dataAllColors[str(j)]=St1dataAllColors[str(j-1)]
                     St2dataAllColors[str(j)]=St2dataAllColors[str(j-1)]
                     St3dataAllColors[str(j)]=St3dataAllColors[str(j-1)]
                 else:
                     St1dataAllColors[str(j)]=0
                     St2dataAllColors[str(j)]=0
                     St3dataAllColors[str(j)]=0
         
         ## Unite All Sets  - Expected color position          
         RefSETloc=DataAllMeanColorSET1;
         RefSETloc.insert(1,'Set #2 X' ,list(DataAllMeanColorSET2['Set #2 X']))
         RefSETloc.insert(2,'Set #3 X' ,list(DataAllMeanColorSET3['Set #3 X']))
         RefSETloc=RefSETloc.drop(columns=['Ink\Sets'])
       
         ## Create Panel Set per color - Actual color position
         PanelColorSet={};
         ListColorDict={};

         for c in St1dataAllColors.columns:
             for inx in St1dataAllColors.index:
                 ListColorDict[inx]=[St1dataAllColors[c][inx],St2dataAllColors[c][inx],St3dataAllColors[c][inx]]
             PanelColorSet[c]=ListColorDict
             ListColorDict={}
         
         ## Calc Scale per Color 
         

             
         
         Scale=St1dataAllColors.copy();


         for c in St1dataAllColors.columns:
             try:
                 for inx in St1dataAllColors.index:
                     y= PanelColorSet[c][inx]
                     z = np.polyfit(x, y, 1)
                     p = np.poly1d(z)
                     Scale[c][inx]=(RefSETloc['Slop'][inx]/list(p)[0]-1)*self.PanelLengthInMM*1000
             except:
                 continue;
         
         ## Calc Scale Max - Min        
         ScaleMaxMin=[]
         for c in Scale.columns:
             ScaleMaxMin.append(np.max(Scale[c])-np.min(Scale[c]));
             
         
         
         
         
         return St1dataAllColors,St2dataAllColors,St3dataAllColors,RefSETloc,Scale,ScaleMaxMin,colorDic
     
    def CalcScaleForAllJOBS(self):
        
        ScaleMaxMinDF=pd.DataFrame();
        
        
        fname= 'Registration_'+self.pageSide+'.csv';
        
        DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3,colorDic,RefSETloc = self.CalcScaleFromTarget();

        for f in self.fldrs:
           stP=pd.DataFrame();
           try:
               St1dataAllColors,St2dataAllColors,St3dataAllColors,RefSETloc,Scale,ScaleMaxMin,colorDic=self.DataForCalcSCALE_FromData(DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3,colorDic,RefSETloc,fname,f);
             
               stP[f]=ScaleMaxMin;
               ScaleMaxMinDF=pd.concat([ScaleMaxMinDF, stP[f]],axis=1);
           except:
               continue;
        return ScaleMaxMinDF
                
        
    def CalcC2CSingleSide(self,fname,f,DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3):
        RawDataSuccess,flatNumberFailed,l1= self.LoadRawData(fname,f);
        
        RawDataSuccess=self.ConvertRowsToInt(RawDataSuccess);
        
        #DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3=self.CalcMeanByColor(RawDataSuccess);
        
        indexNumberFailed=[]
        C2Creg=[]
        
        for j,l in enumerate(l1):
            if not(l in flatNumberFailed):
                FlatIDdata=RawDataSuccess[RawDataSuccess['Flat Id']==l].reset_index();
                St1data=[];
                [St1data.append(x-DataAllMeanColorSET1['Set #1 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #1 X'])];
                St2data=[];
                [St2data.append(x-DataAllMeanColorSET2['Set #2 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #2 X'])];
                St3data=[];
                [St3data.append(x-DataAllMeanColorSET3['Set #3 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #3 X'])];
                tmp=[(np.max(St1data)-np.min(St1data)),(np.max(St2data)-np.min(St2data)),(np.max(St3data)-np.min(St3data))];
                C2Creg.append(np.max(tmp));
                # C2Creg.append(tmp[np.argmax([abs((np.max(St1data)-np.min(St1data))),abs((np.max(St2data)-np.min(St2data))),abs((np.max(St2data)-np.min(St2data)))])]);

            else:
                indexNumberFailed.append(j)
                if j>0:
                    C2Creg.append(C2Creg[j-1])
                else:
                    C2Creg.append(0.0)
    
        return C2Creg,indexNumberFailed;
    
    def merge_lists_without_duplicates(self,list1, list2):
        merged_list = list(set(list1 + list2))
        return merged_list
    
    def CreatStNdata(self,FlatIDdata,SetSTR,DataAllMeanColorSET):
        Stdata={};
        for i,x in enumerate(FlatIDdata[SetSTR]):
            Stdata[FlatIDdata['Ink\Sets'][i]]=x-DataAllMeanColorSET[SetSTR][FlatIDdata['Ink\Sets'][i]]
            
        return Stdata
    
    def CalcC2CSingleSideColorPair(self,fnameLeft,fnameRight,f,DataAllMeanColorSET1Left,DataAllMeanColorSET2Left,DataAllMeanColorSET3Left,DataAllMeanColorSET1Right,DataAllMeanColorSET2Right,DataAllMeanColorSET3Right):
        
        RawDataSuccessLeft,flatNumberFailedRight,l1Left= self.LoadRawData(fnameLeft,f);
        RawDataSuccessRight,flatNumberFailedLeft,l1Right= self.LoadRawData(fnameRight,f);

        RawDataSuccessLeft=self.ConvertRowsToInt(RawDataSuccessLeft);
        RawDataSuccessRight=self.ConvertRowsToInt(RawDataSuccessRight);

        
        flatNumberFailed =self.merge_lists_without_duplicates(flatNumberFailedRight,flatNumberFailedLeft)
        l1=self.merge_lists_without_duplicates(l1Left,l1Right)
        
        #DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3=self.CalcMeanByColor(RawDataSuccess);
        MaxLeftRightDiff={}
        for clr in FullColorList:
            MaxLeftRightDiff[clr]=0
            
        indexNumberFailed=[]
        C2Creg=[]
        for j,l in enumerate(l1):
            if not(l in flatNumberFailed):
                FlatIDdataLeft=RawDataSuccessLeft[RawDataSuccessLeft['Flat Id']==l].reset_index();
                FlatIDdataRight=RawDataSuccessRight[RawDataSuccessRight['Flat Id']==l].reset_index();

                St1dataLeft= self.CreatStNdata(FlatIDdataLeft,'Set #1 X',DataAllMeanColorSET1Left);
                St2dataLeft= self.CreatStNdata(FlatIDdataLeft,'Set #2 X',DataAllMeanColorSET2Left);
                St3dataLeft= self.CreatStNdata(FlatIDdataLeft,'Set #3 X',DataAllMeanColorSET3Left);
                
                St1dataRight= self.CreatStNdata(FlatIDdataRight,'Set #1 X',DataAllMeanColorSET1Right);
                St2dataRight= self.CreatStNdata(FlatIDdataRight,'Set #2 X',DataAllMeanColorSET2Right);
                St3dataRight= self.CreatStNdata(FlatIDdataRight,'Set #3 X',DataAllMeanColorSET3Right);
                
                MaxLeftRightDiff={}
                for clr in St1dataLeft.keys():
                    # tmp=[St1dataLeft[clr]-St1dataRight[clr],St2dataLeft[clr]-St2dataRight[clr],St3dataLeft[clr]-St3dataRight[clr]]
                    tmp=[St1dataLeft[clr]-St1dataRight[clr]]

                    MaxLeftRightDiff[clr]=np.max(tmp)
                C2Creg.append(MaxLeftRightDiff);
                # C2Creg.append(tmp[np.argmax([abs((np.max(St1data)-np.min(St1data))),abs((np.max(St2data)-np.min(St2data))),abs((np.max(St2data)-np.min(St2data)))])]);

            else:
                indexNumberFailed.append(j)
                if j>0:
                    C2Creg.append(C2Creg[j-1])
                else:
                    C2Creg.append(MaxLeftRightDiff)
    
        return C2Creg,indexNumberFailed;
    
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
    
    
    def CreateWaveChangeDataOld(self,JobLengthWave):
        
        WaveChangeList=[];
        indexJobNameDic={}
        
        # DataAllMeanColorSET1Left,DataAllMeanColorSET2Left,DataAllMeanColorSET3Left,colorDic = self.CalcMeanByColorForAllJobs('Registration_Left.csv')
        # DataAllMeanColorSET1Right,DataAllMeanColorSET2Right,DataAllMeanColorSET3Right,colorDic = self.CalcMeanByColorForAllJobs('Registration_Right.csv')




        MeregedDataAllMeanColorLeft= self.LoadMeanColorPos_PickSide('Left');
        MeregedDataAllMeanColorRight= self.LoadMeanColorPos_PickSide('Right');
        
        
        colorDic={}
        
        for i in MeregedDataAllMeanColorLeft.index:
           colorDic[i]= MeregedDataAllMeanColorLeft['Ink\Sets'][i]
        
        DataAllMeanColorSET1Left=MeregedDataAllMeanColorLeft[['Ink\Sets','Set #1 X']].rename(index=colorDic)
        DataAllMeanColorSET2Left=MeregedDataAllMeanColorLeft[['Ink\Sets','Set #2 X']].rename(index=colorDic)
        DataAllMeanColorSET3Left=MeregedDataAllMeanColorLeft[['Ink\Sets','Set #3 X']].rename(index=colorDic)


        DataAllMeanColorSET1Right=MeregedDataAllMeanColorRight[['Ink\Sets','Set #1 X']].rename(index=colorDic)
        DataAllMeanColorSET2Right=MeregedDataAllMeanColorRight[['Ink\Sets','Set #2 X']].rename(index=colorDic)
        DataAllMeanColorSET3Right=MeregedDataAllMeanColorRight[['Ink\Sets','Set #3 X']].rename(index=colorDic)

        # MeregedDataAllMeanColor= self.LoadMeanColorPos();
        
        
        
        JobNmeSORTED= list(self.SortJobsByTime(self.fldrs).values())
   
        for f in JobNmeSORTED:
            try:
                vlid,lngth=self.CheckIfFileValid_forWave(f,JobLengthWave)
                if vlid:
                    C2Creg,indexNumberFailed = self.CalcC2CSingleSideColorPair('Registration_Left.csv','Registration_Right.csv',f,DataAllMeanColorSET1Left,DataAllMeanColorSET2Left,DataAllMeanColorSET3Left,DataAllMeanColorSET1Right,DataAllMeanColorSET2Right,DataAllMeanColorSET3Right)
                    WaveChangeList=WaveChangeList+C2Creg

                    indexJobNameDic[len(WaveChangeList)-1]=[f,str(lngth)]

            except:
                    continue;
            
        return WaveChangeList,indexJobNameDic;
    
    def CreateWaveChangeData(self,JobLengthWave):
        
        WaveChangeList=[];
        indexJobNameDic={}
        
        # DataAllMeanColorSET1Left,DataAllMeanColorSET2Left,DataAllMeanColorSET3Left,colorDic = self.CalcMeanByColorForAllJobs('Registration_Left.csv')
        # DataAllMeanColorSET1Right,DataAllMeanColorSET2Right,DataAllMeanColorSET3Right,colorDic = self.CalcMeanByColorForAllJobs('Registration_Right.csv')




        MeregedDataAllMeanColorLeft= self.LoadMeanColorPos_PickSide('Left');
        MeregedDataAllMeanColorRight= self.LoadMeanColorPos_PickSide('Right');
        
        
        colorDic={}
        
        for i in MeregedDataAllMeanColorLeft.index:
           colorDic[i]= MeregedDataAllMeanColorLeft['Ink\Sets'][i]
        
        DataAllMeanColorSET1Left=MeregedDataAllMeanColorLeft[['Ink\Sets','Set #1 X']].rename(index=colorDic)
        DataAllMeanColorSET2Left=MeregedDataAllMeanColorLeft[['Ink\Sets','Set #2 X']].rename(index=colorDic)
        DataAllMeanColorSET3Left=MeregedDataAllMeanColorLeft[['Ink\Sets','Set #3 X']].rename(index=colorDic)


        DataAllMeanColorSET1Right=MeregedDataAllMeanColorRight[['Ink\Sets','Set #1 X']].rename(index=colorDic)
        DataAllMeanColorSET2Right=MeregedDataAllMeanColorRight[['Ink\Sets','Set #2 X']].rename(index=colorDic)
        DataAllMeanColorSET3Right=MeregedDataAllMeanColorRight[['Ink\Sets','Set #3 X']].rename(index=colorDic)

        # MeregedDataAllMeanColor= self.LoadMeanColorPos();
        
        
        JobNmeSORTED= list(self.SortJobsByTime(self.fldrs).values())

        ValidSortedJobListWithWave=[]

        for f in JobNmeSORTED:
            vlid,lngth=self.CheckIfFileValid_forWave(f,JobLengthWave)
            if vlid or ('WaveCalibration' in f):
                    ValidSortedJobListWithWave.append(f)
                    
                    
        WaveFilesInx=self.find_indexes_with_substring(ValidSortedJobListWithWave, 'WaveCalibration')
        WaveJobPrintedDic={}


        k=0
        for i,f in enumerate(ValidSortedJobListWithWave):
            try:
            
                C2Creg,indexNumberFailed = self.CalcC2CSingleSideColorPair('Registration_Left.csv','Registration_Right.csv',f,DataAllMeanColorSET1Left,DataAllMeanColorSET2Left,DataAllMeanColorSET3Left,DataAllMeanColorSET1Right,DataAllMeanColorSET2Right,DataAllMeanColorSET3Right)
                WaveChangeList=WaveChangeList+C2Creg
                indexJobNameDic[len(WaveChangeList)-1]=[f,lngth]

                if len(WaveFilesInx)>0:
                    if i>WaveFilesInx[k] or WaveFilesInx[k] == 0:
                        inxForW=list(indexJobNameDic.keys())[len(list(indexJobNameDic.keys()))-2]
                        WaveJobPrintedDic[inxForW]=[ValidSortedJobListWithWave[WaveFilesInx[k]],i]
                        k=k+1;
                

            except:
                    continue;
            
            
        return WaveChangeList,indexJobNameDic,WaveJobPrintedDic;
    
    def CreateC2CchangeData(self,c2c,JobLengthWave):
        
        c2cChangeList=[];
        indexJobNameDic={}
    
        ValidSortedJobListWithWave=[]
        
        for f in c2c.columns:
             vlid,lngth=self.CheckIfFileValid_forWave(f,JobLengthWave)
             if vlid or ('WaveCalibration' in f):
                     ValidSortedJobListWithWave.append(f)
             
              
                    
        WaveFilesInx=self.find_indexes_with_substring(ValidSortedJobListWithWave, 'WaveCalibration')
        WaveJobPrintedDic={}
        
        k=0
        for i,f in enumerate(ValidSortedJobListWithWave):
            try:
            
                c2cChangeList=c2cChangeList+list(c2c[f].dropna())
                indexJobNameDic[len(c2cChangeList)-1]=[f,lngth]

                if len(WaveFilesInx)>0:
                    if i>WaveFilesInx[k] or WaveFilesInx[k] == 0:
                        inxForW=list(indexJobNameDic.keys())[len(list(indexJobNameDic.keys()))-2]
                        WaveJobPrintedDic[inxForW]=[ValidSortedJobListWithWave[WaveFilesInx[k]],i]
                        k=k+1;
            except:
                      continue;
              
              
        return c2cChangeList,indexJobNameDic,WaveJobPrintedDic;
        
    
    def find_indexes_with_substring(self,lst, substring):
        indexes = []
        for i, string in enumerate(lst):
            if substring in string:
                indexes.append(i)
        return indexes
    
    def CheckIfFileValid_forWave(self,f,JobLengthWave):
        vlid=False;
        lngth=0;
        dbtmp=pd.DataFrame();
        # pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
        
        zip_file_path = self.pthF+f
        subdir_name_in_zip = self.side+'/'+'RawResults';
        file_name_in_zip='ImagePlacement_Left.csv';
        
        try:
            dbtmp=self.GetFileFromZip(zip_file_path,subdir_name_in_zip,file_name_in_zip);
            if len(dbtmp['Flat Id'])>JobLengthWave:
                vlid= True;
            lngth=len(dbtmp['Flat Id'])
        except:
            vlid=False;
            
        return vlid,lngth;

    def CalcC2CregForLeftRight(self):
        ImagePlacement_pp=pd.DataFrame()
        flatNumberFailed_pp=pd.DataFrame();
        
        DataAllMeanColorSET1left,DataAllMeanColorSET2left,DataAllMeanColorSET3left,colorDic = self.CalcMeanByColorForAllJobs('Registration_Left.csv')
        DataAllMeanColorSET1right,DataAllMeanColorSET2right,DataAllMeanColorSET3right,colorDic = self.CalcMeanByColorForAllJobs('Registration_Right.csv')
        
        for f in self.fldrs:
            stP=pd.DataFrame();
            flatNumberFailed=pd.DataFrame();
            try:
                if self.CheckIfFileValid(f):
                    C2CregLeft,indexNumberFailedLeft=self.CalcC2CSingleSide('Registration_Left.csv',f,DataAllMeanColorSET1left,DataAllMeanColorSET2left,DataAllMeanColorSET3left);
                    C2CregRight,indexNumberFailedRight=self.CalcC2CSingleSide('Registration_Right.csv',f,DataAllMeanColorSET1right,DataAllMeanColorSET2right,DataAllMeanColorSET3right);
                    C2CMaxLeftRight=[];
                    for i in range(len(C2CregRight)):
                            tmp=[C2CregLeft[i],C2CregRight[i]];
                            C2CMaxLeftRight.append(np.max(tmp));
                            # C2CMaxLeftRight.append(tmp[np.argmax([abs(C2CregLeft[i]),abs(C2CregRight[i])])]);
                     
                    stP[f]=C2CMaxLeftRight;
                    flatNumberFailed[f]=list(OrderedDict.fromkeys(indexNumberFailedLeft+indexNumberFailedRight));
                    flatNumberFailed_pp=pd.concat([flatNumberFailed_pp, flatNumberFailed[f]],axis=1);
                    ImagePlacement_pp=pd.concat([ImagePlacement_pp, stP[f]],axis=1);
            except:
                continue;
        
        return ImagePlacement_pp,flatNumberFailed_pp                         

def SortJobsByTime(ColmnList):
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




import threading


class myThread (threading.Thread):
   def __init__(self, threadID, name , folder):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.folder = folder

   def run(self):
      
      global DataPivotFront,flatNumberFailedFront;
      global DataPivotBack,flatNumberFailedBack;
      global ScaleMaxMinDF_FRONTFLeft,ScaleMaxMinDF_FRONTRight;
      global ScaleMaxMinDF_BACKFLeft,ScaleMaxMinDF_BACKRight;
      global ImagePlacement_Leftpp,ImagePlacement_Rightpp;
      global ImagePlacement_Leftpp_BACK,ImagePlacement_Rightpp_BACK;


    
          
      if self.name == "Thread-C2C":
          print ("Starting " + self.name)
          DataPivotFront,flatNumberFailedFront=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CalcC2CregForLeftRight();

        
          try:
            DataPivotBack,flatNumberFailedBack=CalcC2C_AvrgOfAll(pthF,folder,'Back',JobLength,PanelLengthInMM,'Left').CalcC2CregForLeftRight();
            
        
          except:
            1;
          print ("Exiting " + self.name)     
          
      if self.name == "Thread-Scale":   
          print ("Starting " + self.name)
      
          ScaleMaxMinDF_FRONTFLeft=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CalcScaleForAllJOBS();
          ScaleMaxMinDF_FRONTRight=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Right').CalcScaleForAllJOBS();
        
        
          try:
            ScaleMaxMinDF_BACKFLeft=CalcC2C_AvrgOfAll(pthF,folder,'Back',JobLength,PanelLengthInMM,'Left').CalcScaleForAllJOBS();
            ScaleMaxMinDF_BACKRight=CalcC2C_AvrgOfAll(pthF,folder,'Back',JobLength,PanelLengthInMM,'Right').CalcScaleForAllJOBS();
          except:
            1;
          
          print ("Exiting " + self.name)
       
      if  self.name == "Thread-I2S": 
          print ("Starting " + self.name)
          
          ImagePlacement_Leftpp=DispImagePlacment(pthF,folder,'Front','Left',JobLength).ReadImagePlacmentData()
          ImagePlacement_Rightpp=DispImagePlacment(pthF,folder,'Front','Right',JobLength).ReadImagePlacmentData()
        
          try:
            ImagePlacement_Leftpp_BACK=DispImagePlacment(pthF,folder,'Back','Left',JobLength).ReadImagePlacmentData()
            ImagePlacement_Rightpp_BACK=DispImagePlacment(pthF,folder,'Back','Right',JobLength).ReadImagePlacmentData()
          except:
            1
    
          print ("Exiting " + self.name)
      

class PlotPlotly():
    def __init__(self, pthF, side):
        self.pthF = pthF;
        self.side = side;
        
    
    def Plot2subPlots(self,subplot_titles1,subplot_titles2,PlotTitle,db1,db2,dbName1,dbName2,fileName):
        
        fig = go.Figure()
        #fig_back = go.Figure()
        fig = make_subplots(rows=2, cols=1,subplot_titles=(subplot_titles1, subplot_titles2), vertical_spacing=0.1, shared_xaxes=True)
       
        
        col=list(SortJobsByTime(list(db1.columns)).values())
        rnge=range(len(col))
        
        for i in rnge:
        # for i in rnge:
            fig.add_trace(go.Scatter(y=list(db1[col[i]]),
                        name=col[i]+' '+dbName1),row=1, col=1)
            try:
                fig.add_trace(go.Scatter(y=list(db2[col[i]]),
                            name=col[i]+' '+dbName2), row=2, col=1)
            except:
                continue;
        
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
        #plot(fig_back,filename="AQM-Back.html")  
        fig.show()
        
        return fig
    
    def PlotWaveChange(self,WaveChangeDF,indexJobNameDic, PlotTitle,fileName):
        
       fig = go.Figure()
       
       WaveChangeDF = WaveChangeDF.dropna(axis=1)

       ColorList= list(WaveChangeDF.columns)
       
       for clr in ColorList:     
           lineColor=clr;
         
           
           if lineColor=='Yellow':
               lineColor='gold';
           
           fig.add_trace(
           go.Scatter(y=WaveChangeDF[clr],line_color= lineColor,
                       name='Wave Differance Left-Right '+' color '+clr))
           
           
           # ymax=max(WaveRawDataDic[ColorList[0]]-WaveDataWithMaxFilterDic[self.ColorList[0]])
       ymax=np.max(WaveChangeDF[clr])+100
        
       for key, value in indexJobNameDic.items():
            fig.add_trace(go.Scatter(x=[key], y=[ymax],
                                    marker=dict(color="green", size=6),
                                    mode="markers",
                                    text=value[0]+','+value[1],
                                    # font_size=18,
                                    hoverinfo='text'))
            
            fig.data[len(fig.data)-1].showlegend = False
            fig.add_vline(x=key, line_width=2, line_dash="dash", line_color="green")
           
        
       fig.update_layout(
                hoverlabel=dict(
                    namelength=-1
                )
            )
       fig.update_layout(title=self.side+' '+PlotTitle)
           
    
       plot(fig,filename=self.side+' '+fileName+".html") 
       
       return fig
    
    def PlotWaveChange_WithMovingAVRG(self,WaveChangeDF,indexJobNameDic,WaveJobPrintedDic,MoveAveWave, PlotTitle,fileName):
        
       fig = go.Figure()
       
       WaveChangeDF = WaveChangeDF.dropna(axis=1)

       
       ColorList= list(WaveChangeDF.columns)
       
       for clr in ColorList:     
           lineColor=clr;
         
           
           if lineColor=='Yellow':
               lineColor='gold';
           
           fig.add_trace(
           go.Scatter(y=WaveChangeDF[clr],line_color= lineColor,
                       name='Wave Differance Left-Right '+' color '+clr))
           fig.data[len(fig.data)-1].visible = 'legendonly';

           
           fig.add_trace(
           go.Scatter(y=WaveChangeDF[clr].rolling(MoveAveWave).mean(),line_color= lineColor,
                       name='Wave Differance Left-Right- moving average of '+str(MoveAveWave)+' color '+clr))
           
           
           # ymax=max(WaveRawDataDic[ColorList[0]]-WaveDataWithMaxFilterDic[self.ColorList[0]])
       ymax=np.max(WaveChangeDF[clr].rolling(MoveAveWave).mean())+20
       ymaxWaveJob=np.max(WaveChangeDF[clr].rolling(MoveAveWave).mean())
        
       for key, value in indexJobNameDic.items():
            fig.add_trace(go.Scatter(x=[key], y=[ymax],
                                    marker=dict(color="green", size=10),
                                    mode="markers",
                                    text=value[0],
                                    # font_size=18,
                                    hoverinfo='text'))
            
            fig.data[len(fig.data)-1].showlegend = False
            fig.add_vline(x=key, line_width=2, line_dash="dash", line_color="green")
       pxWave=0     
       for i ,(key, value) in enumerate(WaveJobPrintedDic.items()):
            xWave=key+(1+int(JobLengthWave/10))
            if i>0:
               if abs(list(WaveJobPrintedDic.values())[i-1][1]- list(WaveJobPrintedDic.values())[i][1])<2:
                   xWave=pxWave + 1
            fig.add_trace(go.Scatter(x=[xWave], y=[ymaxWaveJob],
                                    marker=dict(color="red", size=10),
                                    mode="markers",
                                    text=value,
                                    # font_size=18,
                                    hoverinfo='text'))
            
            fig.data[len(fig.data)-1].showlegend = False
            fig.add_vline(x=xWave, line_width=2,  line_color="red")
            pxWave=xWave
           
        
       fig.update_layout(
                hoverlabel=dict(
                    namelength=-1
                )
            )
       fig.update_layout(title=self.side+' '+PlotTitle)
           
    
       plot(fig,filename=self.side+' '+fileName+".html") 
       
       return fig


    def Plotc2cChange_WithMovingAVRG(self,c2cChangeDF,indexJobNameDic,WaveJobPrintedDic,MoveAveWave, PlotTitle,fileName):
        
        fig = go.Figure()
       
        c2cChangeDF = c2cChangeDF.dropna(axis=1)

       
       
     
           
        fig.add_trace(
        go.Scatter(y=list(c2cChangeDF[0]),
                    name='C2C '))
        fig.data[len(fig.data)-1].visible = 'legendonly';

        
        fig.add_trace(
        go.Scatter(y=list(c2cChangeDF[0].rolling(MoveAveWave).mean()),
                    name='C2C moving average '))
           
           
           # ymax=max(WaveRawDataDic[ColorList[0]]-WaveDataWithMaxFilterDic[self.ColorList[0]])
        ymax=np.max(list(c2cChangeDF[0].rolling(MoveAveWave).mean())[MoveAveWave+10:])+20
        ymaxWaveJob=np.max(list(c2cChangeDF[0].rolling(MoveAveWave).mean())[MoveAveWave+10:])
        
        for key, value in indexJobNameDic.items():
            fig.add_trace(go.Scatter(x=[key], y=[ymax],
                                    marker=dict(color="green", size=10),
                                    mode="markers",
                                    text=value[0],
                                    # font_size=18,
                                    hoverinfo='text'))
            
            fig.data[len(fig.data)-1].showlegend = False
            fig.add_vline(x=key, line_width=2, line_dash="dash", line_color="green")
        pxWave=0     
        for i ,(key, value) in enumerate(WaveJobPrintedDic.items()):
            xWave=key+(1+int(JobLengthWave/10))
            if i>0:
               if abs(list(WaveJobPrintedDic.values())[i-1][1]- list(WaveJobPrintedDic.values())[i][1])<2:
                   xWave=pxWave + 1
            fig.add_trace(go.Scatter(x=[xWave], y=[ymaxWaveJob],
                                    marker=dict(color="red", size=10),
                                    mode="markers",
                                    text=value,
                                    # font_size=18,
                                    hoverinfo='text'))
            
            fig.data[len(fig.data)-1].showlegend = False
            fig.add_vline(x=xWave, line_width=2,  line_color="red")
            pxWave=xWave
           
        
        fig.update_layout(
                hoverlabel=dict(
                    namelength=-1
                )
            )
        fig.update_layout(title=self.side+' '+PlotTitle)
           
    
        plot(fig,filename=self.side+' '+fileName+".html") 
       
        return fig




# ################################### 
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthF = filedialog.askdirectory()


pthF=pthF+'/';



folder=PreapareData(pthF).ExtractFilesFromZip();




###################################################################################################################
import time

startCalc = time.time()


    

DataPivotFront=pd.DataFrame();
flatNumberFailedFront=pd.DataFrame();
DataPivotBack=pd.DataFrame();
flatNumberFailedBack=pd.DataFrame();
ScaleMaxMinDF_FRONTFLeft=pd.DataFrame();
ScaleMaxMinDF_FRONTRight=pd.DataFrame();
ScaleMaxMinDF_BACKFLeft=pd.DataFrame();
ScaleMaxMinDF_BACKRight=pd.DataFrame();
ImagePlacement_Leftpp=pd.DataFrame();
ImagePlacement_Rightpp=pd.DataFrame();
ImagePlacement_Leftpp_BACK=pd.DataFrame();
ImagePlacement_Rightpp_BACK=pd.DataFrame();



threadList=[]

if C2Cplot:
    threadList.append(myThread(1,"Thread-C2C",folder))


# threadList.append(myThread(2, "Thread-Scale",folder))

    
# for f in folder:
#     threadList.append(myThread(3, "Thread-I2S",f))    
        
if I2Splot:
    threadList.append(myThread(3, "Thread-I2S",folder))    


for itm in threadList:
    itm.start()

for itm in threadList:
    itm.join()





endCalc = time.time()
print(endCalc - startCalc)






####################Thread###################################
if ScalePlot:
    ScaleMaxMinDF_FRONTFLeft=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CalcScaleForAllJOBS();
    ScaleMaxMinDF_FRONTRight=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Right').CalcScaleForAllJOBS();
      
      
    try:
      ScaleMaxMinDF_BACKFLeft=CalcC2C_AvrgOfAll(pthF,folder,'Back',JobLength,PanelLengthInMM,'Left').CalcScaleForAllJOBS();
      ScaleMaxMinDF_BACKRight=CalcC2C_AvrgOfAll(pthF,folder,'Back',JobLength,PanelLengthInMM,'Right').CalcScaleForAllJOBS();
    except:
      1;

#################################Wave Prograss Over Time ######################
if WaveChangePlot:
    WaveChangeListFRONT,indexJobNameDicFRONT,WaveJobPrintedDicFRONT=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CreateWaveChangeData(JobLengthWave);
    WaveChangeDF_FRONT=pd.DataFrame(WaveChangeListFRONT)
    
    try:
       WaveChangeListBACK,indexJobNameDicBACK,WaveJobPrintedDicBACK=CalcC2C_AvrgOfAll(pthF,folder,'Back',JobLength,PanelLengthInMM,'Left').CreateWaveChangeData(JobLengthWave);
       WaveChangeDF_BACK=pd.DataFrame(WaveChangeListBACK)
    
    except:
      1; 

######################C2C Prograss Over Time ######################

if c2cChangePlot:
    c2cChangeListFRONT,indexJobNameDicFRONT,WaveJobPrintedDicFRONT=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CreateC2CchangeData(DataPivotFront, JobLengthWave);
    c2cChangeDF_FRONT=pd.DataFrame(c2cChangeListFRONT)
    
    try:
        c2cChangeListBACK,indexJobNameDicBACK,WaveJobPrintedDicBACK=CalcC2C_AvrgOfAll(pthF,folder,'Back',JobLength,PanelLengthInMM,'Left').CreateC2CchangeData(DataPivotBack, JobLengthWave);
        c2cChangeDF_BACK=pd.DataFrame(c2cChangeListBACK)
        
    except:
      1; 




os.chdir(pthF)
# #########################################################################################################################

###FOR DEBUG

################### PLOT SIGNALS ############################################
startFigure = time.time()

############################Plot I2S #################################
###########Front
if I2Splot:
    subplot_titles1="LEFT"
    subplot_titles2="RIGHT"
    PlotTitle= 'I2S- FRONT'
    db1= ImagePlacement_Leftpp
    db2= ImagePlacement_Rightpp
    fileName= "I2S_FRONT_AQM.html"
    side='Front'
    dbName1=''
    dbName2=''
    figI2Sfront=PlotPlotly(pthF, side).Plot2subPlots(subplot_titles1, subplot_titles2, PlotTitle, db1, db2,dbName1,dbName2,fileName);
    
    ###########Back
    
    subplot_titles1="LEFT"
    subplot_titles2="RIGHT"
    PlotTitle= 'I2S- BACK'
    fileName= "I2S_BACK_AQM.html"
    side='Back'
    dbName1=''
    dbName2=''
    try:
        db1= ImagePlacement_Leftpp_BACK
        db2= ImagePlacement_Rightpp_BACK
        figI2SBack=PlotPlotly(pthF, side).Plot2subPlots(subplot_titles1, subplot_titles2, PlotTitle, db1, db2,dbName1,dbName2,fileName);
    except:
        1
    
    
    
############################Plot C2C #################################
###########Front & Back

if C2Cplot:
    subplot_titles1="FRONT"
    subplot_titles2="BACK"
    PlotTitle= 'C2C'
    fileName= "C2C_AQM.html"
    side='Front'
    dbName1='FRONT'
    dbName2='BACK'
    db1= DataPivotFront
    db2= pd.DataFrame()
    try:
        db2= DataPivotBack
    except:
        1
    figC2C=  PlotPlotly(pthF, side).Plot2subPlots(subplot_titles1, subplot_titles2, PlotTitle, db1, db2,dbName1,dbName2,fileName);
    
    
    
#############################################Scale ####################################################
#################Front
if ScalePlot:
    subplot_titles1="LEFT"
    subplot_titles2="RIGHT"
    PlotTitle= 'Scale-FRONT'
    db1= ScaleMaxMinDF_FRONTFLeft
    db2= ScaleMaxMinDF_FRONTRight
    fileName= "Scale_FRONT_AQM.html"
    side='Front'
    dbName1=''
    dbName2=''
    try:
        figScaleFRONT=PlotPlotly(pthF, side).Plot2subPlots(subplot_titles1, subplot_titles2, PlotTitle, db1, db2,dbName1,dbName2,fileName);
    
    except:
        1
    
    
    ###########Back
    
    subplot_titles1="LEFT"
    subplot_titles2="RIGHT"
    PlotTitle= 'Scale- BACK'
    fileName= "Scale_BACK_AQM.html"
    side='Back'
    dbName1=''
    dbName2=''
    try:
        db1= ScaleMaxMinDF_BACKFLeft
        db2= ScaleMaxMinDF_BACKRight
        figScaleBACK=PlotPlotly(pthF, side).Plot2subPlots(subplot_titles1, subplot_titles2, PlotTitle, db1, db2,dbName1,dbName2,fileName);
    except:
        1



#############################################WaveCahnge ####################################################
#################Front
if WaveChangePlot:
    PlotTitle= 'WaveCange-FRONT'
    fileName= "WaveChange_FRONT_AQM"
    side='Front'
    
    try:
        waveChangeFRONT=PlotPlotly(pthF, side).PlotWaveChange_WithMovingAVRG(WaveChangeDF_FRONT,indexJobNameDicFRONT,WaveJobPrintedDicFRONT,MoveAveWave, PlotTitle,fileName);
        # waveChangeFRONT=PlotPlotly(pthF, side).PlotWaveChange(WaveChangeDF_FRONT,indexJobNameDicFRONT,PlotTitle,fileName);
    
    except:
        1
        
        
    ###########Back
    
    PlotTitle= 'WaveCange-BACK'
    fileName= "WaveChange_BACK_AQM"
    side='Back'
    
    try:
        waveChangeBACK=PlotPlotly(pthF, side).PlotWaveChange_WithMovingAVRG(WaveChangeDF_BACK,indexJobNameDicBACK,WaveJobPrintedDicBACK,MoveAveWave, PlotTitle,fileName);
        # waveChangeBACK=PlotPlotly(pthF, side).PlotWaveChange(WaveChangeDF_BACK,indexJobNameDicBACK,PlotTitle,fileName);
    
    except:
        1

#############################################c2cCahnge ####################################################
#################Front
if c2cChangePlot:
    PlotTitle= 'c2cCange-FRONT'
    fileName= "c2cChange_FRONT_AQM"
    side='Front'
    
    try:
        c2cChangeFRONT=PlotPlotly(pthF, side).Plotc2cChange_WithMovingAVRG(c2cChangeDF_FRONT,indexJobNameDicFRONT,WaveJobPrintedDicFRONT,MoveAveWave, PlotTitle,fileName);
        # waveChangeFRONT=PlotPlotly(pthF, side).PlotWaveChange(WaveChangeDF_FRONT,indexJobNameDicFRONT,PlotTitle,fileName);
    
    except:
        1
        
        
    ###########Back
    
    PlotTitle= 'c2cCange-BACK'
    fileName= "c2cChange_BACK_AQM"
    side='Back'
    
    try:
        c2cChangeBACK=PlotPlotly(pthF, side).Plotc2cChange_WithMovingAVRG(c2cChangeDF_BACK,indexJobNameDicBACK,WaveJobPrintedDicBACK,MoveAveWave, PlotTitle,fileName);
        # waveChangeBACK=PlotPlotly(pthF, side).PlotWaveChange(WaveChangeDF_BACK,indexJobNameDicBACK,PlotTitle,fileName);
    
    except:
        1



############################################################################################
endFigure = time.time()
print(endFigure - startFigure)


# 
##### TILL HERE!!!!


# c2cChangeDF=c2cChangeDF_FRONT
# indexJobNameDic=indexJobNameDicFRONT
# WaveJobPrintedDic=WaveJobPrintedDicFRONT


# fig = go.Figure()
   
# c2cChangeDF = c2cChangeDF.dropna(axis=1)

   
   
 
   
# fig.add_trace(
# go.Scatter(y=list(c2cChangeDF[0]),
#             name='C2C '))
# fig.data[len(fig.data)-1].visible = 'legendonly';


# fig.add_trace(
# go.Scatter(y=list(c2cChangeDF[0].rolling(MoveAveWave).mean()),
#             name='C2C moving average '))
   
# # xx=  list(c2cChangeDF[0].rolling(30).mean() )
#     # ymax=max(WaveRawDataDic[ColorList[0]]-WaveDataWithMaxFilterDic[self.ColorList[0]])
# ymax=np.max(list(c2cChangeDF[0].rolling(MoveAveWave).mean())[MoveAveWave+10:])+20
# ymaxWaveJob=np.max(list(c2cChangeDF[0].rolling(MoveAveWave).mean())[MoveAveWave+10:])

# for key, value in indexJobNameDic.items():
#     fig.add_trace(go.Scatter(x=[key], y=[ymax],
#                             marker=dict(color="green", size=10),
#                             mode="markers",
#                             text=value[0],
#                             # font_size=18,
#                             hoverinfo='text'))
    
#     fig.data[len(fig.data)-1].showlegend = False
#     fig.add_vline(x=key, line_width=2, line_dash="dash", line_color="green")
# pxWave=0     
# for i ,(key, value) in enumerate(WaveJobPrintedDic.items()):
#     xWave=key+(1+int(JobLengthWave/10))
#     if i>0:
#         if abs(list(WaveJobPrintedDic.values())[i-1][1]- list(WaveJobPrintedDic.values())[i][1])<2:
#             xWave=pxWave + 1
#     fig.add_trace(go.Scatter(x=[xWave], y=[ymaxWaveJob],
#                             marker=dict(color="red", size=10),
#                             mode="markers",
#                             text=value,
#                             # font_size=18,
#                             hoverinfo='text'))
    
#     fig.data[len(fig.data)-1].showlegend = False
#     fig.add_vline(x=xWave, line_width=2,  line_color="red")
#     pxWave=xWave
   

# fig.update_layout(
#         hoverlabel=dict(
#             namelength=-1
#         )
#     )
# fig.update_layout(title=PlotPlotly(pthF, side).side+' '+PlotTitle)
   

# plot(fig,filename=PlotPlotly(pthF, side).side+' '+fileName+".html") 

# #########################################################

# c2c=DataPivotFront

# c2cChangeList=[];
# indexJobNameDic={}

# # JobNmeSORTED= list(CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').SortJobsByTime(CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').fldrs).values())

# ValidSortedJobListWithWave=[]

# for f in c2c.columns:
#     vlid,lngth=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CheckIfFileValid_forWave(f,JobLengthWave)
#     if vlid or ('WaveCalibration' in f):
#             ValidSortedJobListWithWave.append(f)
            
            
# WaveFilesInx=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').find_indexes_with_substring(ValidSortedJobListWithWave, 'WaveCalibration')
# WaveJobPrintedDic={}

# k=0
# for i,f in enumerate(ValidSortedJobListWithWave):
#     try:
    
#         c2cChangeList=c2cChangeList+list(c2c[f].dropna())
#         indexJobNameDic[len(c2cChangeList)-1]=[f,lngth]

#         if len(WaveFilesInx)>0:
#             if i>WaveFilesInx[k] or WaveFilesInx[k] == 0:
#                 inxForW=list(indexJobNameDic.keys())[len(list(indexJobNameDic.keys()))-2]
#                 WaveJobPrintedDic[inxForW]=[ValidSortedJobListWithWave[WaveFilesInx[k]],i]
#                 k=k+1;
#     except:
#               continue;
# # WaveChangeDF=WaveChangeDF_FRONT
# indexJobNameDic=indexJobNameDicFRONT
# WaveJobPrintedDic=WaveJobPrintedDicFRONT

# plt.figure()
# plt.plot(c2cChangeList)


# fig = go.Figure()

# WaveChangeDF = WaveChangeDF.dropna(axis=1)

# ColorList= list(WaveChangeDF.columns)

# for clr in ColorList:     
#     lineColor=clr;
  
    
#     if lineColor=='Yellow':
#         lineColor='gold';
    
#     fig.add_trace(
#     go.Scatter(y=WaveChangeDF[clr],line_color= lineColor,
#                 name='Wave Differance Left-Right '+' color '+clr))
#     fig.data[len(fig.data)-1].visible = 'legendonly';

    
#     fig.add_trace(
#     go.Scatter(y=WaveChangeDF[clr].rolling(MoveAveWave).mean(),line_color= lineColor,
#                 name='Wave Differance Left-Right- moving average of '+str(MoveAveWave)+' color '+clr))
    
    
#     # ymax=max(WaveRawDataDic[ColorList[0]]-WaveDataWithMaxFilterDic[self.ColorList[0]])
# ymax=np.max(WaveChangeDF[clr].rolling(MoveAveWave).mean())+20
# ymaxWaveJob=np.max(WaveChangeDF[clr].rolling(MoveAveWave).mean())
 
# for key, value in indexJobNameDic.items():
#      fig.add_trace(go.Scatter(x=[key], y=[ymax],
#                              marker=dict(color="green", size=10),
#                              mode="markers",
#                              text=value[0],
#                              # font_size=18,
#                              hoverinfo='text'))
     
#      fig.data[len(fig.data)-1].showlegend = False
#      fig.add_vline(x=key, line_width=2, line_dash="dash", line_color="green")
# pxWave=0     
# for i ,(key, value) in enumerate(WaveJobPrintedDic.items()):
#      xWave=key+(1+int(JobLengthWave/10))
#      if i>0:
#         if abs(list(WaveJobPrintedDic.values())[i-1][1]- list(WaveJobPrintedDic.values())[i][1])<2:
#             xWave=pxWave + 1
#      fig.add_trace(go.Scatter(x=[xWave], y=[ymaxWaveJob],
#                              marker=dict(color="red", size=10),
#                              mode="markers",
#                              text=value,
#                              # font_size=18,
#                              hoverinfo='text'))
     
#      fig.data[len(fig.data)-1].showlegend = False
#      fig.add_vline(x=xWave, line_width=2,  line_color="red")
#      pxWave=xWave
    
 
# fig.update_layout(
#          hoverlabel=dict(
#              namelength=-1
#          )
#      )
# fig.update_layout(title=PlotPlotly(pthF, side).side+' '+PlotTitle)
    
 
# plot(fig,filename=PlotPlotly(pthF, side).side+' '+fileName+".html") 

# # # ###################################################################



# WaveChangeList=[];
# indexJobNameDic={}

# # DataAllMeanColorSET1Left,DataAllMeanColorSET2Left,DataAllMeanColorSET3Left,colorDic = self.CalcMeanByColorForAllJobs('Registration_Left.csv')
# # DataAllMeanColorSET1Right,DataAllMeanColorSET2Right,DataAllMeanColorSET3Right,colorDic = self.CalcMeanByColorForAllJobs('Registration_Right.csv')




# MeregedDataAllMeanColorLeft= CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').LoadMeanColorPos_PickSide('Left');
# MeregedDataAllMeanColorRight= CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').LoadMeanColorPos_PickSide('Right');


# colorDic={}

# for i in MeregedDataAllMeanColorLeft.index:
#     colorDic[i]= MeregedDataAllMeanColorLeft['Ink\Sets'][i]

# DataAllMeanColorSET1Left=MeregedDataAllMeanColorLeft[['Ink\Sets','Set #1 X']].rename(index=colorDic)
# DataAllMeanColorSET2Left=MeregedDataAllMeanColorLeft[['Ink\Sets','Set #2 X']].rename(index=colorDic)
# DataAllMeanColorSET3Left=MeregedDataAllMeanColorLeft[['Ink\Sets','Set #3 X']].rename(index=colorDic)


# DataAllMeanColorSET1Right=MeregedDataAllMeanColorRight[['Ink\Sets','Set #1 X']].rename(index=colorDic)
# DataAllMeanColorSET2Right=MeregedDataAllMeanColorRight[['Ink\Sets','Set #2 X']].rename(index=colorDic)
# DataAllMeanColorSET3Right=MeregedDataAllMeanColorRight[['Ink\Sets','Set #3 X']].rename(index=colorDic)

# # MeregedDataAllMeanColor= self.LoadMeanColorPos();



# JobNmeSORTED= list(CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').SortJobsByTime(CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').fldrs).values())

# ValidSortedJobListWithWave=[]

# for f in JobNmeSORTED:
#     vlid,lngth=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CheckIfFileValid_forWave(f,JobLengthWave)
#     if vlid or ('WaveCalibration' in f):
#             ValidSortedJobListWithWave.append(f)
            
            
# WaveFilesInx=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').find_indexes_with_substring(ValidSortedJobListWithWave, 'WaveCalibration')
# WaveJobPrintedDic={}


# k=0
# for i,f in enumerate(ValidSortedJobListWithWave):
#     try:
    
#         C2Creg,indexNumberFailed = CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CalcC2CSingleSideColorPair('Registration_Left.csv','Registration_Right.csv',f,DataAllMeanColorSET1Left,DataAllMeanColorSET2Left,DataAllMeanColorSET3Left,DataAllMeanColorSET1Right,DataAllMeanColorSET2Right,DataAllMeanColorSET3Right)
#         WaveChangeList=WaveChangeList+C2Creg
#         indexJobNameDic[len(WaveChangeList)-1]=[f,lngth]

#         if len(WaveFilesInx)>0:
            
#             # indexJobNameDic[len(WaveChangeList)-1]=[f,ValidSortedJobListWithWave[WaveFilesInx[k]]]
#             if i>WaveFilesInx[k]:
#                 inxForW=list(indexJobNameDic.keys())[len(list(indexJobNameDic.keys()))-2]
#                 WaveJobPrintedDic[inxForW]=[ValidSortedJobListWithWave[WaveFilesInx[k]],i]
#                 k=k+1;

#     except:
#             continue;





# C2CregDF=pd.DataFrame(C2Creg)

# plt.figure()
# plt.plot(C2CregDF['Orange'])    