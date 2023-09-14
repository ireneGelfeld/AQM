# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 12:35:16 2022

@author: Ireneg
"""
 

#from IPython import get_ipython
#get_ipython().magic('reset -sf')

#######################PARAMS#########################################
global LoadTarget,GlobalScale,DistBetweenSets,JobLength,PanelNumber,DataPracent_toConcider,BaseDefult


#For 252
MarkSetVersion=252

if MarkSetVersion==252:

    
    GlobalScale = 0.9945 # Drop3 simplex = 0.9976, Duplex = 0.9984 ,,,, Drop5 Simplex = 0.9953, Duplex = 0.9945 
    DistBetweenSets =  125686/GlobalScale; 
    firstSetDistance=31053/GlobalScale; 

else:
#For 201
    GlobalScale = 0.9976
    DistBetweenSets =  102693;     
    firstSetDistance=31159;



JobLength = 0;
PanelNumber = 1;#Panel number for calculating Mean,STD,sum   
DataPracent_toConcider= 94 #in % --> for example => 90 % --> cuts Off 5 %  from top and 5 % from bottomm

DataPracent_toConcider= 95 #in % --> for example => 90 % --> cuts Off 5 %  from top and 5 % from bottomm
#DistBetweenSets =  126357  #102693 Duplex Drop3 = 125864,   Duplex-Drop5 126357
#Simplex Drop3 = 125965,  Simplex Drop5 = 126256 
LoadTarget = 0 ; #True from targets in the AQM or False - from the tabel 

StatisticsCalcStartPage = 100;

######## Plot Selection

Plot_correction = 1;
Plot_Image_Placment = 1;
Plot_RegForAllColors_Left = 1;
Plot_RegForAllColors_Right = 1;
Plot_Scale= 1;
Plot_MinMaxScale= 1;
Plot_MinMaxSETS= 1;


BaseDefult= 40103.853777615

######################################################################

import os


import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from datetime import datetime
import glob
from zipfile import ZipFile 
from pathlib import Path
import zipfile 
from io import BytesIO
import math

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import plotly.graph_objects as go
import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.subplots import make_subplots





class CalcC2C():
    def  __init__(self, pthF,side,fname,JobLength,LoadTarget): 
        self.pthF = pthF;
        self.side = side;
        self.fname=fname;
        self.JobLength = JobLength;
        self.LoadTarget=LoadTarget;
        
        
    def CheckForAI(self,pageSide):
        
        zip_file_path = self.pthF
        sub_folder = self.side+'/'+'RawResults'  # Path to the subfolder within the zip
        file_name = 'C2CRegistration_'+pageSide+'.csv'      # Name of the file you want to check
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            sub_folder_files = [name for name in zip_ref.namelist() if name.startswith(sub_folder)]
        
            if sub_folder+'/'+file_name in sub_folder_files:
                fname = file_name
            else:
                
                fname = 'Registration_'+pageSide+'.csv'
        
        return  fname;
    
    def LoadRawDataOLD(self):
        RawData=pd.read_csv(self.pthF+self.side+'/'+'RawResults'+'/'+self.fname);
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
    
    def LoadRawData(self):
        
        zip_file_path = self.pthF
        subdir_name_in_zip = self.side+'/'+'RawResults';
        file_name_in_zip = self.fname;
        
        RawData=self.GetFileFromZip(zip_file_path, subdir_name_in_zip, file_name_in_zip);
        l1=RawData.iloc[:,1].unique().tolist()
        RawDataSuccess=RawData[RawData['Registration Status']=='Success'].reset_index(drop=True)
        flatNumberFailed=(RawData[RawData['Registration Status']!='Success'].iloc[:,1].unique().tolist());
        return  RawDataSuccess,flatNumberFailed,l1; 
    
    def CalcMeanByColor(self,RawDataSuccess):
        DataAllMeanColorSET1 = RawDataSuccess.groupby(['Ink\Sets'])['Set #1 X'].mean().reset_index()
        DataAllMeanColorSET2 = RawDataSuccess.groupby(['Ink\Sets'])['Set #2 X'].mean().reset_index()
        DataAllMeanColorSET3 = RawDataSuccess.groupby(['Ink\Sets'])['Set #3 X'].mean().reset_index()       
        colorDic={}
        for i,cl in enumerate(DataAllMeanColorSET1['Ink\Sets']):
            colorDic[i]=cl
        DataAllMeanColorSET1=DataAllMeanColorSET1.rename(index=colorDic)
        DataAllMeanColorSET2=DataAllMeanColorSET2.rename(index=colorDic)
        DataAllMeanColorSET3=DataAllMeanColorSET3.rename(index=colorDic)
        
        return colorDic,DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3;
    
    def ConvertRowsToInt(self,RawDataSuccess):       
        RawDataSuccess['Set #1 X']  = RawDataSuccess['Set #1 X'].astype('int64');
        RawDataSuccess['Set #2 X']  = RawDataSuccess['Set #2 X'].astype('int64');
        RawDataSuccess['Set #3 X']  = RawDataSuccess['Set #3 X'].astype('int64');
        return RawDataSuccess;
    
    def LoadMeanColorPos(self):
        Lname=self.fname.split('.');
        LLname=Lname[0].split('_');
        pthComp=self.pthF.replace(self.pthF.split('/')[len(self.pthF.split('/'))-1],"")[:-1]
           
        MeregedDataAllMeanColor = pd.read_csv(pthComp +'/'+'MeregedDataAllMeanColor_'+self.side+'_'+LLname[1]+'.csv')    
            
            
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
        return value + (geometry[XYS][ink]) * target_distance[XYS] * (21.16666666 * 0.9976 / GlobalScale)
    
    def get_key(self,my_dict,val):
        for key, value in my_dict.items():
             if val == value:
                 return key
    def DataForCalcSCALE(self,PanelLengthInMM):
        
         RawDataSuccess,flatNumberFailed,l1= self.LoadRawData();
         
         RawDataSuccess=self.ConvertRowsToInt(RawDataSuccess);
         
         
         # MeregedDataAllMeanColor= self.LoadMeanColorPos();
         
         # colorDic={}
         # for i,cl in enumerate(MeregedDataAllMeanColor['Ink\Sets']):
         #     colorDic[i]=cl
         
         #valueSet1= MeregedDataAllMeanColor['Set #1 X'][self.get_key(colorDic,'Cyan')]

         valueSet1= 31451/GlobalScale;

         valueSet2= valueSet1+(125686/GlobalScale);

         valueSet3= valueSet2+(125686/GlobalScale);
         
         if LoadTarget:
            ColorList=RawDataSuccess.iloc[:,4].unique().tolist()
            colorDic={}
            for i,cl in enumerate(ColorList):
                colorDic[i]=cl            
            col=['Ink\Sets', 'Set #1 X', 'Set #2 X', 'Set #3 X'];            
            MeregedDataAllMeanColor= pd.DataFrame(columns=col)            
            MeregedDataAllMeanColor['Ink\Sets']=ColorList
            
            for key, value in colorDic.items():
                MeregedDataAllMeanColor['Set #1 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet1)
                MeregedDataAllMeanColor['Set #2 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet2)
                MeregedDataAllMeanColor['Set #3 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet3)
         else:
            MeregedDataAllMeanColor= self.LoadMeanColorPos();
            colorDic={}
            for i,cl in enumerate(MeregedDataAllMeanColor['Ink\Sets']):
                colorDic[i]=cl
         
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
         
         x=[0,1,2];

         slp=[]

            
         for inx in St1dataAllColors.index:
             y=list(RefSETloc.iloc[get_key(colorDic,inx),:])
             z = np.polyfit(x, y, 1)
             p = np.poly1d(z)
             slp.append(list(p)[0])
             
         RefSETloc.insert(3,'Slop' ,slp)
             
         
         Scale=St1dataAllColors;


         for c in St1dataAllColors.columns:
             for inx in St1dataAllColors.index:
                 try:
                     y= PanelColorSet[c][inx]
                     z = np.polyfit(x, y, 1)
                     p = np.poly1d(z)
                 
                     Scale[c][inx]=(RefSETloc['Slop'][inx]/list(p)[0]-1)*PanelLengthInMM*1000
                 except:
                     continue;
         
         ## Calc Scale Max - Min        
         SMinMax={}
         for c in Scale.columns:
             SMinMax[c]=[np.max(Scale[c])-np.min(Scale[c])];
             
         ScaleMaxMin= pd.DataFrame(SMinMax)
         
         
         
         
         return St1dataAllColors,St2dataAllColors,St3dataAllColors,RefSETloc,Scale,ScaleMaxMin,colorDic
     
        
     
    
    def CalcC2CSingleSidePerSetDEFULTvalue(self):
        RawDataSuccess,flatNumberFailed,l1= self.LoadRawData();
        
        RawDataSuccess=self.ConvertRowsToInt(RawDataSuccess);
        

        # MeregedDataAllMeanColor= self.LoadMeanColorPos();
        
        # colorDic={}
        # for i,cl in enumerate(MeregedDataAllMeanColor['Ink\Sets']):
        #     colorDic[i]=cl
        
        #valueSet1= MeregedDataAllMeanColor['Set #1 X'][self.get_key(colorDic,'Cyan')]
        valueSet1= 31451/GlobalScale;

        valueSet2= valueSet1+(125686/GlobalScale);

        valueSet3= valueSet2+(125686/GlobalScale);
        
        if LoadTarget:
            ColorList=RawDataSuccess.iloc[:,4].unique().tolist()
            colorDic={}
            for i,cl in enumerate(ColorList):
                colorDic[i]=cl            
            col=['Ink\Sets', 'Set #1 X', 'Set #2 X', 'Set #3 X'];            
            MeregedDataAllMeanColor= pd.DataFrame(columns=col)            
            MeregedDataAllMeanColor['Ink\Sets']=ColorList
            
            for key, value in colorDic.items():
                MeregedDataAllMeanColor['Set #1 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet1)
                MeregedDataAllMeanColor['Set #2 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet2)
                MeregedDataAllMeanColor['Set #3 X'][self.get_key(colorDic,value)]= self.f('X',value,valueSet3)
        else:
            MeregedDataAllMeanColor= self.LoadMeanColorPos();
            colorDic={}
            for i,cl in enumerate(MeregedDataAllMeanColor['Ink\Sets']):
                colorDic[i]=cl
        
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
                    St1dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x)-DataAllMeanColorSET1['Set #1 X'][FlatIDdata['Ink\Sets'][i]];
                for i,x in enumerate(FlatIDdata['Set #2 X']):
                    St2dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x)-DataAllMeanColorSET2['Set #2 X'][FlatIDdata['Ink\Sets'][i]];
                for i,x in enumerate(FlatIDdata['Set #3 X']):
                    St3dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x)-DataAllMeanColorSET3['Set #3 X'][FlatIDdata['Ink\Sets'][i]];
                
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

        return St1dataAllColors,St2dataAllColors,St3dataAllColors,indexNumberFailed,MeregedDataAllMeanColor;          
        
    def CalcC2CSingleSidePerSet(self):
        RawDataSuccess,flatNumberFailed,l1= self.LoadRawData();
        
        RawDataSuccess=self.ConvertRowsToInt(RawDataSuccess);
        
        #colorDic,DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3=self.CalcMeanByColor(RawDataSuccess);
        # Lname=self.fname.split('.');
        # LLname=Lname[0].split('_');
        
        # pthComp=self.pthF.split('/');
        # RecPath = pthComp[0] + '/';
        # for i,pt in enumerate(pthComp):
        #     if i>0 and i<len(pthComp)-2:
        #         RecPath= RecPath + pt + '/';
            
        # # RecPath= pthComp[0]+'/'+pthComp[1]+'/'+pthComp[2]+'/'+pthComp[3]+'/'+pthComp[4]+'/'+pthComp[5]+'/'+pthComp[6]+'/'+pthComp[7]+'/'+pthComp[8];
        # MeregedDataAllMeanColor = pd.read_csv(RecPath +'/'+'MeregedDataAllMeanColor_'+LLname[1]+'.csv')
        
        MeregedDataAllMeanColor= self.LoadMeanColorPos();
        
        
        colorDic={}
        for i,cl in enumerate(MeregedDataAllMeanColor['Ink\Sets']):
            colorDic[i]=cl
        
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
                    St1dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x)-DataAllMeanColorSET1['Set #1 X'][FlatIDdata['Ink\Sets'][i]];
                for i,x in enumerate(FlatIDdata['Set #2 X']):
                    St2dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x)-DataAllMeanColorSET2['Set #2 X'][FlatIDdata['Ink\Sets'][i]];
                for i,x in enumerate(FlatIDdata['Set #3 X']):
                    St3dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x)-DataAllMeanColorSET3['Set #3 X'][FlatIDdata['Ink\Sets'][i]];
                
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

        return St1dataAllColors,St2dataAllColors,St3dataAllColors,indexNumberFailed;
                        
class DispImagePlacment():
    def __init__(self, pthF,f,side,pageSide,JobLength):
        ''' constructor ''' 
        self.pthF = pthF;
        self.f=f;
        self.side = side;
        self.pageSide=pageSide;
        self.JobLength = JobLength;

        
    def CheckIfFileValid(self):
        vlid=False;
        dbtmp=pd.DataFrame();
        # pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
        
        zip_file_path = self.pthF+'/'+f
        subdir_name_in_zip = self.side+'/'+'RawResults';
        file_name_in_zip = 'ImagePlacement_Left.csv';
        
        try:
            dbtmp=self.GetFileFromZip(zip_file_path,subdir_name_in_zip,file_name_in_zip);
            if len(dbtmp['Flat Id'])>self.JobLength:
                vlid= True;
        except:
            vlid=False;
                        
        return vlid;



        
    # def CheckIfFileValid(self):
    #     vlid=False;
    #     dbtmp=pd.DataFrame();
    #     pthFf=self.pthF+'/'+self.side+'/'+'RawResults';
    #     os.chdir(pthFf);
    #     fname='ImagePlacement_Left.csv';
    #     if os.path.exists(fname) and Path(fname).stat().st_size:
    #         dbtmp=pd.read_csv(fname,usecols=[0]);
    #         if len(dbtmp['Flat Id'])>self.JobLength:
    #             vlid= True;
    #     return vlid;
    
    # def ExtractBaslineAndAppliedCorrection(self):
    #     ImagePlacement_res=pd.read_csv(self.pthF+'/'+self.side+'/'+'AnalysisResults/ImagePlacementAnalysis_'+self.side+'.csv')
    #     BaseLine=ImagePlacement_res['Expected X'][0];
    #     dbtmpSentCrr=ImagePlacement_res[(ImagePlacement_res['Correction Sent-Applied']) == 'Sent'].reset_index(drop=True)

    #     return BaseLine,dbtmpSentCrr              

    def GetFileFromZip(self,zip_file_path,subdir_name_in_zip,file_name_in_zip):
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_path_in_zip = subdir_name_in_zip + "/" + file_name_in_zip
            with zip_ref.open(file_path_in_zip) as file:
                # read the contents of the file into memory
                file_content = file.read()
                
                # convert the file content to a pandas dataframe
                df = pd.read_csv(BytesIO(file_content))
        return  df;     

    def ExtractBaslineAndAppliedCorrection(self):
        # magePlacement_res=pd.read_csv(self.pthF+f+'/'+self.side+'/'+'AnalysisResults/ImagePlacementAnalysis_'+self.side+'.csv',usecols=[5])
        zip_file_path = self.pthF+'/'+self.f
        subdir_name_in_zip = self.side+'/'+'AnalysisResults';
        file_name_in_zip = 'ImagePlacementAnalysis_'+self.side+'.csv';
        try:
            ImagePlacement_res=self.GetFileFromZip(zip_file_path,subdir_name_in_zip,file_name_in_zip);
            dbtmpSentCrr=ImagePlacement_res[(ImagePlacement_res['Correction Sent-Applied']) == 'Sent'].reset_index(drop=True)

        except:
            ImagePlacement_res=pd.DataFrame(columns=['Expected X'])
            ImagePlacement_res['Expected X']= [BaseDefult]
            dbtmpSentCrr= pd.DataFrame(columns=['X Correction Moving Avarage'])
        BaseLine=ImagePlacement_res['Expected X'][0];

        return BaseLine,dbtmpSentCrr;
    
    
    def ReadImagePlacmentAndCorrectionData(self):
                os.chdir(self.pthF);
                dbtmpSentCrr=pd.DataFrame()
                ImagePlacement_pp=pd.DataFrame()
                PanelCorrection=pd.DataFrame()
                BaseLine=0;
                fname='ImagePlacement_'+self.pageSide+'.csv';
                pthFf=self.pthF+'/'+self.side+'/'+'RawResults/';
                pthCrr=pthF+'/'+self.side+'/CorrectionOperators'+'/';
                fcorr='PanelCorrection.csv'
                #if self.CheckIfFileValid():
                BaseLine,dbtmpSentCrr=self.ExtractBaslineAndAppliedCorrection();
                
                zip_file_path = self.pthF+'/'+self.f
                subdir_name_in_zip = self.side+'/'+'RawResults';
                file_name_in_zip = fname;
                
                # dbtmp=pd.read_csv(fname);
                dbtmp=self.GetFileFromZip(zip_file_path,subdir_name_in_zip,file_name_in_zip);
                ImagePlacement_pp[f]=dbtmp['T1->X']-BaseLine
                ImagePlacement_pp['Flat Id']=dbtmp['Flat Id'];
                ImagePlacement_pp['Panel Id']=dbtmp['Panel Id'];
                try:
                   # os.chdir(pthCrr);
                   # PanelCorrection=pd.read_csv(fcorr)
                   
                   zip_file_path = self.pthF+'/'+self.f
                   subdir_name_in_zip = self.side+'/CorrectionOperators';
                   file_name_in_zip = fcorr;
                   
                   PanelCorrection=self.GetFileFromZip(zip_file_path,subdir_name_in_zip,file_name_in_zip);
                   
                except:
                    1
              
                    
                return  ImagePlacement_pp,dbtmpSentCrr,BaseLine,PanelCorrection; 


def get_key(my_dict,val):
    for key, value in my_dict.items():
          if val == value:
              return key

def max_minSetS(StdataAllColors,colorDic):
    C2CMinMax=pd.DataFrame()
    colorList=[];
    for c in StdataAllColors.columns:
        C2CMinMax[c]=[np.max(StdataAllColors[c])-np.min(StdataAllColors[c])];
        try:
            colorList.append(colorDic[np.argmax(list(StdataAllColors[c]))]+'-'+colorDic[np.argmin(list(StdataAllColors[c]))])
        except:
            1
    return C2CMinMax,colorList;  
        
def findMinMaxDiv(Vector):
    
    MaxDiv = list(Vector)
    MaxDivNotNaN = [x for x in MaxDiv if not math.isnan(x)]
    DataPracent_toIgnor=  ((100- DataPracent_toConcider)/2)
    percentile_99 = np.percentile(MaxDivNotNaN, DataPracent_toConcider+DataPracent_toIgnor)
    percentile_1 = np.percentile(MaxDivNotNaN, DataPracent_toIgnor )
    
    filtered_data1 = [x for x in MaxDivNotNaN if  x >= percentile_1 and x <= percentile_99]
    
    return  max(filtered_data1)-min(filtered_data1),max(filtered_data1),min(filtered_data1)        
####################################################################################################


from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthFf = filedialog.askopenfilename()

# pthF='C:/Users/Ireneg/OneDrive - Landa Group/שולחן העבודה/AQM/all (1)/QCS Production_0 Archive 22-12-2021 11-35-17'
f=pthFf.split('/')[len(pthFf.split('/'))-1]

pthF=pthFf.replace(pthFf.split('/')[len(pthFf.split('/'))-1],"")[:-1]

import time

start = time.time()

############## Calc Image Placment

ImagePlacement_ppFRONT,dbtmpSentCrrFRONT,BaseLineFRONT,PanelCorrectionFRONT=DispImagePlacment(pthF,f,'Front','Left',JobLength).ReadImagePlacmentAndCorrectionData();


# ImagePlacement_ppFRONT,dbtmpSentCrrFRONT,BaseLineFRONT,PanelCorrectionFRONT=ReadImagePlacmentDataAndCorrection(pthF,'Front',fname,fnameResFRONT,JobLength)
try:
    CyanI2SFRONT=PanelCorrectionFRONT[(PanelCorrectionFRONT['Color']) == 'Cyan'].reset_index(drop=True)
    
    CyanI2SPnlnumFRONT=pd.DataFrame();
    tmpdb=pd.DataFrame();
    for j in range(1,12):
       tmpdb=CyanI2SFRONT[CyanI2SFRONT['Panel']==j].reset_index(drop=True);
       CyanI2SPnlnumFRONT=pd.concat((CyanI2SPnlnumFRONT, tmpdb['C2C_X Correction'].rename(str(j))), axis=1)
    
    
    CyanI2SPanelnumFRONT=CyanI2SFRONT[CyanI2SFRONT['Panel']==PanelNumber].reset_index(drop=True)
    try:
        ImagePlacement_ppBACK,dbtmpSentCrrBACK,BaseLineBACK,PanelCorrectionBACK=DispImagePlacment(pthF+'/',f,'Back','Left',JobLength).ReadImagePlacmentAndCorrectionData();
        CyanI2SBACK=PanelCorrectionBACK[(PanelCorrectionBACK['Color']) == 'Cyan'].reset_index(drop=True)
        CyanI2SPnlnumBACK=pd.DataFrame();
        tmpdb=pd.DataFrame();
        for j in range(1,12):
           tmpdb=CyanI2SBACK[CyanI2SBACK['Panel']==j].reset_index(drop=True);
           CyanI2SPnlnumBACK=pd.concat((CyanI2SPnlnumBACK, tmpdb['C2C_X Correction'].rename(str(j))), axis=1);
        CyanI2SPanelnumBACK=CyanI2SBACK[CyanI2SBACK['Panel']==PanelNumber].reset_index(drop=True);
    except:
        1
except:
    1        

#pthF=pthF+'/';

##################### Calc Reg for each Color
side='Front';

fname = CalcC2C(pthF+'/'+f,side,'',JobLength,LoadTarget).CheckForAI('Left')
# fname='Registration_Left.csv'
# St1dataAllColorsFrontLeft,St2dataAllColorsFrontLeft,St3dataAllColorsFrontLeft,indexNumberFailed=CalcC2C(pthF+'/',side,fname,JobLength).CalcC2CSingleSidePerSet()


St1dataAllColorsFrontLeft,St2dataAllColorsFrontLeft,St3dataAllColorsFrontLeft,indexNumberFailed,MeregedDataAllMeanColorFRONTLeft=CalcC2C(pthF+'/'+f,side,fname,JobLength,LoadTarget).CalcC2CSingleSidePerSetDEFULTvalue()


 
side='Front';
fname = CalcC2C(pthF+'/'+f,side,'',JobLength,LoadTarget).CheckForAI('Right')

# fname='Registration_Right.csv'
# St1dataAllColorsFrontRight,St2dataAllColorsFrontRight,St3dataAllColorsFrontRight,indexNumberFailed=CalcC2C(pthF+'/',side,fname,JobLength).CalcC2CSingleSidePerSet()
St1dataAllColorsFrontRight,St2dataAllColorsFrontRight,St3dataAllColorsFrontRight,indexNumberFailed,MeregedDataAllMeanColorFRONTRight=CalcC2C(pthF+'/'+f,side,fname,JobLength,LoadTarget).CalcC2CSingleSidePerSetDEFULTvalue()
# 
inx=list(St1dataAllColorsFrontLeft.index)
St1dataAllColorsFrontMean=pd.DataFrame()
St2dataAllColorsFrontMean=pd.DataFrame()
St3dataAllColorsFrontMean=pd.DataFrame()

for i in inx:
    St1dataAllColorsFrontMean[i]=(St1dataAllColorsFrontLeft.loc[i]+St1dataAllColorsFrontRight.loc[i])/2;
    St2dataAllColorsFrontMean[i]=(St2dataAllColorsFrontLeft.loc[i]+St2dataAllColorsFrontRight.loc[i])/2;
    St3dataAllColorsFrontMean[i]=(St3dataAllColorsFrontLeft.loc[i]+St3dataAllColorsFrontRight.loc[i])/2


try:
    side='Back';
    # fname='Registration_Left.csv'
    fname = CalcC2C(pthF+'/'+f,side,'',JobLength,LoadTarget).CheckForAI('Left')

    St1dataAllColorsBackLeft,St2dataAllColorsBackLeft,St3dataAllColorsBackLeft,indexNumberFailed,MeregedDataAllMeanColorBackLeft=CalcC2C(pthF+'/'+f ,side,fname,JobLength,LoadTarget).CalcC2CSingleSidePerSetDEFULTvalue()
        
    side='Back';
    # fname='Registration_Right.csv'
    fname = CalcC2C(pthF+'/'+f,side,fname,JobLength,LoadTarget).CheckForAI('Right')

    St1dataAllColorsBackRight,St2dataAllColorsBackRight,St3dataAllColorsBackRight,indexNumberFailed,MeregedDataAllMeanColorBackRight=CalcC2C(pthF+'/'+f ,side,fname,JobLength,LoadTarget).CalcC2CSingleSidePerSetDEFULTvalue()
    
    inx=list(St1dataAllColorsBackLeft.index)
    St1dataAllColorsBackMean=pd.DataFrame()
    St2dataAllColorsBackMean=pd.DataFrame()
    St3dataAllColorsBackMean=pd.DataFrame()
    
    for i in inx:
        St1dataAllColorsBackMean[i]=(St1dataAllColorsBackLeft.loc[i]+St1dataAllColorsBackRight.loc[i])/2;
        St2dataAllColorsBackMean[i]=(St2dataAllColorsBackLeft.loc[i]+St2dataAllColorsBackRight.loc[i])/2;
        St3dataAllColorsBackMean[i]=(St3dataAllColorsBackLeft.loc[i]+St3dataAllColorsBackRight.loc[i])/2;
except:
    1




############## Calc Max Min Set 
if Plot_MinMaxSETS:
    colorDicRight={}
    colorDicLeft={}
    
    C2C_set1MaxMinFrontRight= pd.DataFrame();
    C2C_set2MaxMinFrontRight= pd.DataFrame();
    C2C_set3MaxMinFrontRight= pd.DataFrame();
    
    C2C_set1MaxMinFrontLeft= pd.DataFrame();
    C2C_set2MaxMinFrontLeft= pd.DataFrame();
    C2C_set3MaxMinFrontLeft= pd.DataFrame();
    
    
    C2C_set1MaxMinBackRight= pd.DataFrame();
    C2C_set2MaxMinBackRight= pd.DataFrame();
    C2C_set3MaxMinBackRight= pd.DataFrame();
    
    C2C_set1MaxMinBackLeft= pd.DataFrame();
    C2C_set2MaxMinBackLeft= pd.DataFrame();
    C2C_set3MaxMinBackLeft= pd.DataFrame();
    
    C2C_set1MaxMinFrontRight,colorListset1MaxMinFrontRight=max_minSetS(St1dataAllColorsFrontRight,colorDicRight)
    C2C_set2MaxMinFrontRight,colorListset2MaxMinFrontRight=max_minSetS(St2dataAllColorsFrontRight,colorDicRight)
    C2C_set3MaxMinFrontRight,colorListset3MaxMinFrontRight=max_minSetS(St3dataAllColorsFrontRight,colorDicRight)
    
    
    C2C_set1MaxMinFrontLeft,colorListset1MaxMinFrontLeft=max_minSetS(St1dataAllColorsFrontLeft,colorDicLeft)
    C2C_set2MaxMinFrontLeft,colorListset2MaxMinFrontLeft=max_minSetS(St2dataAllColorsFrontLeft,colorDicLeft)
    C2C_set3MaxMinFrontLeft,colorListset3MaxMinFrontLeft=max_minSetS(St3dataAllColorsFrontLeft,colorDicLeft)
    
    C2C_MaxFront = pd.DataFrame();
    
    for c in C2C_set1MaxMinFrontLeft.columns:
        C2C_MaxFront[c]=[np.max([C2C_set1MaxMinFrontRight[c][0],C2C_set2MaxMinFrontRight[c][0],C2C_set3MaxMinFrontRight[c][0],C2C_set1MaxMinFrontLeft[c][0],C2C_set2MaxMinFrontLeft[c][0],C2C_set3MaxMinFrontLeft[c][0]])];
    
    try:
        C2C_set1MaxMinBackRight,colorListset1MaxMinBackRight=max_minSetS(St1dataAllColorsBackRight,colorDicRight)
        C2C_set2MaxMinBackRight,colorListset2MaxMinBackRight=max_minSetS(St2dataAllColorsBackRight,colorDicRight)
        C2C_set3MaxMinBackRight,colorListset3MaxMinBackRight=max_minSetS(St3dataAllColorsBackRight,colorDicRight)
        
        
        C2C_set1MaxMinBackLeft,colorListset1MaxMinBackLeft=max_minSetS(St1dataAllColorsBackLeft,colorDicLeft)
        C2C_set2MaxMinBackLeft,colorListset2MaxMinBackLeft=max_minSetS(St2dataAllColorsBackLeft,colorDicLeft)
        C2C_set3MaxMinBackLeft,colorListset3MaxMinBackLeft=max_minSetS(St3dataAllColorsBackLeft,colorDicLeft)
        
        C2C_MaxBack = pd.DataFrame();
    
        
        for c in C2C_set1MaxMinBackLeft.columns:
            C2C_MaxBack[c]= [np.max([C2C_set1MaxMinBackRight[c][0],C2C_set2MaxMinBackRight[c][0],C2C_set3MaxMinBackRight[c][0],C2C_set1MaxMinBackLeft[c][0],C2C_set2MaxMinBackLeft[c][0],C2C_set3MaxMinBackLeft[c][0]])]
    
        
        
    except:
        1    
######################################### Debug
side='Front';
# fname='Registration_Left.csv'
fname = CalcC2C(pthF+'/'+f,side,'',JobLength,LoadTarget).CheckForAI('Left')

RawDataSuccess,flatNumberFailed,l1= CalcC2C(pthF+'/'+f ,side,fname,JobLength,LoadTarget).LoadRawData();
        
# RawDataSuccess=CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).ConvertRowsToInt(RawDataSuccess);


# MeregedDataAllMeanColor= CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).LoadMeanColorPos();


# #########################
# Lname=CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).fname.split('.');
# LLname=Lname[0].split('_');

# pthComp=CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).pthF.split('/');
# RecPath = pthComp[0] + '/';
# for i,pt in enumerate(pthComp):
#     if i>0 and i<len(pthComp)-2:
#         RecPath= RecPath + pt + '/';
    
# # RecPath= pthComp[0]+'/'+pthComp[1]+'/'+pthComp[2]+'/'+pthComp[3]+'/'+pthComp[4]+'/'+pthComp[5]+'/'+pthComp[6]+'/'+pthComp[7]+'/'+pthComp[8];
# if CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).LoadTarget:
#     MeregedDataAllMeanColor = pd.read_csv(RecPath +'/'+'MeregedDataAllTargetColor_'+self.side+'_'+LLname[1]+'.csv')

# else:
#     MeregedDataAllMeanColor = pd.read_csv(RecPath +'/'+'MeregedDataAllMeanColor_'+CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).side+'_'+LLname[1]+'.csv')    
    
# ################################            


# colorDic={}
# for i,cl in enumerate(MeregedDataAllMeanColor['Ink\Sets']):
#     colorDic[i]=cl

# valueSet1= MeregedDataAllMeanColor['Set #1 X'][CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).get_key(colorDic,'Cyan')]

# valueSet2= valueSet1+(102693/GlobalScale);

# valueSet3= valueSet2+(102693/GlobalScale);

# for key, value in colorDic.items():
#     MeregedDataAllMeanColor['Set #1 X'][CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).get_key(colorDic,value)]= CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).f('X',value,valueSet1)
#     MeregedDataAllMeanColor['Set #2 X'][CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).get_key(colorDic,value)]= CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).f('X',value,valueSet2)
#     MeregedDataAllMeanColor['Set #3 X'][CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).get_key(colorDic,value)]= CalcC2C(pthF+'/',side,fname,JobLength,LoadTarget).f('X',value,valueSet3)


# DataAllMeanColorSET1=MeregedDataAllMeanColor[['Set #1 X','Ink\Sets']].rename(index=colorDic);
# DataAllMeanColorSET2=MeregedDataAllMeanColor[['Set #2 X','Ink\Sets']].rename(index=colorDic);
# DataAllMeanColorSET3=MeregedDataAllMeanColor[['Set #3 X','Ink\Sets']].rename(index=colorDic);

# indexNumberFailed=[]
# col=[];
# [col.append(str(j)) for j in range(len(l1))];
# St1dataAllColors=pd.DataFrame(columns=col,index=colorDic.values());
# St2dataAllColors=pd.DataFrame(columns=col,index=colorDic.values());
# St3dataAllColors=pd.DataFrame(columns=col,index=colorDic.values());


# for j,l in enumerate(l1):
#     if not(l in flatNumberFailed):
#         FlatIDdata=RawDataSuccess[RawDataSuccess['Flat Id']==l].reset_index();
        
#         for i,x in enumerate(FlatIDdata['Set #1 X']):
#             St1dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x)-DataAllMeanColorSET1['Set #1 X'][FlatIDdata['Ink\Sets'][i]];
#         for i,x in enumerate(FlatIDdata['Set #2 X']):
#             St2dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x)-DataAllMeanColorSET2['Set #2 X'][FlatIDdata['Ink\Sets'][i]];
#         for i,x in enumerate(FlatIDdata['Set #3 X']):
#             St3dataAllColors[str(j)][FlatIDdata['Ink\Sets'][i]]=int(x)-DataAllMeanColorSET3['Set #3 X'][FlatIDdata['Ink\Sets'][i]];
        
#     else:
#         indexNumberFailed.append(j)
#         if j>0:
#             St1dataAllColors[str(j)]=St1dataAllColors[str(j-1)]
#             St2dataAllColors[str(j)]=St2dataAllColors[str(j-1)]
#             St3dataAllColors[str(j)]=St3dataAllColors[str(j-1)]
#         else:
#             St1dataAllColors[str(j)]=0
#             St2dataAllColors[str(j)]=0
#             St3dataAllColors[str(j)]=0

#return St1dataAllColors,St2dataAllColors,St3dataAllColors,indexNumberFailed;          
        




     

############################## calc SCALE
if Plot_Scale:
    side='Front';
    fname = CalcC2C(pthF+'/'+f,side,'',JobLength,LoadTarget).CheckForAI('Left')

    # fname='Registration_Left.csv'
    PanelLengthInMM = 650;
    # St1dataAllColorsFrontLeft,St2dataAllColorsFrontLeft,St3dataAllColorsFrontLeft,indexNumberFailed=CalcC2C(pthF+'/',side,fname,JobLength).CalcC2CSingleSidePerSet()
    
    St1dataAllColorsFrontFullLeft,St2dataAllColorsFrontFullLeft,St3dataAllColorsFrontFullLeft,RefSETlocLeft,ScaleFRONTLeft,ScaleLeftFRONTMaxMin,colorDicLeft=CalcC2C(pthF+'/'+f,side,fname,JobLength,LoadTarget).DataForCalcSCALE(PanelLengthInMM)
    
    
    
        
    side='Front';
    fname = CalcC2C(pthF+'/'+f,side,'',JobLength,LoadTarget).CheckForAI('Right')

    # fname='Registration_Right.csv'
    #PanelLengthInMM = 650;
    # St1dataAllColorsFrontRight,St2dataAllColorsFrontRight,St3dataAllColorsFrontRight,indexNumberFailed=CalcC2C(pthF+'/',side,fname,JobLength).CalcC2CSingleSidePerSet()
    St1dataAllColorsFrontFullRight,St2dataAllColorsFrontFullRight,St3dataAllColorsFrontFullRight,RefSETlocRight,ScaleFRONTRight,ScaleRightFRONTMaxMin,colorDicRight=CalcC2C(pthF+'/'+f,side,fname,JobLength,LoadTarget).DataForCalcSCALE(PanelLengthInMM)
    # 
    
    
    try:
        side='Back';
        # fname='Registration_Left.csv'
        fname = CalcC2C(pthF+'/'+f,side,'',JobLength,LoadTarget).CheckForAI('Left')

    #    PanelLengthInMM = 650;
        # St1dataAllColorsFrontLeft,St2dataAllColorsFrontLeft,St3dataAllColorsFrontLeft,indexNumberFailed=CalcC2C(pthF+'/',side,fname,JobLength).CalcC2CSingleSidePerSet()
    
        St1dataAllColorsBackFullLeft,St2dataAllColorsBackFullLeft,St3dataAllColorsBackFullLeft,RefSETlocLeftBACK,ScaleBACKLeft,ScaleLeftBACKMaxMin,colorDicLeft=CalcC2C(pthF+'/'+f,side,fname,JobLength,LoadTarget).DataForCalcSCALE(PanelLengthInMM)
    
    
    
            
        side='Back';
        # fname='Registration_Right.csv'
        fname = CalcC2C(pthF+'/'+f,side,fname,JobLength,LoadTarget).CheckForAI('Right')

    #    PanelLengthInMM = 650;
        # St1dataAllColorsFrontRight,St2dataAllColorsFrontRight,St3dataAllColorsFrontRight,indexNumberFailed=CalcC2C(pthF+'/',side,fname,JobLength).CalcC2CSingleSidePerSet()
        St1dataAllColorsBackFullRight,St2dataAllColorsBackFullRight,St3dataAllColorsBackFullRight,RefSETlocRightBACK,ScaleBACKRight,ScaleRightBACKMaxMin,colorDicRight=CalcC2C(pthF+'/'+f,side,fname,JobLength,LoadTarget).DataForCalcSCALE(PanelLengthInMM)
    except:
        1








########################################################################################################
pthComp=pthF.split('/');
RecPath = pthComp[0] + '/';
for i,pt in enumerate(pthComp):
    if i>0 and i<len(pthComp)-1:
        RecPath= RecPath + pt + '/';


end = time.time()
print(end - start)

# os.chdir(RecPath)
os.chdir(pthF)



                    
############################################################################################################
################### PLOT SIGNALS ############################################
if Plot_correction:
    start = time.time()
    
    try:
        fig0 = go.Figure()
        
        frontTitle="FRONT correction";
        backTitle="BACK correction";
        
        fig0 = make_subplots(rows=2, cols=1,subplot_titles=(frontTitle, backTitle), vertical_spacing=0.1, shared_xaxes=True,print_grid=True)
        
        
        fig0.add_trace(
        go.Scatter(x=list(ImagePlacement_ppFRONT['Flat Id']),y=list(ImagePlacement_ppFRONT[f]),
                    name=f),row=1, col=1)
        
        for j in range(1,12):
            fig0.add_trace(
            go.Scatter(x=list(dbtmpSentCrrFRONT['Flat Id']),y=list(CyanI2SPnlnumFRONT[str(j)]),
                        name=f+' Panel'+str(j)),row=1, col=1)
        try:
            fig0.add_trace(
                go.Scatter(x=list(ImagePlacement_ppBACK['Flat Id']),y=list(ImagePlacement_ppBACK[f]),
                            name=f+' BACK'),row=2, col=1)
            for j in range(1,12):
                fig0.add_trace(
                go.Scatter(x=list(dbtmpSentCrrBACK['Flat Id']),y=list(CyanI2SPnlnumBACK[str(j)]),
                            name=f+'BACK Panel'+str(j)),row=2, col=1)
        except:
            1;
        
        fig0.update_layout(title=f+' All Panels')
        
        
        fig0.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
        
        
    
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        plot(fig0,filename=f+" AQM_Correction"+ dt_string +".html") 
        fig0.show() 
    
    except:
        1    


################### PLOT SIGNALS Image_Placment_plot############################################
if Plot_Image_Placment:
    try:    
        fig = go.Figure()
        
        
        
        frontC=(np.sum(CyanI2SPanelnumFRONT['C2C_X Correction']));
        frontCMean=(np.average(CyanI2SPanelnumFRONT['C2C_X Correction']));
        frontCstd=(np.std(CyanI2SPanelnumFRONT['C2C_X Correction']));
        
        try:
            backC=(np.sum(CyanI2SPanelnumBACK['C2C_X Correction']));
            backCMean=(np.average(CyanI2SPanelnumBACK['C2C_X Correction']));
            backCstd=(np.std(CyanI2SPanelnumBACK['C2C_X Correction']));
        
        except:
            backC=0;
            backCMean=0;
            backCstd=0;
        
        ##I2S

        frontI2SMean=(np.average(ImagePlacement_ppFRONT[f][StatisticsCalcStartPage:]));
        frontI2Sstd=(np.std(ImagePlacement_ppFRONT[f][StatisticsCalcStartPage:]));
        maxDivfrontI2S,maxNfrontI2S,minNfrontI2S=  findMinMaxDiv(ImagePlacement_ppFRONT[f][StatisticsCalcStartPage:])  

        try:
            backI2SMean=(np.average(ImagePlacement_ppBACK[f][StatisticsCalcStartPage:]));
            backI2Sstd=(np.std(ImagePlacement_ppBACK[f][StatisticsCalcStartPage:]));
            maxDivbackI2S,maxNbackI2S,minNbackI2S=  findMinMaxDiv(ImagePlacement_ppBACK[f][StatisticsCalcStartPage:])  

        except:
            backI2SMean=0;
            backI2Sstd=0;
            maxDivbackI2S,maxNbackI2S,minNbackI2S= [0,0,0]
        
            
        frontTitle="FRONT-correction sum(p"+str(PanelNumber)+")="+"{:.2f}".format(frontC)+' Mean(p'+str(PanelNumber)+')='+"{:.2f}".format(frontCMean)+' Std(p'+str(PanelNumber)+')='+"{:.2f}".format(frontCstd);
        # frontSubTitle="--> I2S: mean= "+"{:.2f}".format(frontI2SMean)+'um  STD= '+"{:.2f}".format(frontI2Sstd)+'um'
        frontSubTitle="--> I2S:Max-Min= +-"+"{:.2f}".format(maxDivfrontI2S/2)+'um'

        backTitle="BACK-correction sum(p"+str(PanelNumber)+")="+"{:.2f}".format(backC)+' Mean(p'+str(PanelNumber)+')='+"{:.2f}".format(backCMean)+' Std(p'+str(PanelNumber)+')='+"{:.2f}".format(backCstd);
        # backSubTitle= "--> I2S: mean= "+"{:.2f}".format(backI2SMean)+'um  STD= '+"{:.2f}".format(backI2Sstd)+'um'
        backSubTitle= "--> I2S: Max-Min= +-"+"{:.2f}".format(maxDivbackI2S/2)+'um'
        
        
        fig = make_subplots(rows=2, cols=1,subplot_titles=(frontTitle+frontSubTitle, backTitle+backSubTitle), vertical_spacing=0.1, shared_xaxes=True,print_grid=True)
        
        
        
        fig.add_trace(
        go.Scatter(x=list(ImagePlacement_ppFRONT['Flat Id']),y=list(ImagePlacement_ppFRONT[f]),
                    name=f),row=1, col=1)
        index = ImagePlacement_ppFRONT.index;
        
        ymin=np.min(list(ImagePlacement_ppFRONT[f]))
        for i in range(len(dbtmpSentCrrFRONT['X Correction Moving Avarage'])):
            try:
                
                cc=list(CyanI2SPnlnumFRONT.columns);
                pctext='p'+cc[0]+'='+"{:.2f}".format(CyanI2SPnlnumFRONT[cc[0]][i]);
                for c in cc[1:]:
                    pctext=pctext+'<br>'+'p'+c+'='+"{:.2f}".format(CyanI2SPnlnumFRONT[c][i]);
                    
                    
                # fig.add_trace(go.Scatter(x=[dbtmpSentCrrFRONT['Flat Id'][i]], y=[ImagePlacement_ppFRONT[f][index[ImagePlacement_ppFRONT['Flat Id']==dbtmpSentCrrFRONT['Flat Id'][i]]].reset_index(drop=True)[0]],
                #             marker=dict(color="crimson", size=6),
                #             mode="markers",
                #             text=pctext,
                #             # font_size=18,
                #             hoverinfo='text'))
                fig.add_trace(go.Scatter(x=[dbtmpSentCrrFRONT['Flat Id'][i]], y=[ymin],
                            marker=dict(color="crimson", size=6),
                            mode="markers",
                            text='Flat Id='+str(dbtmpSentCrrFRONT['Flat Id'][i])+':'+'<br>'+pctext,
                            # font_size=18,
                            hoverinfo='text'))
                fig.add_vline(x=dbtmpSentCrrFRONT['Flat Id'][i], line_width=0.5, line_dash="dash", line_color="red")
            except:
                continue;
                
     
    #####BACK####
        try:
            fig.add_trace(
            go.Scatter(x=list(ImagePlacement_ppBACK['Flat Id']),y=list(ImagePlacement_ppBACK[f]),
                        name=f+' BACK'),row=2, col=1)
            index = ImagePlacement_ppBACK.index;
            ymin=np.min(list(ImagePlacement_ppBACK[f]))
            
            for i in range(len(dbtmpSentCrrBACK['X Correction Moving Avarage'])):
                try:
                    cc=list(CyanI2SPnlnumBACK.columns);
                    pctext='p'+cc[0]+'='+"{:.2f}".format(CyanI2SPnlnumBACK[cc[0]][i]);
                    for c in cc[1:]:
                        pctext=pctext+'<br>'+'p'+c+'='+"{:.2f}".format(CyanI2SPnlnumBACK[c][i]);
                        
                    # fig.add_trace(xref ='x2',
                    #                    yref ='y2',x=[dbtmpSentCrrBACK['Flat Id'][i]], y=[ImagePlacement_ppBACK[f][index[ImagePlacement_ppBACK['Flat Id']==dbtmpSentCrrBACK['Flat Id'][i]]].reset_index(drop=True)[0]],
                    #             marker=dict(color="crimson", size=6),
                    #             mode="markers",
                    #             text=pctext,
                    #             # font_size=18,
                    #             hoverinfo='text')
                    
                    fig.add_trace(xref ='x2',
                                       yref ='y2',x=[dbtmpSentCrrBACK['Flat Id'][i]], y=[ymin],
                                #opacity=0.5,
                                marker=dict(color="crimson", size=6),
                                mode="markers",
                                text='Flat Id='+str(dbtmpSentCrrBACK['Flat Id'][i])+':'+'<br>'+ pctext,
                                # font_size=18,
                                hoverinfo='text')
                    fig.add_vline(x=dbtmpSentCrrBACK['Flat Id'][i], line_width=0.5, line_dash="dash", line_color="red")
                except:
                    continue;
        except:
            1
        
        #fig.update_layout(title=f+' Panel Number ='+str(PanelNumber))
        
        fig.update_layout(title=f)
    
        fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
        
        fig.update_layout(showlegend=False)
        

        
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        plot(fig,filename=f+" AQM_"+ dt_string +".html") 
        fig.show() 
        
        
        
    
        # plot(fig)  
         
        #plot(fig_back,filename="AQM-Back.html")  
    except:
        1      

################### PLOT SIGNALS ############################################

if Plot_RegForAllColors_Left:   
    fig = go.Figure()
    #fig_back = go.Figure()
    fig = make_subplots(rows=3, cols=1,subplot_titles=("Set1", "Set2","Set3"), vertical_spacing=0.1, shared_xaxes=True)
    
    # rnge=[3,6,7]
    
    # db=ImagePlacement_Rightpp
    # db=ImagePlacement_pp
    db1=St1dataAllColorsFrontLeft
    db2=St2dataAllColorsFrontLeft
    db3=St3dataAllColorsFrontLeft
    
    
    inx=list(St1dataAllColorsFrontLeft.index)
    
    
    
    for i in inx:
    # for i in rnge:
        fig.add_trace(go.Scatter(y=list(db1.loc[i]),line_color=i,
                    name=i+' set1'),row=1, col=1)
        fig.add_trace(go.Scatter(y=list(db2.loc[i]),line_color=i,
                    name=i+' set2'), row=2, col=1)
        fig.add_trace(go.Scatter(y=list(db3.loc[i]),line_color=i,
                    name=i+' set3'), row=3, col=1)    
        
    
    
    
    
    
    for i in range(len(dbtmpSentCrrFRONT['X Correction Moving Avarage'])):
        try:
            ymin=np.min(list(db1.loc[inx[0]]))
            for clr in inx:
                clrFRONT=PanelCorrectionFRONT[(PanelCorrectionFRONT['Color']) == clr].reset_index(drop=True)
            
                clrPnlnumFRONT=pd.DataFrame();
                tmpdb=pd.DataFrame();
                for j in range(1,12):
                   tmpdb=clrFRONT[clrFRONT['Panel']==j].reset_index(drop=True);
                   clrPnlnumFRONT=pd.concat((clrPnlnumFRONT, tmpdb['C2C_X Correction'].rename(str(j))), axis=1)
                
                cc=list(clrPnlnumFRONT.columns);
                pctext=clr+':'+'<br>'+'p'+cc[0]+'='+"{:.2f}".format(clrPnlnumFRONT[cc[0]][i]);
                for c in cc[1:]:
                    pctext=pctext+'<br>'+'p'+c+'='+"{:.2f}".format(clrPnlnumFRONT[c][i]);
                    
                    
                fig.add_trace(go.Scatter(x=[dbtmpSentCrrFRONT['Flat Id'][i]-ImagePlacement_ppFRONT['Flat Id'][0]], y=[ymin],
                            #opacity=0.5,
                            marker=dict(color=clr, size=6),
                            mode="markers",
                            text='Flat Id='+str(dbtmpSentCrrFRONT['Flat Id'][i])+' '+pctext,
                            # font_size=18,
                            hoverinfo='text'))
                fig.data[len(fig.data)-1].showlegend = False
                ymin=ymin-50;
            
            
            
            fig.add_vline(x=dbtmpSentCrrFRONT['Flat Id'][i]-ImagePlacement_ppFRONT['Flat Id'][0], line_width=0.5, line_dash="dash", line_color="red")
        except:
            continue;
    
    
    
                
    
    
    
    fig.update_layout(title=f+' FRONT-LEFT')
    fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
    #fig.update_layout(showlegend=False)
         
    
    fig.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    
    # datetime object containing current date and time
    now = datetime.now()
    # # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    plot(fig,filename=f+" FRONT-LEFT_C2C"+ dt_string +".html")  
    fig.show() 
    
    #plot(fig_back,filename="AQM-Back.html")  
    # plot(fig)  
################### PLOT SIGNALS- RIGHT ############################################
if Plot_RegForAllColors_Right:   

    fig = go.Figure()
    #fig_back = go.Figure()
    fig = make_subplots(rows=3, cols=1,subplot_titles=("Set1", "Set2","Set3"), vertical_spacing=0.1, shared_xaxes=True)
    
    # rnge=[3,6,7]
    
    # db=ImagePlacement_Rightpp
    # db=ImagePlacement_pp
    db1=St1dataAllColorsFrontRight
    db2=St2dataAllColorsFrontRight
    db3=St3dataAllColorsFrontRight
    
    
    inx=list(St1dataAllColorsFrontRight.index)
    
    
    
    for i in inx:
    # for i in rnge:
        fig.add_trace(go.Scatter(y=list(db1.loc[i]),line_color=i,
                    name=i+' set1'),row=1, col=1)
        fig.add_trace(go.Scatter(y=list(db2.loc[i]),line_color=i,
                    name=i+' set2'), row=2, col=1)
        fig.add_trace(go.Scatter(y=list(db3.loc[i]),line_color=i,
                    name=i+' set3'), row=3, col=1)    
        
          
    
    
    
    fig.update_layout(title=f+' FRONT-RIGHT')
    fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
    #fig.update_layout(showlegend=False)
         
    
    fig.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    
    # datetime object containing current date and time
    now = datetime.now()
    # # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    plot(fig,filename=f+" FRONT-RIGHT_C2C"+ dt_string +".html")  
    fig.show() 



##################################BACK########################################################
################### PLOT SIGNALS ############################################

    try:    
        fig = go.Figure()
        #fig_back = go.Figure()
        fig = make_subplots(rows=3, cols=1,subplot_titles=("Set1", "Set2","Set3"), vertical_spacing=0.1, shared_xaxes=True)
        
        # rnge=[3,6,7]
        
        # db=ImagePlacement_Rightpp
        # db=ImagePlacement_pp
        db1=St1dataAllColorsBackLeft
        db2=St2dataAllColorsBackLeft
        db3=St3dataAllColorsBackLeft
        
        
        inx=list(St1dataAllColorsBackLeft.index)
        
        
        
        for i in inx:
        # for i in rnge:
            fig.add_trace(go.Scatter(y=list(db1.loc[i]),line_color=i,
                        name=i+' set1'),row=1, col=1)
            fig.add_trace(go.Scatter(y=list(db2.loc[i]),line_color=i,
                        name=i+' set2'), row=2, col=1)
            fig.add_trace(go.Scatter(y=list(db3.loc[i]),line_color=i,
                        name=i+' set3'), row=3, col=1)    
            
        for i in range(len(dbtmpSentCrrBACK['X Correction Moving Avarage'])):
            try:
                ymin=np.min(list(db1.loc[inx[0]]))
                for clr in inx:
                    clrBACK=PanelCorrectionBACK[(PanelCorrectionBACK['Color']) == clr].reset_index(drop=True)
                
                    clrPnlnumBACK=pd.DataFrame();
                    tmpdb=pd.DataFrame();
                    for j in range(1,12):
                       tmpdb=clrBACK[clrBACK['Panel']==j].reset_index(drop=True);
                       clrPnlnumBACK=pd.concat((clrPnlnumBACK, tmpdb['C2C_X Correction'].rename(str(j))), axis=1)
                    
                    cc=list(clrPnlnumBACK.columns);
                    pctext=clr+':'+'<br>'+'p'+cc[0]+'='+"{:.2f}".format(clrPnlnumBACK[cc[0]][i]);
                    for c in cc[1:]:
                        pctext=pctext+'<br>'+'p'+c+'='+"{:.2f}".format(clrPnlnumBACK[c][i]);
                        
                        
                    fig.add_trace(go.Scatter(x=[dbtmpSentCrrBACK['Flat Id'][i]-ImagePlacement_ppBACK['Flat Id'][0]], y=[ymin],
                                marker=dict(color=clr, size=6),
                                mode="markers",
                                text='Flat Id='+str(dbtmpSentCrrBACK['Flat Id'][i])+' '+pctext,
                                # font_size=18,
                                hoverinfo='text'))
                    fig.data[len(fig.data)-1].showlegend = False

                    ymin=ymin-50;   
            except:
                continue;                 
        
            fig.add_vline(x=dbtmpSentCrrBACK['Flat Id'][i]-ImagePlacement_ppBACK['Flat Id'][0], line_width=0.5, line_dash="dash", line_color="red")
    
        
        
        fig.update_layout(title=f+' BACK-LEFT')
        fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
       # fig.update_layout(showlegend=False)
        
        
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        
        # datetime object containing current date and time
        now = datetime.now()
        # # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        plot(fig,filename=f+" BACK-LEFT_C2C"+ dt_string +".html")  
        fig.show() 
    
        #plot(fig_back,filename="AQM-Back.html")  
        # plot(fig)  
        ################### PLOT SIGNALS- RIGHT ############################################
        fig = go.Figure()
        #fig_back = go.Figure()
        fig = make_subplots(rows=3, cols=1,subplot_titles=("Set1", "Set2","Set3"), vertical_spacing=0.1, shared_xaxes=True)
        
        # rnge=[3,6,7]
        
        # db=ImagePlacement_Rightpp
        # db=ImagePlacement_pp
        db1=St1dataAllColorsBackRight
        db2=St2dataAllColorsBackRight
        db3=St3dataAllColorsBackRight
        
        
        inx=list(St1dataAllColorsBackRight.index)
        
        
        
        for i in inx:
        # for i in rnge:
            fig.add_trace(go.Scatter(y=list(db1.loc[i]),line_color=i,
                        name=i+' set1'),row=1, col=1)
            fig.add_trace(go.Scatter(y=list(db2.loc[i]),line_color=i,
                        name=i+' set2'), row=2, col=1)
            fig.add_trace(go.Scatter(y=list(db3.loc[i]),line_color=i,
                        name=i+' set3'), row=3, col=1)    
            
                    
     
        
        
        fig.update_layout(title=f+' BACK-RIGHT')
        fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
        #fig.update_layout(showlegend=False)
          
        
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        
        # datetime object containing current date and time
        now = datetime.now()
        # # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        plot(fig,filename=f+" BACK-RIGHT_C2C"+ dt_string +".html")  
        fig.show() 
      
    except:
        1

################### PLOT SIGNALS ############################################
if Plot_Scale:
    
    fig = go.Figure()
    #fig_back = go.Figure()
    fig = make_subplots(rows=2, cols=1,subplot_titles=("Left", "Right"), vertical_spacing=0.1, shared_xaxes=True)
    
    # rnge=[3,6,7]
    
    # db=ImagePlacement_Rightpp
    # db=ImagePlacement_pp
    db1=ScaleFRONTLeft
    db2=ScaleFRONTRight
    
    
    inx=list(ScaleFRONTLeft.index)
    
    
    
    for i in inx:
    # for i in rnge:
        fig.add_trace(go.Scatter(y=list(db1.loc[i]),line_color=i,
                    name=i+' Left'),row=1, col=1)
        fig.add_trace(go.Scatter(y=list(db2.loc[i]),line_color=i,
                    name=i+' Right'), row=2, col=1)
    
    for i in range(len(dbtmpSentCrrFRONT['X Correction Moving Avarage'])):
        try:
            ymin=np.min(list(db1.loc[inx[0]]))
            for clr in inx:
                clrFRONT=PanelCorrectionFRONT[(PanelCorrectionFRONT['Color']) == clr].reset_index(drop=True)
            
                clrPnlnumFRONT=pd.DataFrame();
                tmpdb=pd.DataFrame();
                for j in range(1,12):
                   tmpdb=clrFRONT[clrFRONT['Panel']==j].reset_index(drop=True);
                   clrPnlnumFRONT=pd.concat((clrPnlnumFRONT, tmpdb['Scaling Correction'].rename(str(j))), axis=1)
                
                cc=list(clrPnlnumFRONT.columns);
                pctext=clr+':'+'<br>'+'p'+cc[0]+'='+"{:.1f}".format((clrPnlnumFRONT[cc[0]][i]-1)*PanelLengthInMM*1000);
                for c in cc[1:]:
                    pctext=pctext+'<br>'+'p'+c+'='+"{:.1f}".format((clrPnlnumFRONT[c][i]-1)*PanelLengthInMM*1000);
                    
                    
                fig.add_trace(go.Scatter(x=[dbtmpSentCrrFRONT['Flat Id'][i]-ImagePlacement_ppFRONT['Flat Id'][0]], y=[ymin],
                            #opacity=0.5,
                            marker=dict(color=clr, size=6),
                            mode="markers",
                            text='Flat Id='+str(dbtmpSentCrrFRONT['Flat Id'][i])+' '+pctext,
                            # font_size=18,
                            hoverinfo='text'))
                fig.data[len(fig.data)-1].showlegend = False

                ymin=ymin-50;
            
            
            
        except:
            continue;
        fig.add_vline(x=dbtmpSentCrrFRONT['Flat Id'][i]-ImagePlacement_ppFRONT['Flat Id'][0], line_width=0.5, line_dash="dash", line_color="red")
    
    fig.update_layout(title=f+' FRONT-SCALE')
    fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
    #fig.update_layout(showlegend=False)
         
    
    fig.update_layout(
        hoverlabel=dict(
            namelength=-1
        )
    )
    
    # datetime object containing current date and time
    now = datetime.now()
    # # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    plot(fig,filename=f+" FRONT-SCALE"+ dt_string +".html")  
    fig.show() 
    
    
    try:
            
            fig01 = go.Figure()
            #fig_back = go.Figure()
            fig01 = make_subplots(rows=2, cols=1,subplot_titles=("Left", "Right"), vertical_spacing=0.1, shared_xaxes=True)
            
            # rnge=[3,6,7]
            
            # db=ImagePlacement_Rightpp
            # db=ImagePlacement_pp
            db1=ScaleBACKLeft
            db2=ScaleBACKRight
            
            
            inx=list(ScaleBACKLeft.index)
            
            
            
            for i in inx:
            # for i in rnge:
                fig01.add_trace(go.Scatter(y=list(db1.loc[i]),line_color=i,
                            name=i+' Left'),row=1, col=1)
                fig01.add_trace(go.Scatter(y=list(db2.loc[i]),line_color=i,
                            name=i+' Right'), row=2, col=1)
            for i in range(len(dbtmpSentCrrBACK['X Correction Moving Avarage'])):
                try:
                    ymin=np.min(list(db1.loc[inx[0]]))
                    for clr in inx:
                        clrBACK=PanelCorrectionBACK[(PanelCorrectionBACK['Color']) == clr].reset_index(drop=True)
                    
                        clrPnlnumBACK=pd.DataFrame();
                        tmpdb=pd.DataFrame();
                        for j in range(1,12):
                           tmpdb=clrBACK[clrBACK['Panel']==j].reset_index(drop=True);
                           clrPnlnumBACK=pd.concat((clrPnlnumBACK, tmpdb['Scaling Correction'].rename(str(j))), axis=1)
                        
                        cc=list(clrPnlnumBACK.columns);
                        pctext=clr+':'+'<br>'+'p'+cc[0]+'='+"{:.1f}".format((clrPnlnumBACK[cc[0]][i]-1)*PanelLengthInMM*1000);
                        for c in cc[1:]:
                            pctext=pctext+'<br>'+'p'+c+'='+"{:.1f}".format((clrPnlnumBACK[c][i]-1)*PanelLengthInMM*1000);
                            
                            
                        fig01.add_trace(go.Scatter(x=[dbtmpSentCrrBACK['Flat Id'][i]-ImagePlacement_ppBACK['Flat Id'][0]], y=[ymin],
                                    marker=dict(color=clr, size=6),
                                    mode="markers",
                                    text='Flat Id='+str(dbtmpSentCrrBACK['Flat Id'][i])+' '+pctext,
                                    # font_size=18,
                                    hoverinfo='text'))
                        fig.data[len(fig.data)-1].showlegend = False

                        ymin=ymin-50;   
                        
    
                except:
                    continue;
            
                fig01.add_vline(x=PanelCorrectionBACK['Flat Id'][i]-ImagePlacement_ppBACK['Flat Id'][0], line_width=0.5, line_dash="dash", line_color="red")
    
            fig01.update_layout(title=f+' BACK-SCALE')
            fig01.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
            #fig.update_layout(showlegend=False)
                 
            
            fig01.update_layout(
                hoverlabel=dict(
                    namelength=-1
                )
            )
            
            # datetime object containing current date and time
            now = datetime.now()
            # # dd/mm/YY H:M:S
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            plot(fig01,filename=f+" BACK-SCALE"+ dt_string +".html")  
            fig01.show() 
    except:
        1

####################################################Min Max Scale ################################

if Plot_MinMaxScale:

    try:
        fig1 = go.Figure()
        
        LeftTitle="FRONT Left Max-Min      Average =  "+"{:.3f}".format(np.mean(ScaleLeftFRONTMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0))+" C2C(2*std+avg) = "+"{:.3f}".format(2*np.std(ScaleLeftFRONTMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0)+np.mean(ScaleLeftFRONTMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0));
        RightTitle="FRONT Right Max-Min    Average =  "+"{:.3f}".format(np.mean(ScaleRightFRONTMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0))+" C2C(2*std+avg) = "+"{:.3f}".format(2*np.std(ScaleRightFRONTMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0)+np.mean(ScaleRightFRONTMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0));
        
        fig1 = make_subplots(rows=2, cols=1,subplot_titles=(LeftTitle, RightTitle), vertical_spacing=0.1, shared_xaxes=True,print_grid=True)
        
        intList = [int(i) for i in ScaleRightFRONTMaxMin.columns]
        
        fig1.add_trace(
        go.Scatter(x=intList,y=list(ScaleLeftFRONTMaxMin.iloc[0,:]),
                    name=f+' Left'),row=1, col=1)
        
        
       
        fig1.add_trace(
            go.Scatter(x=intList,y=list(ScaleRightFRONTMaxMin.iloc[0,:]),
                        name=f+' Right'),row=2, col=1)
      
        
        fig1.update_layout(title=f+' FRONT Scale Error (estimation) in C2C ')
        
        
        fig1.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
        
        
    
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        plot(fig1,filename=f+" FRONT Scale Error (estimation) in C2C "+ dt_string +".html") 
        fig1.show() 
    
    except:
        1    
    
    try:
        fig2 = go.Figure()
        
        LeftTitle="Back Left Max-Min      Average =  "+"{:.3f}".format(np.mean(ScaleLeftBACKMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0))+" C2C(2*std+avg) = "+"{:.3f}".format(2*np.std(ScaleLeftBACKMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0)+np.mean(ScaleLeftBACKMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0));
        RightTitle="Back Right Max-Min    Average =  "+"{:.3f}".format(np.mean(ScaleRightBACKMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0))+" C2C(2*std+avg) = "+"{:.3f}".format(2*np.std(ScaleRightBACKMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0)+np.mean(ScaleRightBACKMaxMin.iloc[0][StatisticsCalcStartPage:],axis=0));
       
        fig2 = make_subplots(rows=2, cols=1,subplot_titles=(LeftTitle, RightTitle), vertical_spacing=0.1, shared_xaxes=True,print_grid=True)
        
        intList = [int(i) for i in ScaleRightBACKMaxMin.columns]
        
        fig2.add_trace(
        go.Scatter(x=intList,y=list(ScaleLeftBACKMaxMin.iloc[0,:]),
                    name=f+' Left'),row=1, col=1)
        
        
       
        fig2.add_trace(
            go.Scatter(x=intList,y=list(ScaleRightBACKMaxMin.iloc[0,:]),
                        name=f+' Right'),row=2, col=1)
      
        
        fig2.update_layout(title=f+' BACK Scale Error (estimation) in C2C ')
        
        
        fig2.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
        
        
    
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        plot(fig2,filename=f+" BACK Scale Error (estimation) in C2C"+ dt_string +".html") 
        fig2.show() 
    
    except:
        1    

########################################Sets 1,2 3 Min - Max Front #####################################

if Plot_MinMaxSETS:
    try:
        fig4 = go.Figure()
        
        LeftTitle="C2C Front       Average =  "+"{:.3f}".format(np.mean(C2C_MaxFront.iloc[0][StatisticsCalcStartPage:],axis=0))+" C2C(2*std+avg) = "+"{:.3f}".format(2*np.std(C2C_MaxFront.iloc[0][StatisticsCalcStartPage:],axis=0)+np.mean(C2C_MaxFront.iloc[0][StatisticsCalcStartPage:],axis=0));
        RightTitle="C2C Back";
        
        try:
            RightTitle="C2C Back     Average =  "+"{:.3f}".format(np.mean(C2C_MaxBack.iloc[0][StatisticsCalcStartPage:],axis=0))+" C2C(2*std+avg) = "+"{:.3f}".format(2*np.std(C2C_MaxBack.iloc[0][StatisticsCalcStartPage:],axis=0)+np.mean(C2C_MaxBack.iloc[0][StatisticsCalcStartPage:],axis=0));
        except:
            1
        
        fig4 = make_subplots(rows=2, cols=1,subplot_titles=(LeftTitle, RightTitle), vertical_spacing=0.1, shared_xaxes=True,print_grid=True)
        
        intList = [int(i) for i in C2C_set1MaxMinFrontRight.columns]
        
        ###right FRONT####
        fig4.add_trace(
        go.Scatter(x=intList,y=list(C2C_set1MaxMinFrontRight.iloc[0,:]),
                    name=f+' set1 MaxMin Front Right'),row=1, col=1)
        
        fig4.add_trace(
        go.Scatter(x=intList,y=list(C2C_set2MaxMinFrontRight.iloc[0,:]),
                    name=f+' set2 MaxMin Front Right'),row=1, col=1)
        
        fig4.add_trace(
        go.Scatter(x=intList,y=list(C2C_set3MaxMinFrontRight.iloc[0,:]),
                    name=f+' set3 MaxMin Front Right'),row=1, col=1)
        ###left FRONT####
        fig4.add_trace(
        go.Scatter(x=intList,y=list(C2C_set1MaxMinFrontLeft.iloc[0,:]),
                    name=f+' set1 MaxMin Front Left'),row=1, col=1)
        
        fig4.add_trace(
        go.Scatter(x=intList,y=list(C2C_set2MaxMinFrontLeft.iloc[0,:]),
                    name=f+' set2 MaxMin Front Left'),row=1, col=1)
        
        fig4.add_trace(
        go.Scatter(x=intList,y=list(C2C_set3MaxMinFrontLeft.iloc[0,:]),
                    name=f+' set3 MaxMin Front Left'),row=1, col=1)
       
        fig4.add_trace(
        go.Scatter(x=intList,y=list(C2C_MaxFront.iloc[0,:]),
                    name=f+' Max Front'),row=1, col=1)
        
        try:
            ###right BACK####
    
            fig4.add_trace(
            go.Scatter(x=intList,y=list(C2C_set1MaxMinBackRight.iloc[0,:]),
                        name=f+' set1 MaxMin Back Right'),row=2, col=1)
            
            fig4.add_trace(
            go.Scatter(x=intList,y=list(C2C_set2MaxMinBackRight.iloc[0,:]),
                        name=f+' set2 MaxMin Back Right'),row=2, col=1)
            
            fig4.add_trace(
            go.Scatter(x=intList,y=list(C2C_set3MaxMinBackRight.iloc[0,:]),
                        name=f+' set3 MaxMin Back Right'),row=2, col=1)
            
            ###left BACK####
            
            fig4.add_trace(
            go.Scatter(x=intList,y=list(C2C_set1MaxMinBackLeft.iloc[0,:]),
                        name=f+' set1 MaxMin Back Left'),row=2, col=1)
            
            fig4.add_trace(
            go.Scatter(x=intList,y=list(C2C_set2MaxMinBackLeft.iloc[0,:]),
                        name=f+' set2 MaxMin Back Left'),row=2, col=1)
            
            fig4.add_trace(
            go.Scatter(x=intList,y=list(C2C_set3MaxMinBackLeft.iloc[0,:]),
                        name=f+' set3 MaxMin Back Left'),row=2, col=1) 
            
            fig4.add_trace(
            go.Scatter(x=intList,y=list(C2C_MaxBack.iloc[0,:]),
                    name=f+' Max Back'),row=2, col=1)
        
        except:
                1
        
        
        fig4.update_layout(title=f+' Back Front C2C Max-Min')
        
        
        fig4.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
        
        
    
    
    
    except:
        1    
    
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    plot(fig4,filename=f+" C2C Max-Min"+ dt_string +".html") 
    fig4.show() 

##### TILL HERE!!!!

end = time.time()
print(end - start)

################## Histogram #####################
if Plot_MinMaxSETS:
    C2CmaxFront=C2C_MaxFront.iloc[0][StatisticsCalcStartPage:];
    bins=int(len(list(C2CmaxFront))/10);
    fig5 = px.histogram(C2CmaxFront, nbins=bins)
    fig5.update_layout(title=f+' Front C2C Max-Min [um]')
    plot(fig5,filename=f+" Front C2C Max-Min Histogram"+ dt_string +".html")
    fig5.show()
    try:
        xBack=C2C_MaxBack.iloc[0][StatisticsCalcStartPage:];
        
        fig6 = px.histogram(xBack, nbins=bins)
        fig6.update_layout(title=f+' Back C2C Max-Min [um]')
        plot(fig6,filename=f+" Back C2C Max-Min Histogram"+ dt_string +".html")
        fig6.show()
    except:
        1;



