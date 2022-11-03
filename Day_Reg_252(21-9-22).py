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
#DistBetweenSets =  126357; 
GlobalScale = 0.9945 # Drop3 simplex = 0.9976, Duplex = 0.9984 ,,,, Drop5 Simplex = 0.9953, Duplex = 0.9945 
PanelLengthInMM = 650;
JobLength = 0;
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
        folder=self.UnzipFilesAndSaveToFolderList();
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
        pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
        os.chdir(pthFf);
        fname='ImagePlacement_Left.csv';
        if os.path.exists(fname) and Path(fname).stat().st_size:
            dbtmp=pd.read_csv(fname,usecols=[0]);
            if len(dbtmp['Flat Id'])>self.JobLength:
                vlid= True;
        return vlid;
    
    def ExtractBasline(self,f):
        magePlacement_res=pd.read_csv(self.pthF+f+'/'+self.side+'/'+'AnalysisResults/ImagePlacementAnalysis_'+self.side+'.csv',usecols=[5])
        BaseLine=magePlacement_res['Expected X'][0];
        return BaseLine;
    
    def ReadImagePlacmentData(self):
        os.chdir(self.pthF);
        dbtmp=pd.DataFrame()
        ImagePlacement_pp=pd.DataFrame()
        fname='ImagePlacement_'+self.pageSide+'.csv';
        for f in self.fldrs:
            pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults/';
            # print(pthFf)
            try:
               os.chdir(pthFf);
            except:
               continue;
            
            if self.CheckIfFileValid(f):
                try:
                    BaseLine=self.ExtractBasline(f);
                    dbtmp=pd.read_csv(fname,usecols=[10]);
                    ImagePlacement_pp = pd.concat((ImagePlacement_pp, (dbtmp['T1->X'].rename(f)-BaseLine)), axis=1)
                except:
                    continue;
            
        return  ImagePlacement_pp;    
                

class CalcWave():
    def  __init__(self, pthF,fldrs,side,JobLength,clrNumber,setNumber): 
        self.pthF = pthF;
        self.fldrs = fldrs;
        self.side = side;
        self.JobLength = JobLength;
        self.clrNumber=clrNumber;
        self.setNumber=setNumber;


    
    def LoadRawData(self,fname,f):
        RawData=pd.read_csv(self.pthF+f+'/'+self.side+'/'+'RawResults'+'/'+fname);
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
        
        
    def CalcC2CSingleSidePerColor(self,fname,f):
        RawDataSuccess,flatNumberFailed,l1= self.LoadRawData(fname,f);
        
        RawDataSuccess=self.ConvertRowsToInt(RawDataSuccess);
        
        colorDic,DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3=self.CalcMeanByColor(RawDataSuccess);
        
        indexNumberFailed=[]
        C2Creg=[]
        
        for j,l in enumerate(l1):
            if not(l in flatNumberFailed):
                FlatIDdata=RawDataSuccess[RawDataSuccess['Flat Id']==l].reset_index();
                Stdata=[];
                if setNumber==1:
                    [Stdata.append(x-DataAllMeanColorSET1['Set #1 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #1 X'])];
                if setNumber==2:  
                    [Stdata.append(x-DataAllMeanColorSET2['Set #2 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #2 X'])];
                if setNumber==3:  
                    [Stdata.append(x-DataAllMeanColorSET3['Set #3 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #3 X'])];
                C2Creg.append(Stdata[self.clrNumber]);

            else:
                indexNumberFailed.append(j)
                if j>0:
                    C2Creg.append(C2Creg[j-1])
                else:
                    C2Creg.append(0.0)
    
        return colorDic,C2Creg,indexNumberFailed;

    def CalcC2CSingleSidePerColorAVR(self,fname,f):
        RawDataSuccess,flatNumberFailed,l1= self.LoadRawData(fname,f);
        
        RawDataSuccess=self.ConvertRowsToInt(RawDataSuccess);
        
        Lname=fname.split('.');
        LLname=Lname[0].split('_');
        
        MeregedDataAllMeanColor = pd.read_csv(self.pthF+'/'+'MeregedDataAllMeanColor_'+self.side+'_'+LLname[1]+'.csv')
        
        colorDic={}
        for i,cl in enumerate(MeregedDataAllMeanColor['Ink\Sets']):
            colorDic[i]=cl

        DataAllMeanColorSET1=MeregedDataAllMeanColor['Set #1 X'].rename(index=colorDic);
        DataAllMeanColorSET2=MeregedDataAllMeanColor['Set #2 X'].rename(index=colorDic);
        DataAllMeanColorSET3=MeregedDataAllMeanColor['Set #3 X'].rename(index=colorDic);
        
        indexNumberFailed=[]
        C2Creg=[]
        
        for j,l in enumerate(l1):
            if not(l in flatNumberFailed):
                FlatIDdata=RawDataSuccess[RawDataSuccess['Flat Id']==l].reset_index();
                Stdata=[];
                if setNumber==1:
                    [Stdata.append(x-DataAllMeanColorSET1['Set #1 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #1 X'])];
                if setNumber==2:  
                    [Stdata.append(x-DataAllMeanColorSET2['Set #2 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #2 X'])];
                if setNumber==3:  
                    [Stdata.append(x-DataAllMeanColorSET3['Set #3 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #3 X'])];
                C2Creg.append(Stdata[self.clrNumber]);

            else:
                indexNumberFailed.append(j)
                if j>0:
                    C2Creg.append(C2Creg[j-1])
                else:
                    C2Creg.append(0.0)
    
        return colorDic,C2Creg,indexNumberFailed;




    def CheckIfFileValid(self,f):
        vlid=False;
        dbtmp=pd.DataFrame();
        pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
        os.chdir(pthFf);
        fname='Registration_Left.csv';
        if os.path.exists(fname) and Path(fname).stat().st_size:
            dbtmp=pd.read_csv(fname,usecols=[0]);
            if len(dbtmp['Job Id'])>self.JobLength:
                vlid= True;
        return vlid;

    def CalcC2CregForLeftRight(self):
        ImagePlacement_pp=pd.DataFrame()
        flatNumberFailed_pp=pd.DataFrame();         
        for f in self.fldrs:
            stP=pd.DataFrame();
            flatNumberFailed=pd.DataFrame();
            try:
                if self.CheckIfFileValid(f):
                    colorDic,C2CregLeft,indexNumberFailedLeft=self.CalcC2CSingleSidePerColorAVR('Registration_Left.csv',f);
                    colorDic,C2CregRight,indexNumberFailedRight=self.CalcC2CSingleSidePerColorAVR('Registration_Right.csv',f);
                    C2CMaxLeftRight=[];
                    for i in range(len(C2CregRight)):
                            tmp=C2CregLeft[i]-C2CregRight[i];
                            C2CMaxLeftRight.append(tmp);
                            # C2CMaxLeftRight.append(tmp[np.argmax([abs(C2CregLeft[i]),abs(C2CregRight[i])])]);
                     
                    stP[f]=C2CMaxLeftRight;
                    flatNumberFailed[f]=list(OrderedDict.fromkeys(indexNumberFailedLeft+indexNumberFailedRight));
                    flatNumberFailed_pp=pd.concat([flatNumberFailed_pp, flatNumberFailed[f]],axis=1);
                    ImagePlacement_pp=pd.concat([ImagePlacement_pp, stP[f]],axis=1);
            except:
                continue;
        
        return colorDic,ImagePlacement_pp,flatNumberFailed_pp      
    
    def CalcC2CregForLeftRightAVR(self):
        ImagePlacement_pp=pd.DataFrame()
        flatNumberFailed_pp=pd.DataFrame();         
        for f in self.fldrs:
            stP=pd.DataFrame();
            flatNumberFailed=pd.DataFrame();
            try:
                if self.CheckIfFileValid(f):
                    colorDic,C2CregLeft,indexNumberFailedLeft=self.CalcC2CSingleSidePerColor('Registration_Left.csv',f);
                    colorDic,C2CregRight,indexNumberFailedRight=self.CalcC2CSingleSidePerColor('Registration_Right.csv',f);
                    C2CMaxLeftRight=[];
                    for i in range(len(C2CregRight)):
                            tmp=C2CregLeft[i]-C2CregRight[i];
                            C2CMaxLeftRight.append(tmp);
                            # C2CMaxLeftRight.append(tmp[np.argmax([abs(C2CregLeft[i]),abs(C2CregRight[i])])]);
                     
                    stP[f]=C2CMaxLeftRight;
                    flatNumberFailed[f]=list(OrderedDict.fromkeys(indexNumberFailedLeft+indexNumberFailedRight));
                    flatNumberFailed_pp=pd.concat([flatNumberFailed_pp, flatNumberFailed[f]],axis=1);
                    ImagePlacement_pp=pd.concat([ImagePlacement_pp, stP[f]],axis=1);
            except:
                continue;
        return colorDic,ImagePlacement_pp,flatNumberFailed_pp                          

class CalcC2C():
    def  __init__(self, pthF,fldrs,side,JobLength): 
        self.pthF = pthF;
        self.fldrs = fldrs;
        self.side = side;
        self.JobLength = JobLength;
    
    def LoadRawData(self,fname,f):
        RawData=pd.read_csv(self.pthF+f+'/'+self.side+'/'+'RawResults'+'/'+fname);
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
        
        return DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3;
    
    def ConvertRowsToInt(self,RawDataSuccess):       
        RawDataSuccess['Set #1 X']  = RawDataSuccess['Set #1 X'].astype('int64');
        RawDataSuccess['Set #2 X']  = RawDataSuccess['Set #2 X'].astype('int64');
        RawDataSuccess['Set #3 X']  = RawDataSuccess['Set #3 X'].astype('int64');
        return RawDataSuccess;
        
    def FlatNumberSerialized(self):
        FaltNumberSer_pp=pd.DataFrame();
        for f in self.fldrs:
            FaltNumberSerTmp=pd.DataFrame();
            pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
            try:
                os.chdir(pthFf);
            except:
                continue;
            fname='Registration_Left.csv';
            if os.path.exists(fname) and Path(fname).stat().st_size:
                FaltNumberSerTmp[f]=pd.read_csv(fname,usecols=[1]).iloc[:,0].unique().tolist();  
                FaltNumberSerTmp[f]=FaltNumberSerTmp[f].astype('int64');
                FaltNumberSer_pp=pd.concat([FaltNumberSer_pp, FaltNumberSerTmp],axis=1);
        return FaltNumberSer_pp;  
    
    def FlatNumberSerializedDebug(self):
        os.chdir(self.pthF);
        print(pthF)
        FaltNumberSer_pp=pd.DataFrame();
        for f in self.fldrs:
            FaltNumberSerTmp=pd.DataFrame();
            pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
            try:
                print(pthFf)
                os.chdir(pthFf);
            except:
                continue;
            fname='Registration_Left.csv';
            if os.path.exists(fname) and Path(fname).stat().st_size:
                FaltNumberSerTmp[f]=pd.read_csv(fname,usecols=[1]).iloc[:,0].unique().tolist();  
                FaltNumberSerTmp[f]=FaltNumberSerTmp[f].astype('int64');
                FaltNumberSer_pp=pd.concat([FaltNumberSer_pp, FaltNumberSerTmp],axis=1);
        return FaltNumberSer_pp;      
        
    def CalcC2CSingleSide(self,fname,f):
        RawDataSuccess,flatNumberFailed,l1= self.LoadRawData(fname,f);
        
        RawDataSuccess=self.ConvertRowsToInt(RawDataSuccess);
        
        DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3=self.CalcMeanByColor(RawDataSuccess);
        
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

    def CheckIfFileValid(self,f):
        vlid=False;
        dbtmp=pd.DataFrame();
        pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
        os.chdir(pthFf);
        fname='Registration_Left.csv';
        if os.path.exists(fname) and Path(fname).stat().st_size:
            dbtmp=pd.read_csv(fname,usecols=[0]);
            if len(dbtmp['Job Id'])>self.JobLength:
                vlid= True;
        return vlid;

    def CalcC2CregForLeftRight(self):
        ImagePlacement_pp=pd.DataFrame()
        flatNumberFailed_pp=pd.DataFrame();         
        for f in self.fldrs:
            stP=pd.DataFrame();
            flatNumberFailed=pd.DataFrame();
            try:
                if self.CheckIfFileValid(f):
                    C2CregLeft,indexNumberFailedLeft=self.CalcC2CSingleSide('Registration_Left.csv',f);
                    C2CregRight,indexNumberFailedRight=self.CalcC2CSingleSide('Registration_Right.csv',f);
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

class CalcC2C_AvrgOfAll():
    def  __init__(self, pthF,fldrs,side,JobLength,PanelLengthInMM,pageSide): 
        self.pthF = pthF;
        self.fldrs = fldrs;
        self.side = side;
        self.JobLength = JobLength;
        self.PanelLengthInMM = PanelLengthInMM;
        self.pageSide = pageSide;
  
    def LoadRawData(self,fname,f):
        RawData=pd.read_csv(self.pthF+f+'/'+self.side+'/'+'RawResults'+'/'+fname);
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

         valueSet2= valueSet1+(125686/GlobalScale);

         valueSet3= valueSet2+(125686/GlobalScale);
         
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
         
         ## Unite All Sets  - Expected color position          
         # RefSETloc=DataAllMeanColorSET1;
         # RefSETloc.insert(1,'Set #2 X' ,list(DataAllMeanColorSET2['Set #2 X']))
         # RefSETloc.insert(2,'Set #3 X' ,list(DataAllMeanColorSET3['Set #3 X']))
         # RefSETloc=RefSETloc.drop(columns=['Ink\Sets'])
         
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

          # slp=[]

            
          # for inx in St1dataAllColors.index:
          #     y=list(RefSETloc.iloc[self.get_key(colorDic,inx),:])
          #     z = np.polyfit(x, y, 1)
          #     p = np.poly1d(z)
          #     slp.append(list(p)[0])
             
          # RefSETloc.insert(3,'Slop' ,slp)
             
         
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
         
            
         
            
         ## 2 point 
         # PanelColorSet={};
         # ListColorDict={};

         # for c in St1dataAllColors.columns:
         #     for inx in St1dataAllColors.index:
         #         ListColorDict[inx]=[St1dataAllColors[c][inx],St3dataAllColors[c][inx]]
         #     PanelColorSet[c]=ListColorDict
         #     ListColorDict={}
         
            
         
            
         
         # ## Calc Scale per Color 
         
         # x=[0,2];

         # # slp=[]

            
         # # for inx in St1dataAllColors.index:
         # #     y=list(RefSETloc.iloc[self.get_key(colorDic,inx),:])
         # #     z = np.polyfit(x, y, 1)
         # #     p = np.poly1d(z)
         # #     slp.append(list(p)[0])
             
         # # RefSETloc.insert(3,'Slop' ,slp)
             
         
         # Scale=St1dataAllColors;


         # for c in St1dataAllColors.columns:
         #     for inx in St1dataAllColors.index:
         #         y= PanelColorSet[c][inx]
         #         z = np.polyfit(x, y, 1)
         #         p = np.poly1d(z)
         #         try:
         #             Scale[c][inx]=(RefSETloc['Slop'][inx]/list(p)[0]-1)*self.PanelLengthInMM*1000
         #         except:
         #             continue;
         
         
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

         valueSet2= valueSet1+(125686/GlobalScale);

         valueSet3= valueSet2+(125686/GlobalScale);
         
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
                
    def FlatNumberSerialized(self):
        FaltNumberSer_pp=pd.DataFrame();
        for f in self.fldrs:
            FaltNumberSerTmp=pd.DataFrame();
            try:
                pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
                os.chdir(pthFf);
                fname='Registration_Left.csv';
                if os.path.exists(fname) and Path(fname).stat().st_size:
                    FaltNumberSerTmp[f]=pd.read_csv(fname,usecols=[1]).iloc[:,0].unique().tolist();  
                    FaltNumberSerTmp[f]=FaltNumberSerTmp[f].astype('int64');
                    FaltNumberSer_pp=pd.concat([FaltNumberSer_pp, FaltNumberSerTmp],axis=1);
            except:
                 continue;
        return FaltNumberSer_pp;  
    
    def FlatNumberSerializedDebug(self):
        os.chdir(self.pthF);
        print(pthF)
        FaltNumberSer_pp=pd.DataFrame();
        for f in self.fldrs:
            FaltNumberSerTmp=pd.DataFrame();
            pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
            print(pthFf)
            os.chdir(pthFf);
            fname='Registration_Left.csv';
            if os.path.exists(fname) and Path(fname).stat().st_size:
                FaltNumberSerTmp[f]=pd.read_csv(fname,usecols=[1]).iloc[:,0].unique().tolist();  
                FaltNumberSerTmp[f]=FaltNumberSerTmp[f].astype('int64');
                FaltNumberSer_pp=pd.concat([FaltNumberSer_pp, FaltNumberSerTmp],axis=1);
        return FaltNumberSer_pp;      
        
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

    def CheckIfFileValid(self,f):
        vlid=False;
        dbtmp=pd.DataFrame();
        pthFf=self.pthF+f+'/'+self.side+'/'+'RawResults';
        os.chdir(pthFf);
        fname='Registration_Left.csv';
        if os.path.exists(fname) and Path(fname).stat().st_size:
            dbtmp=pd.read_csv(fname,usecols=[0]);
            if len(dbtmp['Job Id'])>self.JobLength:
                vlid= True;
        return vlid;

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
        jobNme=c.split(' ')
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


threadList.append(myThread(1,"Thread-C2C",folder))


# threadList.append(myThread(2, "Thread-Scale",folder))

    
# for f in folder:
#     threadList.append(myThread(3, "Thread-I2S",f))    
        
threadList.append(myThread(3, "Thread-I2S",folder))    


for itm in threadList:
    itm.start()

for itm in threadList:
    itm.join()





endCalc = time.time()
print(endCalc - startCalc)


##################################################################################
# pthF='D:/AQMDuplex/';

# folder=['QCS Production_546 Archive 25-03-2022 10-56-52']

# ImagePlacement_pp=pd.DataFrame()
# flatNumberFailed_pp=pd.DataFrame();

# DataAllMeanColorSET1left,DataAllMeanColorSET2left,DataAllMeanColorSET3left,colorDic = CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CalcMeanByColorForAllJobs('Registration_Left.csv')
# DataAllMeanColorSET1right,DataAllMeanColorSET2right,DataAllMeanColorSET3right,colorDic = CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CalcMeanByColorForAllJobs('Registration_Right.csv')

# for f in CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').fldrs:
#     stP=pd.DataFrame();
#     flatNumberFailed=pd.DataFrame();
#     try:
#         if CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CheckIfFileValid(f):
#             C2CregLeft,indexNumberFailedLeft=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CalcC2CSingleSide('Registration_Left.csv',f,DataAllMeanColorSET1left,DataAllMeanColorSET2left,DataAllMeanColorSET3left);
#             C2CregRight,indexNumberFailedRight=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CalcC2CSingleSide('Registration_Right.csv',f,DataAllMeanColorSET1right,DataAllMeanColorSET2right,DataAllMeanColorSET3right);
#             C2CMaxLeftRight=[];
#             for i in range(len(C2CregRight)):
#                     tmp=[C2CregLeft[i],C2CregRight[i]];
#                     C2CMaxLeftRight.append(np.max(tmp));
#                     # C2CMaxLeftRight.append(tmp[np.argmax([abs(C2CregLeft[i]),abs(C2CregRight[i])])]);
             
#             stP[f]=C2CMaxLeftRight;
#             flatNumberFailed[f]=list(OrderedDict.fromkeys(indexNumberFailedLeft+indexNumberFailedRight));
#             flatNumberFailed_pp=pd.concat([flatNumberFailed_pp, flatNumberFailed[f]],axis=1);
#             ImagePlacement_pp=pd.concat([ImagePlacement_pp, stP[f]],axis=1);
#     except:
#         1
#         continue;
        

# ##################################
# DataAllMeanColorSET1=DataAllMeanColorSET1left
# DataAllMeanColorSET2=DataAllMeanColorSET2left
# DataAllMeanColorSET3=DataAllMeanColorSET3left 
# #CalcC2CSingleSide(self,fname,f,DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3):
# RawDataSuccess,flatNumberFailed,l1= CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').LoadRawData(fname,f);

# RawDataSuccess=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').ConvertRowsToInt(RawDataSuccess);

# #DataAllMeanColorSET1,DataAllMeanColorSET2,DataAllMeanColorSET3=self.CalcMeanByColor(RawDataSuccess);

# indexNumberFailed=[]
# C2Creg=[]

# for j,l in enumerate(l1):
#     #print("j="+str(j)+" l="+str(l)+"\n")
#     if not(l in flatNumberFailed):
#         FlatIDdata=RawDataSuccess[RawDataSuccess['Flat Id']==l].reset_index();
#         St1data=[];
#         # for i,x in enumerate(FlatIDdata['Set #1 X']):
#         #     #print("i="+str(i)+" x="+str(x)+"\n")
#         #     St1data.append(x-DataAllMeanColorSET1['Set #1 X'][FlatIDdata['Ink\Sets'][i]]);
#         [St1data.append(x-DataAllMeanColorSET1['Set #1 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #1 X'])];
#         St2data=[];
#         [St2data.append(x-DataAllMeanColorSET2['Set #2 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #2 X'])];
#         St3data=[];
#         [St3data.append(x-DataAllMeanColorSET3['Set #3 X'][FlatIDdata['Ink\Sets'][i]]) for i,x in enumerate(FlatIDdata['Set #3 X'])];
#         tmp=[(np.max(St1data)-np.min(St1data)),(np.max(St2data)-np.min(St2data)),(np.max(St3data)-np.min(St3data))];
#         C2Creg.append(np.max(tmp));
#         # C2Creg.append(tmp[np.argmax([abs((np.max(St1data)-np.min(St1data))),abs((np.max(St2data)-np.min(St2data))),abs((np.max(St2data)-np.min(St2data)))])]);

#     else:
#         indexNumberFailed.append(j)
#         if j>0:
#             C2Creg.append(C2Creg[j-1])
#         else:
#             C2Creg.append(0.0)


# #return C2Creg,indexNumberFailed;





















####################Thread###################################
ScaleMaxMinDF_FRONTFLeft=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Left').CalcScaleForAllJOBS();
ScaleMaxMinDF_FRONTRight=CalcC2C_AvrgOfAll(pthF,folder,'Front',JobLength,PanelLengthInMM,'Right').CalcScaleForAllJOBS();
  
  
try:
  ScaleMaxMinDF_BACKFLeft=CalcC2C_AvrgOfAll(pthF,folder,'Back',JobLength,PanelLengthInMM,'Left').CalcScaleForAllJOBS();
  ScaleMaxMinDF_BACKRight=CalcC2C_AvrgOfAll(pthF,folder,'Back',JobLength,PanelLengthInMM,'Right').CalcScaleForAllJOBS();
except:
  1;



os.chdir(pthF)
# #########################################################################################################################

###FOR DEBUG

################### PLOT SIGNALS ############################################
startFigure = time.time()

    
fig = go.Figure()
#fig_back = go.Figure()
fig = make_subplots(rows=2, cols=1,subplot_titles=("LEFT", "RIGHT"), vertical_spacing=0.1, shared_xaxes=True)

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
db=ImagePlacement_Leftpp

db_Right=ImagePlacement_Rightpp

# col=list(db.columns)

col=list(SortJobsByTime(list(db.columns)).values())
rnge=range(len(col))

for i in rnge:
# for i in rnge:
    fig.add_trace(go.Scatter(y=list(db[col[i]]),
                name=col[i]),row=1, col=1)
    fig.add_trace(go.Scatter(y=list(db_Right[col[i]]),
                name=col[i]), row=2, col=1)
    
            

# fig.update_layout(title='ImagePlacement_Right')
# fig.update_layout(title='ImagePlacement_Left')

fig.update_layout(title='FRONT')
#fig_back.update_layout(title='ImagePlacement_Left-Back')
fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
# fig.update_layout(
#     legend=dict(x= 1.1,y=1.1,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),
#     width=1000,
#     height=600,
#     autosize=False,
#     template="plotly_white",
#     # side ='left'
# )

        

fig.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)

# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig,auto_play=True,filename="I2S_FRONT_AQM_"+ dt_string +".html")  
#plot(fig_back,filename="AQM-Back.html")  
fig.show()
################### PLOT SIGNALS- RIGHT ############################################

    
fig1 = go.Figure()
#fig_back = go.Figure()
fig1 = make_subplots(rows=2, cols=1,subplot_titles=("LEFT", "RIGHT"), vertical_spacing=0.1, shared_xaxes=True)

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp

try:
    db=ImagePlacement_Leftpp_BACK
    db_Right=ImagePlacement_Rightpp_BACK
except:
    1
# col=list(db.columns)
col=list(SortJobsByTime(list(db.columns)).values())

rnge=range(len(col))

for i in rnge:
# for i in rnge:
    try:
        fig1.add_trace(go.Scatter(y=list(db[col[i]]),x=list(range(len(db[col[i]]))),
                name=col[i]),row=1, col=1)
        fig1.add_trace(go.Scatter(y=list(db_Right[col[i]]),x=list(range(len(db[col[i]]))),
                name=col[i]), row=2, col=1)
    except:
        continue;
            

# fig.update_layout(title='ImagePlacement_Right')
# fig.update_layout(title='ImagePlacement_Left')

fig1.update_layout(title='BACK')
#fig_back.update_layout(title='ImagePlacement_Left-Back')
fig1.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)


# fig.update_layout(
#     legend=dict(x= 1.1,y=1.1,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),
#     width=1000,
#     height=600,
#     autosize=False,
#     template="plotly_white",
#     # side ='left'
# )

       
fig1.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)


# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig1,filename="I2S_BACK_AQM_"+ dt_string +".html")  
#plot(fig_back,filename="AQM-Back.html")  
fig1.show() 

######################################################################################################


fig2 = make_subplots(rows=2, cols=1,subplot_titles=("FRONT", "BACK"), vertical_spacing=0.1, shared_xaxes=True)


db=pd.DataFrame()
db=DataPivotFront
FailedFlatFRONT=pd.DataFrame()
FailedFlatFRONT=flatNumberFailedFront;
# col=list(db.columns)
col=list(SortJobsByTime(list(db.columns)).values())

try:
    dbBACK=pd.DataFrame()
    dbBACK=DataPivotBack
    FailedFlatBACK=pd.DataFrame()
    FailedFlatBACK=flatNumberFailedBack;
    # coldbBACK=list(dbBACK.columns)
    coldbBACK=list(SortJobsByTime(list(dbBACK.columns)).values())

except:
    1;    

rnge=range(len(col))

for i in rnge:
# for i in rnge:
    fig2.add_trace(
    go.Scatter(x=list(db.index),
        y=list(db[col[i]]),
                name=col[i]+' FRONT'),row=1, col=1);
    # try:
    #     for arg in FailedFlatFRONT[col[i]]:
    #         fig2.add_trace(go.Scatter(x=[arg],y=[db[col[i]][arg]],
    #                                  marker=dict(color="crimson", size=12),
    #                                  mode="markers",
    #                                  name=col[i]+' FRONT '+str(arg)),
    #                                  row=1, col=1)
    # except:
    #     1
    try:
      fig2.add_trace(
      go.Scatter(x=list(dbBACK.index),
        y=list(dbBACK[coldbBACK[i]]),
                name=coldbBACK[i]+' BACK'),row=2, col=1); 
      # try:
      #     for arg in FailedFlatBACK[coldbBACK[i]]:
      #       fig2.add_trace(go.Scatter(x=[arg],y=[dbBACK[coldbBACK[i]][arg]],
      #                                marker=dict(color="crimson", size=12),
      #                                mode="markers",
      #                                name=coldbBACK[i]+' BACK '+str(arg)),
      #                                row=2, col=1)
      # except:
      #    1
    except:
          continue;


# fig.update_traces(mode="arkers+lines")

fig2.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
fig2.update_layout(title='C2C')

fig2.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)

   
        

# plot(fig2)  
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig2,filename="C2C_AQM_"+ dt_string +".html")  

fig2.show()    
############################################Scale ####################################################

fig3 = go.Figure()
#fig_back = go.Figure()
fig3 = make_subplots(rows=2, cols=1,subplot_titles=("LEFT", "RIGHT"), vertical_spacing=0.1, shared_xaxes=True)

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp
db=ScaleMaxMinDF_FRONTFLeft

db_Right=ScaleMaxMinDF_FRONTRight

# col=list(db.columns)

col=list(SortJobsByTime(list(db.columns)).values())
rnge=range(len(col))

for i in rnge:
# for i in rnge:
    try:
        fig3.add_trace(go.Scatter(y=list(db[col[i]]),
                    name=col[i]),row=1, col=1)
        fig3.add_trace(go.Scatter(y=list(db_Right[col[i]]),
                    name=col[i]), row=2, col=1)
    except:
        1
    
            

# fig.update_layout(title='ImagePlacement_Right')
# fig.update_layout(title='ImagePlacement_Left')

fig3.update_layout(title='Scale-FRONT')
#fig_back.update_layout(title='ImagePlacement_Left-Back')
fig3.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
# fig.update_layout(
#     legend=dict(x= 1.1,y=1.1,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),
#     width=1000,
#     height=600,
#     autosize=False,
#     template="plotly_white",
#     # side ='left'
# )

        

fig3.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)

# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig3,auto_play=True,filename="Scale_FRONT_AQM_"+ dt_string +".html")  
#plot(fig_back,filename="AQM-Back.html")  
fig3.show()
#######################################################################################################


fig4 = go.Figure()
#fig_back = go.Figure()
fig4 = make_subplots(rows=2, cols=1,subplot_titles=("LEFT", "RIGHT"), vertical_spacing=0.1, shared_xaxes=True)

# rnge=[3,6,7]

# db=ImagePlacement_Rightpp
# db=ImagePlacement_pp

try:
    db=ScaleMaxMinDF_BACKFLeft
    db_Right=ScaleMaxMinDF_BACKRight
except:
    1
# col=list(db.columns)
col=list(SortJobsByTime(list(db.columns)).values())

rnge=range(len(col))

for i in rnge:
# for i in rnge:
    try:
        fig4.add_trace(go.Scatter(y=list(db[col[i]]),x=list(range(len(db[col[i]]))),
                name=col[i]),row=1, col=1)
        fig4.add_trace(go.Scatter(y=list(db_Right[col[i]]),x=list(range(len(db[col[i]]))),
                name=col[i]), row=2, col=1)
    except:
        continue;
            

# fig.update_layout(title='ImagePlacement_Right')
# fig.update_layout(title='ImagePlacement_Left')

fig4.update_layout(title='Scale-BACK')
#fig_back.update_layout(title='ImagePlacement_Left-Back')
fig4.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)


# fig.update_layout(
#     legend=dict(x= 1.1,y=1.1,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),
#     width=1000,
#     height=600,
#     autosize=False,
#     template="plotly_white",
#     # side ='left'
# )

       
fig4.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)


# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig4,filename="Scale_BACK_AQM_"+ dt_string +".html")  
#plot(fig_back,filename="AQM-Back.html")  
fig4.show() 


endFigure = time.time()
print(endFigure - startFigure)

#########################################   wave  ####################################################
# import webbrowser
# chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
# webbrowser.get(chrome_path).open('http://docs.python.org/')

#fig3 = make_subplots(rows=2, cols=1,subplot_titles=("FRONT", "BACK"), vertical_spacing=0.1, shared_xaxes=True)
#
#
#db=pd.DataFrame()
#db=waveFront
#FailedFlatFRONT=pd.DataFrame()
#FailedFlatFRONT=flatNumberFailedwaveFront;
#col=list(db.columns)
#try:
#    dbBACK=pd.DataFrame()
#    dbBACK=waveBack
#    FailedFlatBACK=pd.DataFrame()
#    FailedFlatBACK=flatNumberFailedwaveBack;
#    coldbBACK=list(dbBACK.columns)
#except:
#    1;    
#
#rnge=range(len(col))
#
#for i in rnge:
## for i in rnge:
#    fig3.add_trace(
#    go.Scatter(x=list(db.index),
#        y=list(db[col[i]]),
#                name=col[i]+' FRONT'),row=1, col=1);
#    # try:
#    #     for arg in FailedFlatFRONT[col[i]]:
#    #         fig2.add_trace(go.Scatter(x=[arg],y=[db[col[i]][arg]],
#    #                                  marker=dict(color="crimson", size=12),
#    #                                  mode="markers",
#    #                                  name=col[i]+' FRONT '+str(arg)),
#    #                                  row=1, col=1)
#    # except:
#    #     1
#    try:
#      fig3.add_trace(
#      go.Scatter(x=list(dbBACK.index),
#        y=list(dbBACK[coldbBACK[i]]),
#                name=coldbBACK[i]+' BACK'),row=2, col=1); 
#      # try:
#      #     for arg in FailedFlatBACK[coldbBACK[i]]:
#      #       fig2.add_trace(go.Scatter(x=[arg],y=[dbBACK[coldbBACK[i]][arg]],
#      #                                marker=dict(color="crimson", size=12),
#      #                                mode="markers",
#      #                                name=coldbBACK[i]+' BACK '+str(arg)),
#      #                                row=2, col=1)
#      # except:
#      #    1
#    except:
#          continue;
#
#
## fig.update_traces(mode="arkers+lines")
#
#fig3.update_layout(
#    hoverlabel=dict(
#        namelength=-1
#    )
#)
#fig3.update_layout(title='Wave-'+colorDic[clrNumber])
#
#fig3.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
#
#   
#
## plot(fig2)  
#now = datetime.now()
## dd/mm/YY H:M:S
#dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
#plot(fig3,filename="Wave_AQM_"+ dt_string +".html")  

# 
##### TILL HERE!!!!