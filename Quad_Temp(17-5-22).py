# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:16:56 2022

@author: Ireneg
"""
# from IPython import get_ipython
# get_ipython().magic('reset -sf')

import plotly.io as pio
pio.renderers
pio.renderers.default='browser'



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



import tkinter as tk
from tkinter.filedialog import askopenfilename
import pandas as pd

root = tk.Tk()
root.withdraw() #Prevents the Tkinter window to come up
exlpath = askopenfilename()
root.destroy()
print(exlpath)
RawData = pd.read_csv(exlpath)   
    
    
col = list(RawData.columns);
for c in col:
    if 'Flex2' in c:
      RawData=RawData.drop([c],axis=1);    



RawDataNoFlex=RawData.copy();
col = list(RawData.columns);
for c in col:
    if 'Flex1' in c:
      RawDataNoFlex=RawDataNoFlex.drop([c],axis=1);   

col = list(RawDataNoFlex.columns);
QaudDic={};
QdList=[]
k=0;
for c in col:
    Ph = c.split('_')
    if len(Ph)>1:
        PhNum= int(Ph[0][2:]);
        if not (PhNum % 4 == 0):
            QdList.append(c);
        else:
            QdList.append(c);
            if 'TempAmp2' in Ph[1]:
                QaudDic['Qaud_'+str(k)] = QdList;
                QdList=[];
                k=k+1;


    
        
# v = RawDataNoFlex[QaudDic['Qaud_0']].mean(axis=1).rename('Qaud_0')

RawDataNoFlexQaud= pd.DataFrame();

for key, value in QaudDic.items():
    RawDataNoFlexQaud=pd.concat([RawDataNoFlexQaud,RawDataNoFlex[QaudDic[key]].mean(axis=1).rename(key)],axis =1)
    



pthComp=exlpath.split('/');
RecPath = pthComp[0] + '/';
for i,pt in enumerate(pthComp):
    if i>0 and i<len(pthComp)-1:
        RecPath= RecPath + pt + '/';

os.chdir(RecPath)
##########################################################################################

fig = go.Figure()

db=RawData


col=list(db.columns)

rnge=range(len(col))

for c in col:
    if not('Flex1' in c):
        continue;
    fig.add_trace(go.Scatter(y=list(db[c]),
                name=c))


fig.update_layout(title='Flex1')

fig.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
# 
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig,auto_play=True,filename="Flex1_"+ dt_string +".html")  
#plot(fig_back,filename="AQM-Back.html")  
fig.show()


#################################################################################


fig1 = go.Figure()

db=RawData


col=list(db.columns)

rnge=range(len(col))

for c in col:
    if not('Amp1' in c):
        if not('Amp2' in c):
            continue;
    fig1.add_trace(go.Scatter(y=list(db[c]),
                name=c))


fig1.update_layout(title='Amp1 & Amp2')


        

fig1.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
# 

now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig1,auto_play=True,filename="Amp1 & Amp2_"+ dt_string +".html")  
#plot(fig_back,filename="AQM-Back.html")  
fig1.show()


#################################################################################


fig2 = go.Figure()

db=RawDataNoFlexQaud


col=list(db.columns)

rnge=range(len(col))

for c in col:
    fig2.add_trace(go.Scatter(y=list(db[c]),
                name=c))


fig2.update_layout(title='Quad Avarege')


        

fig2.update_layout(
    hoverlabel=dict(
        namelength=-1
    )
)
# 

now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plot(fig2,auto_play=True,filename="Quad_"+ dt_string +".html")  
#plot(fig_back,filename="AQM-Back.html")  
fig2.show()












