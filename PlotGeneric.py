# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 11:19:02 2022

@author: Ireneg
"""
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
import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter

import plotly.express as px



from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()
pthF = filedialog.askopenfilename()


# f=pthF.split('/')[len(pthF.split('/'))-1]
DirectorypathF=pthF.split('/');
os.chdir(DirectorypathF[0]+'/'+DirectorypathF[1])
# List arguments in wide form


# fig = px.line(x=list(RawDataDancer[2]), y=list(RawDataDancer[3]))
# fig.show()
# plot(fig)     

RawDataDancer=pd.read_csv(pthF,header = None);




RawDataDancer['DayTime'] = pd.to_datetime(RawDataDancer[2],dayfirst=True)

DateDuc={}

for i in range(len(RawDataDancer['DayTime'])):
    DateDuc[i]=RawDataDancer['DayTime'][i];

RawDataDancer=RawDataDancer.rename(index= DateDuc)  
RawDataDancer = RawDataDancer.sort_index()

fig = go.Figure()


    
fig.add_trace(go.Scatter(y=list(RawDataDancer[3]),x=RawDataDancer['DayTime'],mode='markers',
                name='Dancer Position'))


# fig.update_layout(title='ImagePlacement_Right')
fig.update_layout(title='Dancer Position')



plot(fig)  