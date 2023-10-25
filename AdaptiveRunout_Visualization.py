# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:53:55 2023

@author: Ireneg
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import os
###############################################################
global PlotRead,PlotWrite,plot_Bar2show,Title_dictionary

PlotRead = 1
PlotWrite= 1
plot_Bar2show= [2,3,4,5,6,7,8]
    
Title_dictionary = {
    "Read table/write table": 0,
    "Date time": 1,
    "Bar Id": 2,
    "Press speed": 3,
    "Is auto calibration": 4,
    "Num of revolutions": 5,
    "Calibration weight": 6,
    "Sector 1": 7,
    "Sector 2": 8,
    "Sector 3": 9,
    "Sector 4": 10,
    "Sector 5": 11,
    "Sector 6": 12,
    "Sector 7": 13,
    "Sector 8": 14,
    "Sector 9": 15,
    "Sector 10": 16,
    "Sector 11": 17,
    "Sector 12": 18,
    "Sector 13": 19,
    "Sector 14": 20,
    "Sector 15": 21,
    "Sector 16": 22,
    "Sector 17": 23,
    "Sector 18": 24,
    "Sector 19": 25,
    "Sector 20": 26,
    "Sector 21": 27,
    "Sector 22": 28,
    "Sector 23": 29,
    "Sector 24": 30,
    "Sector 25": 31,
    "Sector 26": 32,
    "Sector 27": 33,
    "Sector 28": 34,
    "Sector 29": 35,
    "Sector 30": 36,
    "Sector 31": 37,
    "Sector 32": 38
}

class DataVisualization:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path,index_col=False)

    def filter_by_write_table(self):
        self.filtered_data_write = self.data[self.data.iloc[:, Title_dictionary['Read table/write table']] == 'Write table'].reset_index(drop=True)
        self.filtered_data_read = self.data[self.data.iloc[:, Title_dictionary['Read table/write table']] == 'Read table'].reset_index(drop=True)
        

    def create_dataframes_by_bar_id(self):
        self.unique_bar_ids_write = self.filtered_data_write.iloc[:, Title_dictionary['Bar Id']].unique()
        self.dataframes_write = {bar_id: self.filtered_data_write[self.filtered_data_write.iloc[:, Title_dictionary['Bar Id']] == bar_id].reset_index(drop=True) for bar_id in self.unique_bar_ids_write}
        self.unique_bar_ids_read= self.filtered_data_read.iloc[:, Title_dictionary['Bar Id']].unique()
        self.dataframes_read = {bar_id: self.filtered_data_read[self.filtered_data_read.iloc[:, Title_dictionary['Bar Id']] == bar_id].reset_index(drop=True) for bar_id in self.unique_bar_ids_read}

    def plot_data(self, color_list,colorListCode,plotTitle):
        fig = go.Figure()
        
        for index, row in self.data.iterrows():
            date_time = row[Title_dictionary['Date time']]
            bar_id = row[Title_dictionary['Bar Id']]
            if not(bar_id in plot_Bar2show):
                continue;
            read_write= row[Title_dictionary['Read table/write table']]
            if read_write == 'Write table' and not PlotWrite:
                continue;
            if read_write == 'Read table' and not PlotRead:
                continue;
            color = color_list.get(bar_id, 'gray')  # Default to gray if not found in colorListNm
            
            # Extract values from the 8th column to the last column
            values = row[7:].values
            
            # Create a trace for the current row
            trace = go.Scatter(x=list(range(8, len(values) + 8)), y=values, mode='lines', 
                               name=f"{read_write} - {date_time} (Bar {bar_id})", line=dict(color=colorListCode[color]), hovertemplate=f"%{{y}}<br>Name:{read_write} - {date_time} (Bar {bar_id})", hoverinfo='none')
            
            # Add the trace to the figure
            fig.add_trace(trace)
         # Create a trace for the average line
        average_values_all = self.data.iloc[:, 7:].mean(axis=0)
        average_values_read = self.filtered_data_read.iloc[:, 7:].mean(axis=0)
        average_values_write = self.filtered_data_write.iloc[:, 7:].mean(axis=0)

        average_trace_all = go.Scatter(x=list(range(8, len(average_values_all) + 8)), y=average_values_all, 
                                   mode='lines', name='Average all', line=dict(color='red', width=4))
        if  PlotRead:
            average_trace_read = go.Scatter(x=list(range(8, len(average_values_read) + 8)), y=average_values_read, 
                                   mode='lines', name='Average read', line=dict(color='green', width=4))
            fig.add_trace(average_trace_read)
            
        if PlotWrite:
            average_trace_write = go.Scatter(x=list(range(8, len(average_values_write) + 8)), y=average_values_write, 
                                   mode='lines', name='Average write', line=dict(color='purple', width=4))
            fig.add_trace(average_trace_write)

        
        average_values_barID_write={}
        average_values_barID_read={}
        
        average_barId_trace_write=[]
        average_barId_trace_read=[]


        for barID in self.unique_bar_ids_write:
            if not(barID in plot_Bar2show):
                continue
            try:
                if PlotWrite:
                    average_values_barID_write[barID]= self.dataframes_write[barID].iloc[:, 7:].mean(axis=0)
                    average_barId_trace_write.append(go.Scatter(x=list(range(8, len(average_values_barID_write[barID]) + 8)), y=average_values_barID_write[barID], 
                                               mode='lines', name='Write average bar '+str(barID), line=dict(color=color_list[barID], width=3)))
                if PlotRead:
                    average_values_barID_read[barID]= self.dataframes_read[barID].iloc[:, 7:].mean(axis=0)
                    
                    average_barId_trace_read.append(go.Scatter(x=list(range(8, len(average_values_barID_read[barID]) + 8)), y=average_values_barID_read[barID], 
                                              mode='lines', name='Read average bar '+str(barID), line=dict(color=color_list[barID], dash='dash', width=3)))
            except:
                continue

        
        # Add the average trace to the figure
        fig.add_trace(average_trace_all)
        

        if PlotWrite:
            for itm in average_barId_trace_write:
                fig.add_trace(itm)
        
        if PlotRead:    
            for itm in average_barId_trace_read:
                fig.add_trace(itm)
            
        
        fig.update_layout(title=plotTitle,
                          xaxis_title='Column Index',
                          yaxis_title='Value')
        fig.update_layout(
                hoverlabel=dict(namelength=-1, font=dict(color='black'))
            )
        
        plot(fig,auto_play=True,filename=plotTitle+'.csv')  

        fig.show()

if __name__ == "__main__":
    # Define your color mapping dictionary
    colorListNm = {2: 'black', 3: 'blue', 4: 'cyan', 5: 'green', 6: 'magenta', 7: 'orange', 8: 'yellow'}
    colorListCode = {
    'black': 'rgba(0, 0, 0, 0.15)',
    'blue': 'rgba(0, 0, 255, 0.15)',
    'cyan': 'rgba(0, 255, 255, 0.15)',
    'green': 'rgba(0, 128, 0, 0.15)',
    'magenta': 'rgba(255, 0, 255, 0.15)',
    'orange': 'rgba(255, 165, 0, 0.15)',
    'yellow': 'rgba(255, 223, 0, 0.15)'
    }
    
    from tkinter import filedialog
    from tkinter import *
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    parts = file_path.split('/')
    
    # Join all parts except the last one to get the folder path
    folder_path = '/'.join(parts[:-1])
    
    os.chdir(folder_path)
    # Instantiate the DataVisualization class and perform the steps
    data_visualization = DataVisualization(file_path)
    data_visualization.filter_by_write_table()
    data_visualization.create_dataframes_by_bar_id()
    plotTitle=parts[len(parts)-1][:-4]+'_RunOut'
    data_visualization.plot_data(colorListNm,colorListCode,plotTitle)
    
 
    
#######################################################
 