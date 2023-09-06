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


class DataVisualization:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path,index_col=False)

    def filter_by_write_table(self):
        self.filtered_data_write = self.data[self.data['Read table/write table'] == 'Write table']
        self.filtered_data_read = self.data[self.data['Read table/write table'] == 'Read table']

    def plot_data(self, color_list,colorListCode,plotTitle):
        fig = go.Figure()
        
        for index, row in self.data.iterrows():
            date_time = row[' Date time']
            bar_id = row['Bar Id']
            read_write= row['Read table/write table']
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
        
        average_trace_read = go.Scatter(x=list(range(8, len(average_values_read) + 8)), y=average_values_read, 
                                   mode='lines', name='Average read', line=dict(color='green', width=4))
        
        average_trace_write = go.Scatter(x=list(range(8, len(average_values_write) + 8)), y=average_values_write, 
                                   mode='lines', name='Average write', line=dict(color='purple', width=4))
        
        # Add the average trace to the figure
        fig.add_trace(average_trace_all)
        fig.add_trace(average_trace_read)

        fig.add_trace(average_trace_write)

        
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
    plotTitle=parts[len(parts)-1][:-4]+'_RunOut'
    data_visualization.plot_data(colorListNm,colorListCode,plotTitle)

