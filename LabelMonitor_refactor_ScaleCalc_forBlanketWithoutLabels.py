import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import medfilt2d, medfilt
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.signal import correlate2d
from scipy.ndimage import morphology
from scipy.ndimage import convolve
import os
from sklearn.cluster import KMeans
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.subplots import make_subplots
import re
from scipy.signal import savgol_filter
from tkinter import filedialog
from tkinter import *
import pickle
import copy
from datetime import datetime

####################################################################
global S_G_window,S_G_Degree,TimeBetweenPrints,PlotContinuesBalnkLength,PlotSTD,Joblength_limit,Panel2reset

S_G_window =10
S_G_Degree =1
TimeBetweenPrints = 10

plot_LabelLength_Per_bar=0
Plot_Panel_Length_Per_Bar_and_g_s=0
Plot_Panel_Length_Per_Bar_and_After_g_s=0
PlotContinuesBalnkLength =0
PlotSTD =0
plotJob_Sgolay_std =0
plotJob_Sgolay_db =0
Joblength_limit = int(100/11)


##For NO LABELS calc
Panel2reset=3 # The panel number for which errors reset 
####################################################################
####################################################################
####################################################################

class DataProcessor:
    def __init__(self):
        self.TimeBetweenPrints = TimeBetweenPrints
        self.S_G_window= S_G_window
        self.S_G_Degree= S_G_Degree
        self.colorList = {
            'DPSBar2': 'black',
            'DPSBar3': 'blue',
            'DPSBar4': 'cyan',
            'DPSBar5': 'green',
            'DPSBar6': 'magenta',
            'DPSBar7': 'orange',
            'DPSBar8': 'gold'
        }

    def load_csv(self, csv_file_path):
        db = pd.read_csv(csv_file_path, index_col=False)
        db_filtered = db[(db != -1).all(axis=1)].reset_index(drop=True)
        db_filtered['date_time'] = pd.to_datetime(db_filtered['Date'])
        col = db_filtered.columns
        col_range = col[4:15]
        return db_filtered, col_range

    def create_print_list(self, db):
        time_diff = db['date_time'].diff()
        rows_with_time_diff = db[time_diff > pd.Timedelta(seconds=self.TimeBetweenPrints)][['Date', 'date_time']]
        index_print_name_dic = dict(zip(rows_with_time_diff.index, rows_with_time_diff['Date']))
        return index_print_name_dic

    def calc_panel_mean(self, db, col_range):
        column_means = db[col_range].median()
        db[col_range] = db[col_range] - column_means
        mean_dict = column_means.to_dict()
        return db, mean_dict
    
    def calc_panel_S_goly(self, db, col_range):
        for col in col_range:
            y= savgol_filter(db[col],self.S_G_window,self.S_G_Degree)    
            db[col] = db[col] - y
        return db
    
    def create_continues_panel_lngth_per_bar(self,labelData,colrnge):
        
        ContinuesPanelLngth=pd.DataFrame();
        
        for barNum in labelData.keys():
            
            db= labelData[barNum].copy()
            # db,mean_Panel= self.calc_panel_mean(db,colrnge)
            db= self.calc_panel_S_goly(db,colrnge)

            array = db[colrnge].values
            LongArray = array.reshape(-1)
            new_column_series = pd.Series(list(LongArray), name=barNum)

            ContinuesPanelLngth = pd.concat([ContinuesPanelLngth, new_column_series], axis=1)

            
        return ContinuesPanelLngth

    
    def Create_S_Golay_panel_label_length(self,db,col_range,col_range_g_s):
        
        db_s_g=pd.DataFrame()
        y=[]
        for i,col in enumerate(col_range):
            
          y= savgol_filter(db[col],self.S_G_window,self.S_G_Degree)
          db_s_g=pd.concat((db_s_g,pd.Series(y).rename(col_range_g_s[i])),axis=1)

      
        return   db_s_g          
    
    def Create_average_perPrint(self,db,col_range,col_range_meanVal,index_print_name_dic):
        
        db_mean_val=pd.DataFrame()
        y=[]
        key_perv=0
        for i,col in enumerate(col_range):
            for key in index_print_name_dic.keys():
                
                y=y+ list([np.mean(db[col][key_perv:key])] * (key-key_perv))
                key_perv=key+1
            key=len(db[col])
            y=y+ list([np.mean(db[col][key_perv:key])] * (key-key_perv))
            db_mean_val=pd.concat((db_mean_val,pd.Series(y).rename(col_range_meanVal[i])),axis=1)
            y=[]
      
        return   db_mean_val          
    
    def Create_printLength_db(self,index_print_name_dic_data):
        
        JobLength_st_en={}
        for key,value in index_print_name_dic_data.items():
            barPrintindex=list(value.keys());
            
            JobLength=np.diff(barPrintindex)
            
            JobLength_st_en[key] = [[item1, item2, item3] for item1, item2, item3 in zip(JobLength, barPrintindex[:-1], barPrintindex[1:])]

            
        return JobLength_st_en
            
    def Calc_S_Goly_diff_labelData(self,col_range,col_range_g_s,label_data,label_data_g_s):      
            
        label_data_g_s_db={}

        col_s_g_db={}

        for i in range(len(col_range_g_s)):
            col_s_g_db[col_range[i]]=col_range_g_s[i]

        for key in label_data.keys():
              label_data_g_s_db[key]=label_data[key].rename(columns=col_s_g_db)-label_data_g_s[key]
        
        
        return label_data_g_s_db
        
        

class Plotter:
    def __init__(self):
        self.data_processor = DataProcessor()

    def single_plot(self, db, col_range, plot_title, file_name, index_print_name_dic, ymax, all_bars, fig,showFig):
        
        if not fig:
            fig = go.Figure()

        if not ymax:
            ymax = np.mean(list(db[col_range[int(len(col_range) / 2)]]))

        for i, c in enumerate(col_range):
            try:
                fig.add_trace(go.Scatter(y=list(db[c]), name=c, line_color=self.data_processor.colorList[c]))
                fig = self.plot_print(fig, ymax + 100000 * i, index_print_name_dic[c], self.data_processor.colorList[c], c)
            except:
                fig.add_trace(go.Scatter(y=list(db[c]), name=c))

        if not all_bars:
            fig = self.plot_print(fig, ymax, index_print_name_dic, 'Green', '')

        fig.update_layout(title={
            'text': plot_title,
            'font': {'color': 'black'}
        })

        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        if showFig:
            plot(fig, auto_play=True, filename=file_name)
            fig.show()
        return fig

    def plot_print(self, fig, ymax, index_print_name_dic, clr, br):
        for key, value in index_print_name_dic.items():
            if  isinstance(br, dict):
                br_print=br[key]
            else:
                br_print=br
            fig.add_trace(go.Scatter(x=[key], y=[ymax],
                                     marker=dict(color=clr, size=10),
                                     mode="markers",
                                     text=value + ' ' + br_print,
                                     hoverinfo='text'))
            fig.data[len(fig.data) - 1].showlegend = False
            fig.add_vline(x=key, line_width=2, line_dash="dash", line_color=clr)
        return fig

    def single_plot_array(self, long_array, plot_title, file_name, index_print_name_dic, ymax,fig):
        
        if not fig:
            fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(long_array), name='Panel'))

        fig.update_layout(title={
            'text': plot_title,
            'font': {'color': 'black'}
        })

        if not ymax:
            ymax = abs(np.mean(list(long_array)))
        


        fig = self.plot_print(fig, ymax, index_print_name_dic, 'green', '')

        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )

        plot(fig, auto_play=True, filename=file_name)
        fig.show()
        return fig
    
    def regular_plot(self, db,  plot_title, file_name, titleX, titleY,key,  fig,showFig):
        
        if not fig:
            fig = go.Figure()
            
        for c in db.columns:
            fig.add_trace(go.Scatter(y=list(db[c]*1e-3), name=c+' '+key, line_color=self.data_processor.colorList[key] ))
            
        
        fig.update_layout(title={
            'text': plot_title,
            'font': {'color': 'black'}
        },
            xaxis=dict(title=titleX),
            yaxis=dict(title=titleY))
        
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        
        if showFig:
            plot(fig, auto_play=True, filename=file_name)
            fig.show()
         
        return fig

    def regular_plot_C2C(self, db,  plot_title, file_name, titleX, titleY,key,index_Print_C2C  ,fig,showFig):
        
        if not fig:
            fig = go.Figure()
            

        fig.add_trace(go.Scatter(y=list(db*1e-3), name=key ))
            
        
        fig.update_layout(title={
            'text': plot_title,
            'font': {'color': 'black'}
        },
            xaxis=dict(title=titleX),
            yaxis=dict(title=titleY))
        
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        ymax = abs(np.mean(list(db*1e-3)))
        if isinstance(index_Print_C2C, dict):
            fig = self.plot_print(fig, ymax+100, index_Print_C2C, 'green', '')

        
        if showFig:
            plot(fig, auto_play=True, filename=file_name)
            fig.show()
         
        return fig   

        
    def regular_plot_Each_Color(self, db,  plot_title, file_name, titleX, titleY,key,index_Print_C2C  ,fig,showFig):
        
        sgWindow=50
        if not fig:
            fig = go.Figure()
            

        for col in db.columns:
            fig.add_trace(go.Scatter(y=list(db[col]*1e-3), name=col, line_color=self.data_processor.colorList[col] ))
            
        for col in db.columns:
            fig.add_trace(go.Scatter(y=savgol_filter(list(db[col]*1e-3),sgWindow,1) , name=col+' S_G filter = '+str(sgWindow), line_color=self.data_processor.colorList[col] ))
            
        
        fig.update_layout(title={
            'text': plot_title,
            'font': {'color': 'black'}
        },
            xaxis=dict(title=titleX),
            yaxis=dict(title=titleY))
        
        fig.update_layout(
            hoverlabel=dict(
                namelength=-1
            )
        )
        ymax = abs(np.mean(list(db[col]*1e-3)))
        
        indexPrints=list(index_Print_C2C.keys())
        indexPrintsSTD={}
        for i,index in enumerate(indexPrints):
            if i==0:
                stindex=0
                enindex=index
            else:
                stindex=indexPrints[i-1]
                enindex=index
            Std_sorted={}
            for c in db.columns:
                Std_sorted[np.std(db[c][stindex:enindex]*1e-3)]=c
            
            keysList=sorted(list(Std_sorted.keys()),reverse=True)
            
            pctext='<br>'+Std_sorted[keysList[0]]+':'+'STD '+"{:.2f}".format(keysList[0]);    
            for key in keysList[1:]:
                pctext=pctext+'<br>'+Std_sorted[key]+':'+'STD '+"{:.2f}".format(key);
            indexPrintsSTD[index]=pctext
            pctext=''
            
        if isinstance(index_Print_C2C, dict):
            fig = plotter.plot_print(fig, ymax+100, index_Print_C2C, 'green', indexPrintsSTD)

        
        if showFig:
            plot(fig, auto_play=True, filename=file_name)
            fig.show()
         
        return fig   


class C2C_From_Panel_Length_Difference:
    def __init__(self,label_data,index_print_name_dic_data):
        ''' constructor ''' 
        self.label_data = label_data
        self.index_print_name_dic_data=index_print_name_dic_data
        
    def printSessionStartIndexesPerBar(self):
        BarKeys={}
        for key,value in self.index_print_name_dic_data.items():
            BarKeys[key]=list(value.keys())
        
        return BarKeys
    
    def filterSyncPrintsBetweenBars(self):
        
        BarKeys=self.printSessionStartIndexesPerBar()
        printSession_sync_dic={}  
    
        tmp=pd.DataFrame()  
    
        arr=np.diff(BarKeys[list(BarKeys.keys())[0]])
         
        # for self.DynamicThreshold in range(30,100):
        for self.DynamicThreshold in range(5,100):

            indices = np.where(abs(arr-self.DynamicThreshold) < 3)[0]
            if not len(indices):
                break;
    
        for key in BarKeys.keys():
            arr=np.diff(BarKeys[key])
    
            # indexBorder = np.where((arr == 30 + 1) | (arr == 30 + 2) | (arr == 30))[0]
            # arrBoder=pd.concat([arrBoder,pd.DataFrame(list(arr[indexBorder]))],axis=1).rename(columns={0: key})  
            indices = np.where(arr>self.DynamicThreshold)[0]
            printSession_sync_dic[key]=[]
            for inx in indices:
                printSession_sync_dic[key].append([BarKeys[key][inx],BarKeys[key][inx+1],arr[inx]])
            
            tmp=pd.concat([tmp,pd.DataFrame(list(arr[indices]))],axis=1).rename(columns={0: key})   
    
        Sync_print_length = tmp.min(axis=1)    
        
        return Sync_print_length,printSession_sync_dic

    def findMiddleTimeStampAndPrintStarts(self):
        
        Sync_print_length,printSession_sync_dic=self.filterSyncPrintsBetweenBars()
        Middel_Clock_Value_bar={}
        for key in printSession_sync_dic.keys():
            Middel_Clock_Value_bar[str(self.label_data[key]['date_time'][printSession_sync_dic[key][0][0]])]=key
            
        timestamps=list(Middel_Clock_Value_bar.keys())
            
        timestamps = sorted([datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps])

        # Find the middle timestamp
        middle_timestamp_index = len(timestamps) // 2
        middle_timestamp = timestamps[middle_timestamp_index]

        Main_Clock_bar=Middel_Clock_Value_bar[str(middle_timestamp)]

        index_Print_C2C={}
        Acamulate_PrintLength=0
        i=0
        printLength=Sync_print_length[0]
        for i,printLength in enumerate(Sync_print_length):
            Acamulate_PrintLength=Acamulate_PrintLength+printLength*10
            index_Print_C2C[Acamulate_PrintLength]=self.label_data[Main_Clock_bar]['Date'][printSession_sync_dic[Main_Clock_bar][i][0]]
    
        return Main_Clock_bar, index_Print_C2C

    def Find_Print_Sessions(self):
        
        Sync_print_length,printSession_sync_dic=self.filterSyncPrintsBetweenBars()
        col_range=list(self.label_data[list(self.label_data.keys())[0]].columns)[16:28]
        MaxPrintSession={}
        for i,key in enumerate(self.label_data.keys()):
            tmp_continues=pd.DataFrame()
            for l,printLength in enumerate(Sync_print_length):
                tmp=pd.DataFrame()
                for col in col_range[2:]:
                   inx1=printSession_sync_dic[key][l][0]
                   inx2=inx1+printLength    
                   tmp=pd.concat([tmp,self.label_data[key][col][inx1:inx2].reset_index(drop=True)],axis=1)
                tmp_continues=pd.concat([tmp_continues,tmp],axis=0).reset_index(drop=True)

            MaxPrintSession[key]=tmp_continues
            
        return MaxPrintSession
    
    def filterErrorValuePrints(self,labelData):
        index_to_delete=[]
        for key in labelData.keys():
            if len(labelData[key][labelData[key] <  7.5*1e8].stack().index.tolist()):
                index_to_delete=labelData[key][labelData[key] <  7.5*1e8].stack().index.tolist()+index_to_delete

        unique_indexes = set()

        # Iterate through the tuple list to extract unique indexes
        for tup in index_to_delete:
            unique_indexes.add(tup[0])
        
        # Convert set to a sorted list
        unique_index_list_to_delete = sorted(list(unique_indexes))
        
        return unique_index_list_to_delete
    
    def create_continues_panel_lngth_per_bar(self,labelData,colrnge):
        
        unique_index_list_to_delete=self.filterErrorValuePrints(labelData)
        if len(unique_index_list_to_delete):
            val=labelData['DPSBar2']['Label[1,2]DistanceNM'][unique_index_list_to_delete[0]-1]
            # labelData_copy=copy.deepcopy(labelData)
            
            for key in labelData.keys():
                labelData[key].iloc[unique_index_list_to_delete] = val

        
        ContinuesPanelLngth=pd.DataFrame();
     
  
        
        for barNum in labelData.keys():
            
            db= labelData[barNum].copy()
            
            # dbFilter=copy.deepcopy(labelData[barNum])
            # db,mean_Panel= self.calc_panel_mean(db,colrnge)
            db= plotter.data_processor.calc_panel_S_goly(db,colrnge)
    
            # arrayForFilter=dbFilter[colrnge].values
            array = db[colrnge].values
            LongArray = array.reshape(-1)
            # LongArrey_forFilter=arrayForFilter.reshape(-1)
            new_column_series = pd.Series(list(LongArray), name=barNum)
            
            
            # new_column_seriesForFilter = pd.Series(list(LongArrey_forFilter), name=barNum)
    
            ContinuesPanelLngth = pd.concat([ContinuesPanelLngth, new_column_series], axis=1)
            
        return ContinuesPanelLngth,unique_index_list_to_delete
    
    def AdjastIndexDic_afterFilre(self,unique_index_list_to_delete):
        
        Main_Clock_bar, index_Print_C2C= self.findMiddleTimeStampAndPrintStarts()

        if len(unique_index_list_to_delete):
            unique_index_list_to_delete_by10=multiply_by_num(unique_index_list_to_delete, 10)
            identical_values=find_identical_values(unique_index_list_to_delete_by10, list(index_Print_C2C.keys()))
            if len(identical_values):
               index_Print_C2C=delete_keys_from_dict(index_Print_C2C, identical_values) 
               
        return Main_Clock_bar, index_Print_C2C
    
    def Calc_C2C_for_ongoing_printing(self):
        
        self.MaxPrintSession=self.Find_Print_Sessions()
        key='DPSBar2'
        self.ContinuesPanelLngth,self.unique_index_list_to_delete=self.create_continues_panel_lngth_per_bar(self.MaxPrintSession,self.MaxPrintSession[key].columns)
        
        # plt.figure()
        # # for i in range(7):
        # # plt.plot( ContinuesPanelLngth['DPSBar2']) 
        # # plt.plot( ContinuesPanelLngth['DPSBar4']) 
        # # plt.plot( ContinuesPanelLngth['DPSBar6']) 
        # # plt.plot( ContinuesPanelLngth['DPSBar8']) 
        # plt.plot( ContinuesPanelLngth['DPSBar3']) 
        # # plt.plot( ContinuesPanelLngth['DPSBar5'])
        self.ContinuesPanelLngth_noLabels=self.ContinuesPanelLngth.copy()
        for inx in range(len(self.ContinuesPanelLngth_noLabels)):
            for col  in self.ContinuesPanelLngth_noLabels:
                if inx%Panel2reset == 0:
                    continue;
                else:
                    if inx>0:
                       self.ContinuesPanelLngth_noLabels[col][inx]= self.ContinuesPanelLngth_noLabels[col][inx]+self.ContinuesPanelLngth_noLabels[col][inx-1]


        C2C_diff=[]
        C2C_diff_no_labels=[]


        for inx in range(len(self.ContinuesPanelLngth)):
            maxVal=-1e10
            minVal=1e10
            maxVal_noLabel=-1e10
            minVal_noLabel=1e10
            for col  in self.ContinuesPanelLngth.columns:
                if maxVal<self.ContinuesPanelLngth[col][inx]:
                    maxVal=self.ContinuesPanelLngth[col][inx]
                if minVal>self.ContinuesPanelLngth[col][inx]:
                    minVal=self.ContinuesPanelLngth[col][inx]
                    
                if maxVal_noLabel<self.ContinuesPanelLngth_noLabels[col][inx]:
                    maxVal_noLabel=self.ContinuesPanelLngth_noLabels[col][inx]
                if minVal_noLabel>self.ContinuesPanelLngth_noLabels[col][inx]:
                    minVal_noLabel=self.ContinuesPanelLngth_noLabels[col][inx]
            C2C_diff.append(maxVal-minVal)
            C2C_diff_no_labels.append(maxVal_noLabel-minVal_noLabel)

 
        
        # plt.figure()
        # plt.plot(C2C_diff)

                                     
        C2C_continues=pd.DataFrame(C2C_diff) 
        C2C_continues_no_labels=pd.DataFrame(C2C_diff_no_labels) 

        return C2C_continues,C2C_continues_no_labels
    




def create_even_odd_dict(d):
    """
    Create a dictionary from the even and odd index key-value pairs of an existing dictionary.

    Args:
    - d (dict): The original dictionary.

    Returns:
    - dict: A new dictionary containing the even and odd index key-value pairs.
    """
    even_dict = {k: v for i, (k, v) in enumerate(d.items()) if i % 2 == 0}
    odd_dict = {k: v for i, (k, v) in enumerate(d.items()) if i % 2 != 0}
    return even_dict,odd_dict

def find_identical_values(list1, list2):
    identical_values = []
    inxList=[]
    for item in list1:
        if item in list2:
            inxList.append(list2.index(item))
            identical_values.append(item)
    return identical_values,inxList

def multiply_by_num(input_list,num):
    multiplied_list = [x * num for x in input_list]
    return multiplied_list

def delete_keys_from_dict(input_dict, keys_to_delete):
    for key in keys_to_delete:
        if key in input_dict:
            del input_dict[key]
    return input_dict

def subtract_value_from_list(input_list, value):
    subtracted_list = [x - value for x in input_list]
    return subtracted_list

def subtract_value_from_list_from_index(input_list, value, start_index):
    subtracted_list = input_list.copy()
    for i in range(start_index, len(input_list)):
        subtracted_list[i] -= value
    return subtracted_list

def replace_keys_in_dict(input_dict, new_keys):
    if len(new_keys) != len(input_dict):
        raise ValueError("Number of new keys must match number of keys in the dictionary.")
    
    replaced_dict = dict(zip(new_keys, input_dict.values()))
    return replaced_dict

# def create_continues_panel_lngth_per_bar(labelData,colrnge):
    
#     unique_index_list_to_delete=filterErrorValuePrints(labelData)
    
#     # labelData_copy=copy.deepcopy(labelData)
#     for barNum in labelData.keys():
#         labelData[barNum]=labelData[barNum].drop(unique_index_list_to_delete)
#         labelData[barNum]=labelData[barNum].reset_index(drop=True) 
    
#     ContinuesPanelLngth=pd.DataFrame();
    
#     for barNum in labelData.keys():
        
#         db= labelData[barNum].copy()
        
#         # dbFilter=copy.deepcopy(labelData[barNum])
#         # db,mean_Panel= self.calc_panel_mean(db,colrnge)
#         db= plotter.data_processor.calc_panel_S_goly(db,colrnge)

#         # arrayForFilter=dbFilter[colrnge].values
#         array = db[colrnge].values
#         LongArray = array.reshape(-1)
#         # LongArrey_forFilter=arrayForFilter.reshape(-1)
#         new_column_series = pd.Series(list(LongArray), name=barNum)
#         # new_column_seriesForFilter = pd.Series(list(LongArrey_forFilter), name=barNum)

#         ContinuesPanelLngth = pd.concat([ContinuesPanelLngth, new_column_series], axis=1)
#         # ContinuesPanelLngth_forFiltiring = pd.concat([ContinuesPanelLngth_forFiltiring, new_column_seriesForFilter], axis=1)
      
    
#     return ContinuesPanelLngth

# def filterErrorValuePrints(labelData):
#     index_to_delete=[]
#     for key in labelData.keys():
#         if len(labelData[key][labelData[key] <  6*1e8].stack().index.tolist()):
#             index_to_delete=labelData[key][labelData[key] <  6*1e8].stack().index.tolist()+index_to_delete

#     unique_indexes = set()

#     # Iterate through the tuple list to extract unique indexes
#     for tup in index_to_delete:
#         unique_indexes.add(tup[0])
    
#     # Convert set to a sorted list
#     unique_index_list_to_delete = sorted(list(unique_indexes))
    
#     return unique_index_list_to_delete
    
    
#######################################################
#######################################################
#######################################################
# def main():
plotter = Plotter()

root = Tk()
root.withdraw()
sInput = filedialog.askdirectory()
os.chdir(sInput)
file_list = os.listdir(sInput)
csv_files = [file for file in file_list if file.endswith('.csv')]

figure_list = []
label_data = {}
label_data_g_s = {}

index_print_name_dic_data = {}

for i in range(len(csv_files)):
    csv_file_path = os.path.join(sInput, csv_files[i])
    st = csv_file_path.find('DPSBar')
    plot_title = csv_file_path[st:-32]
    file_name = plot_title + ' label length' + '.html'
    db, col_range1 = plotter.data_processor.load_csv(csv_file_path)
    label_data[plot_title] = db
    index_print_name_dic = plotter.data_processor.create_print_list(db)
    index_print_name_dic_data[plot_title] = index_print_name_dic
    if plot_LabelLength_Per_bar:
        figure_list.append(
            plotter.single_plot(db, col_range1, plot_title + ' label length', file_name, index_print_name_dic, 0, 0,0,0))

figure_list2 = []

col_range = db.columns[16:28]
col_range_g_s=[]
col_range_meanVal=[]
for col in col_range:
    col_range_g_s.append(col+' g_s filter window='+str(plotter.data_processor.S_G_window))
    col_range_meanVal.append(col+' mean Value per print')


for key in label_data.keys():
    plot_title = key + ' panel length'
    file_name = plot_title + '.html'
    db = label_data[key]
    index_print_name_dic = index_print_name_dic_data[key]

    db_s_g= plotter.data_processor.Create_S_Golay_panel_label_length(db, col_range,col_range_g_s)
    db_mean_val= plotter.data_processor.Create_average_perPrint(db,col_range,col_range_meanVal,index_print_name_dic)
    
    label_data_g_s[key]= db_s_g
    if Plot_Panel_Length_Per_Bar_and_g_s:
        figure_list2.append(
            plotter.single_plot(db, col_range, plot_title, file_name, index_print_name_dic, 0, 0,0,0))

    
if Plot_Panel_Length_Per_Bar_and_After_g_s:
    figure_list_g_S=[]
    for i,key in enumerate(label_data_g_s.keys()):
        plot_title = key + ' panel length'
        file_name = plot_title + '.html'
        figure_list_g_S.append(plotter.single_plot(label_data_g_s[key], col_range_g_s, plot_title, file_name, index_print_name_dic_data[key], 0, 0,figure_list2[i],1))

 
###Find longest print session

# path_Fname_index_print_name_dic_dat=sInput+'\index_print_name_dic_data.pkl'
# with open(path_Fname_index_print_name_dic_dat, 'wb') as f:
#     pickle.dump(index_print_name_dic_data, f)

    
# path_Fnamelabel_data=sInput+'\label_data.pkl'
# with open(path_Fnamelabel_data, 'wb') as f:
#     pickle.dump(label_data, f)


# path_Fnamelabel_data_g_s=sInput+'\label_data_g_s.pkl'
# with open(path_Fnamelabel_data_g_s, 'wb') as f:
#     pickle.dump(label_data_g_s, f)
    

#############################################################
##################LOAD PICKLE################################
#############################################################
# path_Fname_index_print_name_dic_dat=sInput+'\index_print_name_dic_data.pkl'
# path_Fnamelabel_data=sInput+'\label_data.pkl'
# path_Fnamelabel_data_g_s=sInput+'\label_data_g_s.pkl'



# file_pathReadPKL = path_Fname_index_print_name_dic_dat
# # Load the object from the .pkl file
# with open(file_pathReadPKL, 'rb') as file:
#     index_print_name_dic_data = pickle.load(file)

# file_pathReadPKL = path_Fnamelabel_data
# # Load the object from the .pkl file
# with open(file_pathReadPKL, 'rb') as file:
#     label_data = pickle.load(file)

# file_pathReadPKL = path_Fnamelabel_data_g_s
# # Load the object from the .pkl file
# with open(file_pathReadPKL, 'rb') as file:
#     label_data_g_s = pickle.load(file)
#############################################################
#############################################################
#############################################################
#############################################################



# MaxPrint={}
# for i,key in enumerate(index_print_name_dic_data.keys()):
#     series=pd.Series(list(index_print_name_dic_data[key].keys()))
#     diff_array = series.diff().abs()
#     max_diff_index = diff_array[1:].idxmax()
#     MaxPrint[key]=[max_diff_index-1,max_diff_index,diff_array[max_diff_index]]

# StartIndex_PerBar={}
# EndIndex_PerBar={}

# for key in index_print_name_dic_data.keys():
#     StartIndex,EndIndex=create_even_odd_dict(index_print_name_dic_data[key])
#     StartIndex_PerBar[key]=StartIndex
#     EndIndex_PerBar[key]=EndIndex
    

# col_range=list(label_data[list(label_data.keys())[0]].columns)[16:28]
# MaxPrintSession={}
# for i,key in enumerate(label_data.keys()):
#     tmp=pd.DataFrame()
#     for col in col_range[2:]:
#        inx1=list(index_print_name_dic_data[key].keys())[MaxPrint[key][0]]
#        inx2=list(index_print_name_dic_data[key].keys())[MaxPrint[key][1]]

#        tmp=pd.concat([tmp,label_data[key][col][inx1:inx2].reset_index(drop=True)],axis=1).reset_index(drop=True)
#     MaxPrintSession[key]=tmp

# C2C_diff={}
# col=col_range[2:][0]
# MaxDiff={}
# for col in col_range[2:]:
#     MaxDiff[col]=[]
#     C2C_diff[col]=[]

# # 
# for col in col_range[2:]:
#     for key in MaxPrintSession.keys():
#         # key_mean=np.mean(MaxPrintSession[key][col])
#         y= savgol_filter(MaxPrintSession[key][col],100,2)

#         MaxDiff[col].append(list(MaxPrintSession[key][col]-y))
#         # plt.plot(MaxPrintSession[key][col])
#         # plt.plot(y)

# plt.figure()
# plt.plot(MaxPrintSession[key][col])
# plt.plot(y)

# for label in MaxDiff.keys():

#     for l in range(len(MaxDiff[label][0])):
#         maxVal=-1e10
#         minVal=1e10
#         for i in range(7):
#             if maxVal<MaxDiff[label][i][l]:
#                 maxVal=MaxDiff[label][i][l]
#             if minVal>MaxDiff[label][i][l]:
#                 minVal=MaxDiff[label][i][l]
#         C2C_diff[label].append(maxVal-minVal)

# C2C_pd=pd.DataFrame(C2C_diff) 

# C2C_continues=pd.DataFrame()

# for inx in C2C_pd.index:
#     C2C_continues=pd.concat([C2C_continues,pd.Series(list(C2C_pd.loc[inx]))],axis=0)


# C2C_continues=C2C_continues.reset_index(drop=True)     

# plot_title='C2C diff'
# file_name='C2C diff.html'
# titleX='Panel'
# titleY='[um]'
# key='C2C'
# figure_C2C_panel = plotter.regular_plot_C2C(C2C_continues, plot_title, file_name,
#                                            titleX, titleY,key,0,0,1)




    
###### Calc   s_golay - data_label

label_data_g_s_db = plotter.data_processor.Calc_S_Goly_diff_labelData(col_range, col_range_g_s, label_data, label_data_g_s);
col_range_g_s_dic = {value: index for index, value in enumerate(col_range_g_s)}

## creates a data base of 1st job length, 2nd start index, 3rd end index

JobLength_st_en = plotter.data_processor.Create_printLength_db(index_print_name_dic_data)

indices={}
std_of_labelLength={}
for key in JobLength_st_en.keys():
    indices[key] = [index for index,value in enumerate(JobLength_st_en[key]) if value[0] > Joblength_limit]
    
for key in label_data_g_s_db.keys():
    if len(indices[key]):
        label_data_g_s_db_col_range_g_s = label_data_g_s_db[key][col_range_g_s]
        std_of_labelLength[key]= pd.DataFrame()
        for index in indices[key]:
            std_of_labelLength[key] = pd.concat([std_of_labelLength[key],np.std(label_data_g_s_db_col_range_g_s[JobLength_st_en[key][index][1]:JobLength_st_en[key][index][2]]).rename(label_data[key]['Date'][JobLength_st_en[key][index][1]])],axis=1);
            # std_of_labelLength[key]= std_of_labelLength[key].rename(index=col_range_g_s_dic)
    else:
        continue;


if plotJob_Sgolay_std:

    for i,key in enumerate(std_of_labelLength.keys()):
        plot_title = 'All bars' + ' STD per panel (S.Golay- db)'
        file_name = plot_title + '.html'
        titleX= 'Panel #'
        titleY= 'STD [um]'
        db=std_of_labelLength[key]
     
        if i==0:
            showFig =0;
            fig= 0;
        if i==len(std_of_labelLength.keys())-1:
            showFig =1;
            
        figure_list_g_S_std=plotter.regular_plot( db, plot_title, file_name, titleX, titleY,key, fig,showFig)
        fig= figure_list_g_S_std;
        # figure_list_g_S_std.append(plotter.regular_plot( db, plot_title, file_name, titleX, titleY,key, fig,showFig))


if plotJob_Sgolay_db:
    figure_list_g_S_db=[]

    for i,key in enumerate(label_data_g_s.keys()):
        plot_title = key + ' panel length S.Golay- db'
        file_name = plot_title + '.html'
        figure_list_g_S_db.append(plotter.single_plot(label_data_g_s_db[key], col_range_g_s, plot_title, file_name, index_print_name_dic_data[key], 0, 0,0,1))




if PlotContinuesBalnkLength:
    index_print_name_dic_long_bar = {}
    for keyC in plotter.data_processor.colorList.keys():
        index_print_name_dic_long = {}
        index_print_name_dic = index_print_name_dic_data[keyC]
        for key, value in index_print_name_dic.items():
            index_print_name_dic_long[key * 10] = value
        index_print_name_dic_long_bar[keyC] = index_print_name_dic_long

    continues_panel_length = plotter.data_processor.create_continues_panel_lngth_per_bar(label_data, col_range[2:])

    plot_title = 'continues panel length'
    file_name = plot_title + '.html'
    figure_label_data = plotter.single_plot(continues_panel_length, continues_panel_length.columns, plot_title, file_name,
                                           index_print_name_dic_long_bar, 200, 1,0,1)

# plt.figure()
# plt.plot(continues_panel_length['DPSBar2'])


if PlotSTD:
    figure_std_list = []
    std_bar_panel = {}
    for key in index_print_name_dic_long_bar.keys():
        index_print_name_dic = index_print_name_dic_data[key]
        dpc_panel_bar = label_data[key]
        col_range = dpc_panel_bar.columns[16:28]
    
        std_panel = pd.DataFrame()
        st = 0
        i = 0
        for key_indx in index_print_name_dic.keys():
            en = key_indx
            std_list = []
            for col in col_range:
                std_list.append(np.std(dpc_panel_bar[col][st:en]))
            i = i + 1
            st = key_indx
            std_panel = pd.concat([std_panel, pd.DataFrame([std_list], columns=list(col_range))], ignore_index=True)
    
        std_bar_panel[key] = std_panel
    
        plot_title = key
        file_name = plot_title + ' STD per panel' + '.html'
        figure_std_list.append(
            plotter.single_plot(std_panel, col_range, plot_title + ' STD per panel', file_name, index_print_name_dic, 0,
                                1,0,1))


# if __name__ == "__main__":
#     main()
####################################################################
C2C_From_Panel_Length_Difference=C2C_From_Panel_Length_Difference(label_data,index_print_name_dic_data)


Main_Clock_bar, index_Print_C2C= C2C_From_Panel_Length_Difference.findMiddleTimeStampAndPrintStarts()

C2C_continues,C2C_continues_no_labels=C2C_From_Panel_Length_Difference.Calc_C2C_for_ongoing_printing()
dynamicthreshold=C2C_From_Panel_Length_Difference.DynamicThreshold
ContinuesPanelLngth=C2C_From_Panel_Length_Difference.ContinuesPanelLngth
unique_index_list_to_delete=C2C_From_Panel_Length_Difference.unique_index_list_to_delete


C2C_continues,C2C_continues_no_labels=C2C_From_Panel_Length_Difference.Calc_C2C_for_ongoing_printing()
ContinuesPanelLngth_noLabel= C2C_From_Panel_Length_Difference.ContinuesPanelLngth_noLabels
MaxPrintSession=C2C_From_Panel_Length_Difference.MaxPrintSession


print('dynamicthreshold='+str(dynamicthreshold))

plot_title='Paper Trail C2C Estimation by Encoder Mesurments'
file_name='Paper Trail C2C Estimation.html'
titleX='Panel'
titleY='[um]'
key='C2C'
figure_C2C_panel_WithStamp=plotter.regular_plot_C2C(C2C_continues[0], plot_title, file_name,
                                           titleX, titleY,key,index_Print_C2C,0,1)


plot_title='Paper Trail C2C Estimation by Encoder Mesurments WITH NO LABELS'
file_name='Paper Trail C2C Estimation_noLabels.html'
titleX='Panel'
titleY='[um]'
key='C2C_noLabels'
figure_C2C_panel_WithStamp=plotter.regular_plot_C2C(C2C_continues_no_labels[0], plot_title, file_name,
                                           titleX, titleY,key,index_Print_C2C,0,1)



plot_title='Paper Trail for all colors Estimation by Encoder Mesurments'
file_name='Paper Trail all colors Estimation.html'
titleX='Panel'
titleY='[um]'

figure_PerColor=plotter.regular_plot_Each_Color(ContinuesPanelLngth,  plot_title, file_name, titleX, titleY,key,index_Print_C2C  ,0,1)


plot_title='Paper Trail for all colors Estimation by Encoder Mesurments NO LABELS'
file_name='Paper Trail all colors EstimationNO LABELS.html'
titleX='Panel'
titleY='[um]'

figure_PerColor=plotter.regular_plot_Each_Color(ContinuesPanelLngth_noLabel,  plot_title, file_name, titleX, titleY,key,index_Print_C2C  ,0,1)




# BarKeys=C2C_From_Panel_Length_Difference.printSessionStartIndexesPerBar()
# printSession_sync_dic={}  

# tmp=pd.DataFrame()  

# arr=np.diff(BarKeys[list(BarKeys.keys())[0]])
 
# for C2C_From_Panel_Length_Difference.DynamicThreshold in range(5,100):
#     indices = np.where(abs(arr-C2C_From_Panel_Length_Difference.DynamicThreshold) < 3)[0]
#     if not len(indices):
#         break;

# for key in BarKeys.keys():
#     arr=np.diff(BarKeys[key])

#     # indexBorder = np.where((arr == 30 + 1) | (arr == 30 + 2) | (arr == 30))[0]
#     # arrBoder=pd.concat([arrBoder,pd.DataFrame(list(arr[indexBorder]))],axis=1).rename(columns={0: key})  
#     indices = np.where(arr>C2C_From_Panel_Length_Difference.DynamicThreshold)[0]
#     printSession_sync_dic[key]=[]
#     for inx in indices:
#         printSession_sync_dic[key].append([BarKeys[key][inx],BarKeys[key][inx+1],arr[inx]])
    
#     tmp=pd.concat([tmp,pd.DataFrame(list(arr[indices]))],axis=1).rename(columns={0: key})   

# Sync_print_length = tmp.min(axis=1)    



# Sync_print_length,printSession_sync_dic=C2C_From_Panel_Length_Difference.filterSyncPrintsBetweenBars()
# Middel_Clock_Value_bar={}
# for key in printSession_sync_dic.keys():
#     Middel_Clock_Value_bar[str(C2C_From_Panel_Length_Difference.label_data[key]['date_time'][printSession_sync_dic[key][0][0]])]=key
    
# timestamps=list(Middel_Clock_Value_bar.keys())
    
# timestamps = sorted([datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps])

# # Find the middle timestamp
# middle_timestamp_index = len(timestamps) // 2
# middle_timestamp = timestamps[middle_timestamp_index]

# Main_Clock_bar=Middel_Clock_Value_bar[str(middle_timestamp)]

# index_Print_C2C={}
# Acamulate_PrintLength=0
# i=0
# printLength=Sync_print_length[0]
# for i,printLength in enumerate(Sync_print_length):
#     Acamulate_PrintLength=Acamulate_PrintLength+printLength*10
#     index_Print_C2C[Acamulate_PrintLength]=C2C_From_Panel_Length_Difference.label_data[Main_Clock_bar]['Date'][printSession_sync_dic[Main_Clock_bar][i][0]]






















