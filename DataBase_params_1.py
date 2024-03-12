# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:40:49 2024

@author: Ireneg
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot

import plotly.graph_objects as go

##########################################
# Create a figure and axis
paramID=['cfa371b8-a281-48ae-9321-528bdc552c6d']
paramListMemory=['ChangeDirectionPageNum','SafeMarginSizeMM','MaxAllowedMovementSimplexMM','ScaleFineTune','LabelInitialShiftMM','LabelShiftStepUM','MaxAllowedMovementDuplexMM']

#QCS
paramListQCS=['SavitzkyGolay','PolynomialOrder','Radius']

##Default PrintDirection
paramList_CrossPrintDirection=['Set_0/Y','Set_1/Y1','Set_2/Y2','Set_3/Y3']
##SyntheticEncoder
paramListQCS_SyntheticEnc=['SlotId_1','SlotId_2','SlotId_3','SlotId_4','SlotId_5','SlotId_6','SlotId_7','SlotId_8']
#########################################
class CsvPickerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('CSV File Picker')
        self.setGeometry(100, 100, 400, 200)

        self.pick_button = QPushButton('Pick CSV File', self)
        self.pick_button.setGeometry(150, 80, 100, 40)
        self.pick_button.clicked.connect(self.pick_csv_file_andCreateTable)
        


    def pick_csv_file_andCreateTable(self):
        
        file_path, _ = QFileDialog.getOpenFileName(self, 'Pick a CSV File', '', 'CSV files (*.csv)')
        if file_path:
            print(f'Selected file: {file_path}')
        # file_path=r'E:\DB_gdls\D16_GDLS_Params_05032024_151946.csv'
        df = pd.read_csv(file_path)
        
        backGroundCLR='rgb(200, 200, 200)'
        HeaderCLR='rgb(0, 255, 255)'
        fillcolorList=[]

        paramList_tmp=[]
        valueList_tmp=[]
        paramList=[]
        valueList=[]
        ##Memory header
        paramList.append('Memory')
        valueList.append('')        
        fillcolorList.append(HeaderCLR)

        ##Memory
        for paramId,param, value,paramPath in zip(df['Parameter ID'],df['Parameter Name'], df['Parameter Actual Value'],df['Parameter Tree Path Name']):
           
            if paramId in paramID:

                paramList.append(param)
                valueList.append(value)
                fillcolorList.append(backGroundCLR)

                
            if param in paramListMemory:
               paramList_tmp.append(param)
               valueList_tmp.append(value)
               # fillcolorList.append(backGroundCLR)

        try:       
            # paramList,valueList=self.OrderParams(paramListMemory,paramList_tmp, valueList_tmp, paramList, valueList)
            L_added_vals,paramList,valueList=self.OrderParams(paramListMemory,paramList_tmp, valueList_tmp, paramList, valueList)
            
            for i in range(L_added_vals):
                fillcolorList.append(backGroundCLR)
            paramList.append('')
            valueList.append('')
            fillcolorList.append(backGroundCLR)
        except:
            1            
        paramList_tmp=[]
        valueList_tmp=[]
        
        ##QCS header
        paramList.append('QCS')
        valueList.append('') 
        fillcolorList.append(HeaderCLR)

        ##QCS       
        for paramId,param, value,paramPath in zip(df['Parameter ID'],df['Parameter Name'], df['Parameter Actual Value'],df['Parameter Tree Path Name']):
           
            if param in paramListQCS:
               paramList_tmp.append(param)
               valueList_tmp.append(value)
            
        try:
            # paramList,valueList=self.OrderParams(paramListQCS,paramList_tmp, valueList_tmp, paramList, valueList)
            L_added_vals,paramList,valueList=self.OrderParams(paramListQCS,paramList_tmp, valueList_tmp, paramList, valueList)
            for i in range(L_added_vals):
                fillcolorList.append(backGroundCLR)

            paramList.append('')
            valueList.append('')
            fillcolorList.append(backGroundCLR)
        except:
            1
        paramList_tmp=[]
        valueList_tmp=[] 
        
        
        ##SyntheticEncoder header
        fillcolorList.append(HeaderCLR)
        paramList.append('Default PrintDirection')
        valueList.append('')
        ##SyntheticEncoder header
        for paramId,param, value,paramPath in zip(df['Parameter ID'],df['Parameter Name'], df['Parameter Actual Value'],df['Parameter Tree Path Name']):
           
            # if param== 'Default PrintDirection':    
                # break
            
                result = self.check_list_in_string(paramList_CrossPrintDirection, paramPath)
                if result:
                    paramList.append(param+' '+result)
                    valueList.append(value)
                    fillcolorList.append(backGroundCLR)
            
      
        paramList.append('')
        valueList.append('')
        fillcolorList.append(backGroundCLR)
     
        paramList_tmp=[]
        valueList_tmp=[] 
        
        ##SyntheticEncoder header
        fillcolorList.append(HeaderCLR)
        paramList.append('SyntheticEncoder')
        valueList.append('')
        ##SyntheticEncoder header
        for paramId,param, value,paramPath in zip(df['Parameter ID'],df['Parameter Name'], df['Parameter Actual Value'],df['Parameter Tree Path Name']):
           
            if param== 'SyntheticEncoder':    
                # break
                # paramList.append(param+' '+self.check_list_in_string(paramList_CrossPrintDirection, paramPath))
                result = self.check_list_in_string(paramListQCS_SyntheticEnc, paramPath)

                if result:
                    paramList.append(param+' '+result)
                    valueList.append(value)
                    fillcolorList.append(backGroundCLR)
                else:
                    paramList.append(param)
                    valueList.append(value)
                    fillcolorList.append(backGroundCLR)
            
        ListofList=[]
        ListofList.append(paramList)
        ListofList.append(valueList)
        fillcolorListofLists=[]
        fillcolorListofLists.append(fillcolorList)
        fillcolorListofLists.append(fillcolorList)

        
        
        table = go.Figure(data=[go.Table(cells=dict(values=ListofList,fill_color=fillcolorListofLists,font=dict(color='black', size=15),align='left',height=25))])
        
        
        table.update_layout(title=file_path.split('/')[-1], autosize=False,
                     width=1000, height=2000, 
                     margin=dict(l=150, r=10, t=50, b=10),
                     font=dict(size=16),  # Adjust font size
                     )
        # table.update_layout(title='Table with Black Row')
        table.show()


        plot(table, auto_play=True, filename=file_path.split('/')[-1]+'.html')

        # paramList.append('')
        # valueList.append('')
        # fillcolorList.append(backGroundCLR)
     
        # paramList_tmp=[]
        # valueList_tmp=[] 
     


    # def pick_csv_file_andCreateTable(self):
    #     file_path, _ = QFileDialog.getOpenFileName(self, 'Pick a CSV File', '', 'CSV files (*.csv)')
    #     if file_path:
    #         print(f'Selected file: {file_path}')
        
    #     df = pd.read_csv(file_path)


    #     fig, ax = plt.subplots()

    #    # Create a table within the plot
    #     table_data = []
    #     for paramId,param, value,paramPath in zip(df['Parameter ID'],df['Parameter Name'], df['Parameter Actual Value'],df['Parameter Tree Path Name']):
    #        if param in paramList:
    #            table_data.append([param, value])
    #        if paramId in paramID:
    #            table_data.append([param, value])

    #     table = ax.table(cellText=table_data, colLabels=['Parameter', 'Value'], loc='center', cellLoc='left', fontsize=20)
    #    # Set font size for the table
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(8)
    #     fig.suptitle(file_path.split('/')[-1])

    #     # table.set_title(file_path.split('\\')[-1])

    #    # Hide axes
    #     ax.axis('off')

    #    # Display the plot
    #     plt.show() 

    def closeEvent(self, event):
        QApplication.quit()

    def OrderParams(self,TargetList,paramList_tmp,valueList_tmp,paramList,valueList):
        # TargetList=paramListQCS
        index_list=[]
        for val in TargetList:
            try:
                index_list.append(paramList_tmp.index(val) )
            except:
                continue
        
        for inx in index_list:
            
            paramList.append(paramList_tmp[inx])
            valueList.append(valueList_tmp[inx]) 
            
        return len(index_list),paramList,valueList
    
    
    def check_list_in_string(self,lst, string):
        result = next((item for item in lst if item in string), None)
    
        return result



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CsvPickerWindow()
    window.show()
    sys.exit(app.exec_())

