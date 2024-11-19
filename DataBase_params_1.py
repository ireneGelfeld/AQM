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
import subprocess 
import plotly.graph_objects as go

##########################################
# Create a figure and axis
paramID=['cfa371b8-a281-48ae-9321-528bdc552c6d']
paramListMemory=['ChangeDirectionPageNum','SafeMarginSizeMM','MaxAllowedMovementSimplexMM','ScaleFineTune','LabelInitialShiftMM','LabelShiftStepUM','MaxAllowedMovementDuplexMM']

#CIS
paramCIS=['CISCurvaturePerPixel']

#QCS
paramListQCS=['SavitzkyGolay','PolynomialOrder','Radius']

##Default PrintDirection
paramList_CrossPrintDirection=['Set_0/Y','Set_1/Y','Set_2/Y','Set_3/Y','Set_1/Y1','Set_2/Y2','Set_3/Y3']
##SyntheticEncoder
paramListQCS_SyntheticEnc=['SlotId_2','SlotId_3','SlotId_4','SlotId_5','SlotId_6','SlotId_7','SlotId_8']
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
        self.df = pd.read_csv(file_path)
        os.chdir(os.path.dirname(file_path))
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
        for paramId,param, value,paramPath in zip(self.df['Parameter ID'],self.df['Parameter Name'], self.df['Parameter Actual Value'],self.df['Parameter Tree Path Name']):
           
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
        for paramId,param, value,paramPath in zip(self.df['Parameter ID'],self.df['Parameter Name'], self.df['Parameter Actual Value'],self.df['Parameter Tree Path Name']):
           
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
        BackFront=''
        RightLeft=''
        for paramId,param, value,paramPath in zip(self.df['Parameter ID'],self.df['Parameter Name'], self.df['Parameter Actual Value'],self.df['Parameter Tree Path Name']):
           
            # if param== 'Default PrintDirection':    
                # break
            
                result = self.check_list_in_string(paramList_CrossPrintDirection, paramPath)
                if result:
                    if 'Front' in paramPath:
                        BackFront='Front'
                    elif 'Back' in paramPath:
                        BackFront='Back'
                    else:
                        BackFront=''
                    if 'Left' in paramPath:
                        RightLeft='Left'
                    elif 'Right' in paramPath:
                        RightLeft='Right'
                    else:
                        RightLeft=''
                    paramList.append(param+' '+result+' '+BackFront+' '+RightLeft)
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
        for paramId,param, value,paramPath in zip(self.df['Parameter ID'],self.df['Parameter Name'], self.df['Parameter Actual Value'],self.df['Parameter Tree Path Name']):
           
            if param== 'SyntheticEncoder':    
                # break
                # paramList.append(param+' '+self.check_list_in_string(paramList_CrossPrintDirection, paramPath))
                result = self.check_list_in_string(paramListQCS_SyntheticEnc, paramPath)

                if result:
                    paramList.append(param+' '+result)
                    valueList.append(value)
                    fillcolorList.append(backGroundCLR)
                # else:
                #     paramList.append(param)
                #     valueList.append(value)
                #     fillcolorList.append(backGroundCLR)
            
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


        plot(table, auto_play=True, filename=file_path.split('/')[-1][:-4]+'.html')
        
        # url = 'file:///' + os.path.dirname(file_path)+'//' + file_path.split('/')[-1][:-4]+'.html'
        # subprocess.Popen(['start', 'chrome', url], shell=True)

        # CIS
        result = self.df.loc[ self.df['Parameter Name'] == paramCIS[0], 'Parameter Actual Value']
        sideFRONT_BACK= self.df.loc[ self.df['Parameter Name'] == paramCIS[0], 'Parameter Tree Path Name']

        try:
                if len(result.index)>1:
                    float_list1 = [float(num) for num in result.iloc[0].split(",")]
                    if  'Back' in sideFRONT_BACK.iloc[0]:
                        side1='Back'
                    else: 
                        side1='Front'
                    float_list2 = [float(num) for num in result.iloc[1].split(",")]
                    if 'Back' in sideFRONT_BACK.iloc[1]:
                        side2='Back'
                    else: 
                        side2='Front'
                else:
                    float_list1 = [float(num) for num in result.iloc[0].split(",")]
 
                    
        
        
                fig1 = go.Figure()
        
                          
                fig1.add_trace(go.Scatter(y=float_list1,line=dict(color='blue') , name='CIS curve'))
                    
        
                 
        
                 
        
                 
                fig1.update_layout(title={
                     'text': side1,
                     'font': {'color': 'black'}
                 })
                  #fig_back.update_layout(title='ImagePlacement_Left-Back')
                  
                  
                fig1.update_layout(
                      hoverlabel=dict(
                          namelength=-1
                      )
                  )
                  
                  # datetime object containing current date and time
        
                plot(fig1,auto_play=True,filename=file_path.split('/')[-1][:-4]+'_CIS_'+side1+'.html')  
                      # plot(fig)  
        
                fig1.show()
                
                fig2 = go.Figure()
        
                          
                fig2.add_trace(go.Scatter(y=float_list2,line=dict(color='red') , name='CIS curve'))
                    
        
                 
        
                 
        
                 
                fig2.update_layout(title={
                     'text': side2,
                     'font': {'color': 'black'}
                 })
                  #fig_back.update_layout(title='ImagePlacement_Left-Back')
                  
                  
                fig2.update_layout(
                      hoverlabel=dict(
                          namelength=-1
                      )
                  )
                  
                  # datetime object containing current date and time
        
                plot(fig2,auto_play=True,filename=file_path.split('/')[-1][:-4]+'_CIS_'+side2+'.html')  
                      # plot(fig)  
        
                fig2.show() 
        except:
                1
        
        
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
    window.pick_csv_file_andCreateTable()
    
    window.show()    
    # df=window.df
    sys.exit(app.exec_())



