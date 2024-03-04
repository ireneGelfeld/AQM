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

##########################################
paramList=['ChangeDirectionPageNum','SafeMarginSizeMM','MaxAllowedMovementSimplexMM','ScaleFineTune','LabelInitialShiftMM','LabelShiftStepUM','MaxAllowedMovementDuplexMM']
# Create a figure and axis
paramID=['cfa371b8-a281-48ae-9321-528bdc552c6d']


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
        
        df = pd.read_csv(file_path)


        fig, ax = plt.subplots()

       # Create a table within the plot
        table_data = []
        for paramId,param, value,paramPath in zip(df['Parameter ID'],df['Parameter Name'], df['Parameter Actual Value'],df['Parameter Tree Path Name']):
           if param in paramList:
               table_data.append([param, value])
           if paramId in paramID:
               table_data.append([param, value])

        table = ax.table(cellText=table_data, colLabels=['Parameter', 'Value'], loc='center', cellLoc='left', fontsize=20)
       # Set font size for the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        fig.suptitle(file_path.split('/')[-1])

        # table.set_title(file_path.split('\\')[-1])

       # Hide axes
        ax.axis('off')

       # Display the plot
        plt.show() 

    def closeEvent(self, event):
        QApplication.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CsvPickerWindow()
    window.show()
    sys.exit(app.exec_())

