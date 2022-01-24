from PyQt6 import QtWidgets, uic
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QMessageBox


def readData():
    age = call.lineEdit_Age.text()
    sexIndex, sex = call.comboBox_Sex.currentIndex(), call.comboBox_Sex.currentText()
    CPIndex, CP = call.comboBox_CP.currentIndex(), call.comboBox_CP.currentText()
    BP = call.lineEdit_BP.text()
    chol = call.lineEdit_Chol.text()
    FBS = call.comboBox_FBS.currentIndex()
    ECGIndex, ECG = call.comboBox_ECG.currentIndex(), call.comboBox_ECG.currentText()
    HR = call.lineEdit_HR.text()
    EIAIndex, EIA = call.comboBox_EIA.currentIndex(), call.comboBox_EIA.currentText()
    STD = call.lineEdit_STP.text()
    STSIndex, STS = call.comboBox_STS.currentIndex(), call.comboBox_STS.currentText()
    CMBVIndex, CMBV = call.comboBox_CMBV.currentIndex(), call.comboBox_CMBV.currentText()
    thalIndex, thal = call.comboBox_Thal.currentIndex(), call.comboBox_Thal.currentText()

    dfData = [age, sexIndex, CPIndex, BP, chol, FBS, ECGIndex, HR, EIAIndex, STD, STSIndex, CMBVIndex, thalIndex]
    for i in range(len(dfData)):
        if dfData[i] == -1 or dfData[i] == "":
            dfData[i] = np.nan
    createDf(dfData)

    data = [age, sex, CP, BP, chol, FBS, ECG, HR, EIA, STD, STS, CMBV, thal]
    showResults(data)



def createDf(data):
    df2 = pd.DataFrame([data], columns=['Age', 'Sex', 'Chest_pain_type', 'Resting_bp',
                                        'Cholesterol', 'Fasting_bs', 'Resting_ecg',
                                        'Max_heart_rate', 'Exercise_induced_angina',
                                        'ST_depression', 'ST_slope', 'Num_major_vessels',
                                        'Thallium_test'])
    print(df2)


def showResults(data):
    msg = QMessageBox()
    result = "placeholder result"
    # maybe use later?
    # msg.setText(f"<html><body><h3>Patient details</h3>"
    #             f"<p>Age: {data[0]}<br>"
    #             f"Sex: {data[1]}<br>"
    #             f"Chest pain type: {data[2]}<br>"
    #             f"Resting BP: {data[3]}mmHG<br>"
    #             f"Cholesterol: {data[4]}mg/dL<br>"
    #             f"Fasting blood sugar: {data[5]}mg/dL<br>"
    #             f"Resting ECG: {data[6]}<br>"
    #             f"Max HR: {data[7]}bpm<br>"
    #             f"Exercise induced angina: {data[8]}<br>"
    #             f"ST depression: {data[9]}<br>"
    #             f"ST slop: {data[10]}<br>"
    #             f"Major blood vessels: {data[11]}<br>"
    #             f"Thalium test: {data[12]}</p>"
    #             f"</body></html>")
    msg.setText(f"Result: {result}")
    msg.setWindowTitle("Results")
    msg.exec()

def clearData():
    call.lineEdit_Age.clear()
    call.comboBox_Sex.setCurrentIndex(-1)
    call.comboBox_CP.setCurrentIndex(-1)
    call.lineEdit_BP.clear()
    call.lineEdit_Chol.clear()
    call.lineEdit_FBS.clear()
    call.comboBox_ECG.setCurrentIndex(-1)
    call.lineEdit_HR.clear()
    call.comboBox_EIA.setCurrentIndex(-1)
    call.lineEdit_STP.clear()
    call.comboBox_STS.setCurrentIndex(-1)
    call.comboBox_CMBV.setCurrentIndex(-1)
    call.comboBox_Thal.setCurrentIndex(-1)


app = QtWidgets.QApplication([])
call = uic.loadUi("testUI.ui")

call.pushButton.clicked.connect(readData)
call.pushButton_2.clicked.connect(clearData)

call.show()
app.exec()
