"""
Seminario de Inteligencia Artificial 2
Proyecto Final
"""

import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import Qt

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

def prediction():
    predictData = []
    predictData.append(leSex.transform([sex.currentText()])[0])
    predictData.append(int(dvrt.text()))
    predictData.append(leEducation.transform([education.currentText()])[0])
    predictData.append(int(familiar.text()))
    predictData.append(leSchool.transform([school.currentText()])[0])
    predictData = np.asanyarray(predictData).reshape(1, -1)
    predictData = scaler.transform(np.asanyarray(predictData))
    
    if model.predict([predictData][0])[0] == 1:
        msg.setText("Si realizará el examen")
    else:
        msg.setText("No realizará el examen")

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle('ProyectoFinal-PauloCelis')
window.setGeometry(100, 100, 400, 300)
window.move(60, 15)

#Información
layout = QFormLayout()
title = QLabel('<h1>Leaving Certificate</h1>')
title.setAlignment(Qt.AlignCenter)
layout.addRow(title)

sex = QComboBox()
sex.addItem("male")
sex.addItem("female")
layout.addRow('Sexo:', sex)

dvrt = QLineEdit()
layout.addRow('DVRT:', dvrt)

education = QComboBox()
education.addItem("Junior_cycle_terminal_leaver-secondary_school")
education.addItem("Junior_cycle_terminal_leaver-vocational_school")
education.addItem("Junior_cycle_incomplete-secondary_school")
education.addItem("Junior_cycle_incomplete-vocational_school")
education.addItem("Primary_terminal_leaver")
education.addItem("Senior_cycle_terminal_leaver-secondary_school")
education.addItem("Senior_cycle_incomplete-secondary_school")
education.addItem("Senior_cycle_incomplete-vocational_school")
education.addItem("3rd_level_complete")
education.addItem("3rd_level_incomplete")
layout.addRow('Educación:', education)

familiar = QLineEdit()
layout.addRow('Score Familiar:', familiar)

school = QComboBox()
school.addItem("secondary")
school.addItem("vocational")
layout.addRow('Tipo:', school)

btn = QPushButton('Predecir')
layout.addRow(btn)
btn.clicked.connect(prediction)

msg = QLabel('')
layout.addWidget(msg)

window.setLayout(layout)

#Modelo
classifiers = { 'KNN':KNeighborsClassifier(4),
                'SVM':SVC(gamma=0.0003),
                'GP':GaussianProcessClassifier(1.5 * RBF(1.3)),
                'DT':DecisionTreeClassifier(max_depth=5),
                'MLP':MLPClassifier(alpha=.01, max_iter=1000),
                'Bayes':GaussianNB() }

data = pd.read_csv('irish.csv')

leCertificate = preprocessing.LabelEncoder()
leSex = preprocessing.LabelEncoder()
leDVRT = preprocessing.LabelEncoder()
leEducation = preprocessing.LabelEncoder()
leFamiliar = preprocessing.LabelEncoder()
leSchool = preprocessing.LabelEncoder()

target, information = data['Leaving_Certificate'].values, data.values

leCertificate.fit(target)
target = leCertificate.transform(target)

leSex.fit(information[0:,0])
information[0:,0] = leSex.transform(information[0:,0])

leEducation.fit(information[0:,2])
information[0:,2] = leEducation.transform(information[0:,2])

leSchool.fit(information[0:,5])
information[0:,5] = leSchool.transform(information[0:,5])

information = np.delete(information, np.s_[-3], axis=1)

scaler = StandardScaler().fit(information)
information = scaler.transform(information)

xTrain, xTest, yTrain, yTest = train_test_split(information, target)

model = classifiers['MLP']

model.fit(xTrain, yTrain)

print('Train: ', model.score(xTrain, yTrain))
print('Test: ', model.score(xTest, yTest))

yPred = model.predict(xTest)

print('Classification report: \n', metrics.classification_report(yTest, yPred))

print('Confusion matrix: \n', metrics.confusion_matrix(yTest, yPred))

#Mostrar ventana
window.show()
sys.exit(app.exec_())