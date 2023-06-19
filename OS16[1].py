#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 21:19:31 2021

@author: mac
"""
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from matplotlib import pyplot as plt



BD_A = pd.read_csv ('X_app_58_3.csv', sep=';')
data_application = BD_A.values
data_application_red = data_application[0:6000,:]



BD_B = pd.read_csv ('X_test_58_3.csv', sep=';')
data_testing = BD_B.values


db_label = pd.read_csv ('Label_app_58_3.csv', sep=';')
label_app = db_label.values
label_app_t = np.ravel(label_app)
label_app_t_red = label_app_t[0:6000]


db_label = pd.read_csv ('Label_test_58_3.csv', sep=';')
label_test = db_label.values
label_test_t = np.ravel(label_test)

X=  data_application_red[1:6000][246]

X = np.reshape(X,(28,28),order='C')
plt.figure()
plt.matshow(X)
plt.title('Figure : Affichage de image')


# standardiser les données

std_scale = preprocessing.StandardScaler().fit(data_application_red)


data_application_std = std_scale.transform(data_application_red)
data_test_std = std_scale.transform(data_testing)


#la premiere  methode avec un noyau lineaire 
# on crée une SVM avec un noyau lineaire 

#classifier = svm.SVC( kernel='linear') 
#la deuxieme methode avec un noyau non-lineaire rbf 
#classifier = svm.SVC(C=11 ,kernel='rbf', gamma=4) 
#la troisieme  methode avec un noyau  polynomial
classifier = svm.SVC(kernel='poly',gamma=0.0001,C=10)
# Entraîner la SVM sur le jeu d'entraînement
classifier.fit(data_application_std ,label_app_t_red)

# prédire sur le jeu de test
y_app_pred = classifier.decision_function(data_application_std)
lab_app_pred = np.sign(y_app_pred)
y_test_pred = classifier.decision_function(data_test_std)
lab_test_pred = np.sign(y_test_pred)

eval_app=lab_app_pred-label_app_t_red
eval_test=lab_test_pred-label_test_t





# la construction  de notre courbe ROC qui precise la perferformance de VSM

fpr, tpr, thr = metrics.roc_curve(label_test_t, lab_test_pred)

# le calcule de l'aire AUC de notre courbe ROC
AUC = metrics.auc(fpr, tpr)

# la creation de notre figure

Figure = plt.figure(figsize=(6, 6))

# l'affichage de la courbe ROC ainsi que l'evaluation de AUC

plt.plot(fpr, tpr, '-', lw=2, label='AUC=%.2f' % AUC)

# le titre des axes et le graphe
plt.xlabel('Taux de faux Positif', fontsize=16)
plt.ylabel('Taux de vrai Positif', fontsize=16)
plt.title('SVM Courbe COR', fontsize=16)

# affichage de la légende
plt.legend(loc="lower right", fontsize=14)

# affichage de courbe COR
plt.show()



