##################################### RANDOM FOREST ALGO ##########################################


######################################## Load Libraries ###########################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, cross_validation, svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')
from Universities import Uni

##################################### Read Data ###################################################

#binary = pd.read_excel("D:/University Selection System/Datasets/Fall 2014 Results.xlsx")
test_data = pd.read_csv('D:/University Selection System/Datasets/TestFor1690.csv')

################################## Data Manipulations #############################################
# binary.dropna(inplace=True)
# print (binary.isnull().any())

test_data.dropna(inplace=True)
print (test_data.isnull().any())
 # print the column names
# print ("TrainData Columns:\n",binary.columns)
# print ("TestData Columns:\n",test_data.columns)

cleanup_nums = {"Result": {"Accept": 1, "Reject": 0}}
test_data.replace(cleanup_nums, inplace=True)
#print(Uni)
for i in Uni:
    if(Uni[i] == 1):
        continue
    x=pd.read_excel("{0}.xlsx".format(i))
    x.replace(cleanup_nums, inplace=True)
    # print(x.isnull().any())
    print("------------------Executing: {0}--------------------------".format(i.upper()))
    x.drop(['Major'], 1, inplace=True)
    x.drop(['Scale'], 1, inplace=True)
    x.drop(['IELTS'], 1, inplace=True)
    x.drop(['Undergrad Univesity'], 1, inplace=True)
    x.drop(['Under Graduate Aggregate'], 1, inplace=True)
    x.drop(['Name'], 1, inplace=True)

    x.dropna(inplace=True)
    #print(x.isnull().any())

    Train_IndepentVars  = x.values[:,2:]
    Train_TargetVars = x.values[:, 1]
    Train_TargetVars=Train_TargetVars.astype('int')
    Train_IndepentVars =Train_IndepentVars.astype('int')

    #print ("-------------Train_IndependentVars of {0}:---------------\n".format(i.upper()),Train_IndepentVars )
    #print ("-------------Train_TargetVars of {0}:--------------------\n".format(i.upper()),Train_TargetVars )

    Test_TargetVar = test_data.values[:, 1:8]
    Estimated = test_data.values[:, 8]
    print ("--------------Test Values:---------------\n",Test_TargetVar)

    ############################## Training and Testing samples #########################################
    ######################################## Random Forest  #############################################

    rf_model = RandomForestClassifier()
    rf_model.fit(Train_IndepentVars, Train_TargetVars)

    # Scoring based on the train RF Model
    predictions_rf = rf_model.predict(Test_TargetVar)
    #predictions_rf=('Accept'if(predictions_rf == 1).all()else('Reject'))
    print ("Random Forest Prediction for {0}:\n".format(i.upper()),predictions_rf)
    print("Estimated Output:\n",Estimated)

    ####################################### Importance ##################################################

    importance_rf = rf_model.feature_importances_
    c=importance_rf*100
    importance_rf = pd.DataFrame(c, index=x.columns[2:],
                                 columns=["Importance"])

    print(importance_rf)

    Accuracy_Score_rf=accuracy_score(Estimated, predictions_rf)
    print("Accuracy Score of Random Forest for {0}:".format(i.upper()),(Accuracy_Score_rf)*100,"%")
    if (i == 'Arizona State University'):
        ################################# Using Gaussian Niive Bayes ########################################
        ######################################################################################################
        print("Gaussian Naiive Bayes:\n")
        model = GaussianNB()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(Train_IndepentVars, Train_TargetVars, test_size=0.2)

        ##################################### Train the classifier: #########################################
        #print(X_train,y_train)
        #model.fit(x, y)
        #model.fit(x,y)
        model.fit(X_train,y_train)

        ########################################### Test: ###################################################

        predicted_gnb= model.predict(Test_TargetVar)
        print ("Gaussian Naiive Bayes Prediction for {0}:\n".format(i.upper()),predicted_gnb)
        print("Estimated Output:\n",Estimated)

        Accuracy_Score_gnb=accuracy_score(Estimated, predicted_gnb)
        print("Accuracy Score of Gaussian Naiive Bayes for {0}:".format(i.upper()),(Accuracy_Score_gnb)*100,"%")

    if (i == 'Arizona State University'):
        break
