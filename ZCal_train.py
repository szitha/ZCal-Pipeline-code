#! /usr/bin/python

from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
import random 
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import expon as ex 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import csv
import time as t
import optparse


"""ZCal_train is a supervised machine learning model which uses ensemble methods 
   to predict calibration solutions by learning from telescope
   sensor data together with calibration solutions as targets.
"""


"""A decision tree regressor predicts an output by splitting the population or sample into two
or more homogeneous sets (or sub-populations) based on certain rules.
""" 
def Decision_tree(Xtrain, Ytrain, Xtest):
    
    tuned_parameters = {'splitter': ['best','random']
                        ,"max_features":["log2","sqrt"]
                        ,'min_samples_split':np.arange(30,60,5)
                        ,'min_samples_leaf':np.arange(7,14)
                        ,'max_depth':np.arange(700,1389,10)}
    
    """Randomized optimizationSearch which used cross validation to optimized best parameters for the estimator. 
    In contrast to GridSearchCV, not all parameter values are tried out, 
    but rather a fixed number of parameter settings is sampled from the specified distributions.
    The number of parameter settings that are tried is given by n_iter.
    """
    Multreg = RandomizedSearchCV(DecisionTreeRegressor(random_state = 0)
                               ,param_distributions=tuned_parameters
                               ,cv = 10
                               ,n_iter = int(args[1])
                               ,n_jobs = -1
                               ,random_state = 0) 
    
    #Fitting decision tree model
    Multreg.fit(Xtrain, Ytrain) 
    #Predicting with unseen testing set
    YMultreg = Multreg.predict(Xtest) 
    # save the model to disk
    filename = 'finalized_DC.sav'
    pickle.dump(Multreg, open(filename, 'wb'))
    return YMultreg


"""A random forest is a forest of Decision tree models which predicts an
observation by averaging the output of each tree"""

def Random_forest(Xtrain, Ytrain, Xtest):
    grid = {"n_estimators":np.arange(100, 1200, 50)
          ,"max_features":["log2", "sqrt","auto"]
          ,"max_depth":np.arange(20, 200, 10)
          ,"min_samples_leaf":np.arange(3, 50, 5)}

    #Randomized search parameter optimization 
    RF = RandomizedSearchCV(RandomForestRegressor(random_state = 0, oob_score = 0)
                          ,param_distributions = grid
                          ,cv = 15, n_iter=int(args[1])
                          ,n_jobs=-1, random_state = 0)
    RF.fit(Xtrain, Ytrain)
    #Predicting using unseen data
    RF_predict = RF.predict(Xtest)  
    # save the model to disk
    filename = 'finalized_RF.sav'
    pickle.dump(RF, open(filename, 'wb'))
    return RF_predict

"""K nearest neighbors is a simple algorithm that stores all available 
cases and predict the numerical target based on a similarity measure (e.g., distance functions).
KNN regression predict the output by  calculating the average of the numerical target of the K nearest neighbors"""
def K_NN(Xtrain, Ytrain, Xtest):
    
    KNNoptparam = {"n_neighbors":np.arange(20, 200, 10)
                 ,"weights": ['uniform','distance']
                 ,"algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
                 #,"leaf_size":np.arange(30,150,15)
                 ,"p":[2,3]}

    #Randomized search parameter optimization 
    RF1=RandomizedSearchCV(KNeighborsRegressor()
                           ,param_distributions = KNNoptparam
                           ,cv = 10
                           ,n_iter = int(args[1])
                           ,n_jobs = -1
                           ,random_state = 0)
     
    RF1.fit(Xtrain, Ytrain)
    #Predicting using unseen data
    KNN_predict = RF1.predict(Xtest)
    # save the model to disk
    filename = 'finalized_KNN.sav'
    pickle.dump(RF1, open(filename, 'wb'))
    return KNN_predict

"""Extremly randomized trees (Extra-tree) This class implements a meta estimator that fits a number
of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset by selecting best splitting 
and use averaging to improve the predictive accuracy and control over-fitting"""
def EXT_tree(Xtrain, Ytrain, Xtest):
    grid2={"n_estimators":np.arange(100, 1200, 50)
          ,"max_features":["log2", "sqrt", "auto"]
          ,"max_depth":np.arange(20,200,10)
          ,"min_samples_leaf":np.arange(3, 50, 5)}
 
    RF3=RandomizedSearchCV(ExtraTreesRegressor(random_state = 0,oob_score = 0)
                           ,param_distributions = grid2
                           ,cv = 15
                           ,n_iter = int(args[1])
                           ,n_jobs = -1
                           ,random_state = 0)
    #MOdel fitting 
    RF3.fit(Xtrain, Ytrain) 
     #Predicting using unseen data
    EXT_predict = RF3.predict(Xtest)
    # save the model to disk
    filename = 'finalized_EXT.sav'
    pickle.dump(RF3, open(filename, 'wb'))
    return EXT_predict


parser = optparse.OptionParser(usage='%prog [options]',
                               description='ZCal pipeline for predicting gain solutions')

(opts,args) = parser.parse_args()

if len(args)<1:
    raise RuntimeError('Please specify the training data and number of iterations')
else:
    
    """Below we load the original constructed data and normalize before splitting"""
    
    #Loading matrix data from .csv matrix
    data = np.loadtxt(args[0],delimiter = ',',skiprows = 1)

    #Attaching index column to dada
    IDX = np.arange(1,data.shape[0]+1) 
    IDX1 = np.matrix(IDX).transpose()
    NewData = np.column_stack((IDX1,data))

    Myscale = StandardScaler()
    """Normalizing sensor data using standard scalar
    Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using the
    'transform' method.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual feature do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).
    """

    """Computing mean and standard deviation for later scaling"""
    Myscale.fit(NewData[:,1:48])
    Mean = Myscale.mean_
    STD = Myscale.std_
    NewData[:,1:48] = Myscale.fit_transform(NewData[:,1:48]) 


    """Splitting Training and testing set using a 80% and 20 % strategy""" 
    X_Train, X_Test, Y_train, Y_test = train_test_split(NewData[:,:48],NewData[:,48:],test_size=0.2) 
    #Storing Index of testing set
    idtest = X_Test[:,0]
    #Storing index  of training set
    idtrain = X_Train[:,0] 

    """Hacky Cheat, assigning polarizations where x2 represents HH and x3  VV"""
    x = range(np.shape(Y_train)[1])
    x2 = x[:-14] #HH
    x3 = x[-14:] #VV 
    """HH phase,where odd is an index position for phase and even is index position for
    amplitude for both HH and VV."""
    odd = x2[1::2] 
    even = x2[::2]  #HH amplitude
    odd1 = x3[1::2] #VV phase
    even1 = x3[::2] #VV Amplitude

    """Calling functions which returns finalised model and perdictions"""
    YMultreg = Decision_tree(X_Train[:,1:], Y_train, X_Test[:,1:])
    RandomForest = Random_forest(X_Train[:,1:], Y_train, X_Test[:,1:])
    EXT = EXT_tree(X_Train[:,1:], Y_train, X_Test[:,1:])
    KNN = K_NN(X_Train[:,1:], Y_train, X_Test[:,1:])
    AVG = (YMultreg+RandomForest+EXT+KNN)/4

    """Defining a list to loop through the models for plotting,
       note that this is to store the
       predicted results once, to svoid model from training multiple
       times when called like this in different line
    """

    Dict = {"Decision-Tree":YMultreg,"Random-Forest":RandomForest,"Extra-tree":EXT,"KNN":KNN,"AVG-model":AVG}
    rmse,Header,r2Score = [],[],[]    
    for Model_name,Model_pred in Dict.iteritems():
        #Calculate and save the roo-mean squared error for each model and polarization 
        rmse.append((np.sqrt(mean_squared_error(Y_test[:,x2], Model_pred[:, x2])),
                         np.sqrt(mean_squared_error(Y_test[:,x3], Model_pred[:,x3]))))
        r2Score.append(r2_score(Y_test[:,x2],Model_pred[:,x2], multioutput='variance_weighted'))
        Header.append(Model_name+'H-V')
        
        #Plot H-polarization for both amplitude and phase
        plt.figure(figsize = (15,15))
        plt.subplots_adjust(hspace = 0.5)
        for i in range(len(odd)):
            plt.subplot(4, 2, i+1)
            plt.scatter(np.ravel(Y_test[:,odd[i]]), Model_pred[:, odd[i]],c='r', marker='o', s=45)
            #Fitting a line betweeen the true and predicted
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.ravel(Y_test[:,odd[i]])
                                                              ,Model_pred[:, odd[i]])
            line = slope*np.ravel(Y_test[:,odd[i]])+intercept
            plt.plot(np.ravel(Y_test[:,odd[i]]),line,'k--',lw=1.3)
            plt.ylabel('$Y_{predicted(\phi)}[Rad]$',fontsize=16)
            plt.xlabel('$Y_{true(\phi)}[Rad]$',fontsize=16)
            plt.grid()
            plt.title(Model_name+'-'+'Ant'+str(i+1)+'H'+'($\phi$)',fontsize=20,fontweight='bold')
            plt.savefig(Model_name+'Hphase.eps')

        plt.figure(figsize = (15,15))
        plt.subplots_adjust(hspace = 0.5)
        for i in range(len(even)):
            plt.subplot(4,2,i+1)
            plt.scatter(np.ravel(Y_test[:,even[i]]),Model_pred[:, even[i]],c = 'g',marker = 'o',s=45)
            #Fitting a line betweeen the true and predicted
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.ravel(Y_test[:,even[i]])
                                                              ,Model_pred[:, even[i]])
            line = slope*np.ravel(Y_test[:,even[i]])+intercept
            plt.plot(np.ravel(Y_test[:,even[i]]),line,'k--',lw=1.3)
            plt.ylabel('$Y_{predicted(A)}$',fontsize = 20)
            plt.xlabel('$Y_{true(A)}$',fontsize=20)
            plt.grid()
            plt.title(Model_name+'-'+'Ant'+str(i+1)+'H'+'($Amp$)',fontsize = 20,fontweight = 'bold')
            plt.savefig(Model_name+'Hamp.eps')
        
        #Plot V-polarization for both amplitude and phase
        plt.figure(figsize = (15,15))
        plt.subplots_adjust(hspace = 0.5)
        for i in range(len(odd1)):
            plt.subplot(4,2,i+1)
            plt.scatter(np.ravel(Y_test[:,odd1[i]]),Model_pred[:, odd1[i]],c='r',marker='o',s=45)
            #Fitting a line betweeen the true and predicted
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.ravel(Y_test[:,odd1[i]])
                                                              ,Model_pred[:, odd1[i]])
            line = slope*np.ravel(Y_test[:,odd1[i]])+intercept
            plt.plot(np.ravel(Y_test[:,odd1[i]]),line,'k--',lw=1.3)
            plt.ylabel('$Y_{predicted(\phi)}[Rad]$',fontsize=16)
            plt.xlabel('$Y_{true(\phi)}[Rad]$',fontsize=16)
            plt.grid()
            plt.title(Model_name+'-'+'Ant'+str(i+1)+'V'+'($\phi$)',fontsize=20,fontweight='bold')
            plt.savefig(Model_name+'Vphase.eps')

        plt.figure(figsize = (15,15))
        plt.subplots_adjust(hspace = 0.5)
        for i in range(len(even1)):
            plt.subplot(4,2,i+1)
            plt.scatter(np.ravel(Y_test[:,even1[i]]),Model_pred[:, even1[i]],c='g',marker='o',s=45)
            #Fitting a line betweeen the true and predicted
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.ravel(Y_test[:,even1[i]])
                                                              ,Model_pred[:, even1[i]])
            line = slope*np.ravel(Y_test[:,even1[i]])+intercept
            plt.plot(np.ravel(Y_test[:,even1[i]]),line,'k--',lw=1.3)
            plt.ylabel('$Y_{predicted(A)}$',fontsize=20)
            plt.xlabel('$Y_{true(A)}$',fontsize=20)
            plt.grid()
            plt.title(Model_name+'-'+'Ant'+str(i+1)+'V'+'($Amp$)',fontsize=20,fontweight='bold')
            plt.savefig(Model_name+'Vamp.eps')


        #NOTE: to be removed later 
        print ("Statistical results for %s"%(Model_name))
        print ("Average explained variance is: ", (explained_variance_score(Y_test[:,x2],Model_pred[:,x2])) )
        print ("Explained variance (Amp) :  ", (explained_variance_score(Y_test[:,x2],Model_pred[:,x2] ,
                                                                         multioutput='raw_values')[::2]))
        print ("Explained variance (Phase) :  ", (explained_variance_score(Y_test[:,x2],Model_pred[:,x2] ,
                                                                           multioutput='raw_values')[1::2]))
        print ("Average  mean_squared_error is :  ", (mean_squared_error(Y_test[:,x2],Model_pred[:,x2])) )
        print (" mean_squared_error is (Amp):  ", (mean_absolute_error(Y_test[:,x2],Model_pred[:,x2] ,
                                                                       multioutput='raw_values')[::2]) )
        print (" mean_squared_error is (Phase):  ", (mean_absolute_error(Y_test[:,x2],Model_pred[:,x2] ,
                                                                         multioutput='raw_values')[1::2]) )
        print ("Average  mean_absolute_error : " ,(mean_absolute_error(Y_test[:,x2],Model_pred[:,x2])) )
        print ("mean_absolute_error (Amp):  ", (mean_absolute_error(Y_test[:,x2],Model_pred[:,x2] ,multioutput='raw_values')[::2]))
        print ("mean_absolute_error (Phase):  ", (mean_absolute_error(Y_test[:,x2],Model_pred[:,x2] ,multioutput='raw_values')[1::2]))
        print ("Average  r2score_error is :  ",(r2_score(Y_test[:,x2],Model_pred[:,x2],multioutput='variance_weighted'))) 
        print ("mean_r2score_error (Amp):  " ,(r2_score(Y_test[:,x2],Model_pred[:,x2] ,multioutput='raw_values')[::2]))
        print ("mean_r2score_error (Phase): ", (r2_score(Y_test[:,x2],Model_pred[:,x2],multioutput='raw_values')[1::2]))
        print ("--------------------------------------------------------------------------------------------------------\n")

    # Write mse to file
    try:
        with open('RMS.csv') as f:
            f.close()
            f = open('RMS.csv','a')
            c = csv.writer(f, lineterminator='\n')
            c.writerow(rmse)
            c.writerow(r2Score)# Append values without header
            f.close()
    except IOError:
        f = open('RMS.csv', 'a') # Create file for the first time
        w = csv.writer(f, lineterminator='\n')
        w.writerow(Header)
        w.writerow(rmse)
        w.writerow(r2Score)
        f.close()

    # Write mse to file
    Header2=['Mean']
    try:
        with open('scale.csv') as f:
            f.close()
            f = open('scale.csv','a')
            c = csv.writer(f, lineterminator='\n')
            c.writerow(Mean)
            c.writerow(STD)
            f.close()
    except IOError:
        f = open('scale.csv', 'a') # Create file for the first time
        w = csv.writer(f, lineterminator='\n')
        w.writerow(Header2)
        w.writerow(Mean)
        w.writerow(STD)
        f.close()

