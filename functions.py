"""
This file contains the functions for a Binary Classification Project. The functions are used to implement a workflow working on : Chronic Kidney Dataset & Bank Authentication Dataset.

@Author : Hamid Massaoud & Hatim El Malki 
"""


#Importing the needed libraries

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import os
import seaborn as sns



#Import the dataset:
def load_data(dataset): 
    """
    Loading the data using Pandas and os.path. This module implements some useful functions on pathnames
    allowing us to get access to the filesystem and download the data
    
    @Author : Hatim EL MALKI
    """
    missing_values=["?", "\t?"]
    csv_path = os.path.join(os.getcwd(), dataset) 
    return pd.read_csv(csv_path, na_values=missing_values)


#Data Preprocessing:
def preprocessing(df):
    """
    Computing the average value on the features and use it to fill the missing values in our DataSet.
    Normalization of the DataSet by using Min-max scaling : 
        -> Values are shifted and rescaled so that they end up ranging from 0 to 1. 
        -> We do this by subtracting the min value and dividing by the max minus the min. 
        
    @Author : Hatim EL MALKI
    """
    cat_col = df.select_dtypes(include=['object']).columns # get categorical columns 
    num_col = [x for x in df.columns if x not in cat_col] # get the numerical columns 
    label_col = df.columns[-1] # get the labels column 

    # Min-Max Normalization of the DataSet
    for x in num_col:
        mean = df[x].mean() # average of x column 
        df[x]=df[x].fillna(mean) # replace the missing values by average  
        minimum = df[x].min() # get the minimum of x column 
        maximum = df[x].max() # get the maximum of x column 
        
        df[x]=(df[x]-minimum)/(maximum-minimum) # Apply the min-max normalization on x column 
        
    # Remove Blanks from the labels Column     
    for y in cat_col :
        df[y]=df[y].str.strip()
    
    # Encode Categorical Data
    le = LabelEncoder() 
    le.fit(df[label_col]) # fit the labelEncoder
    label = le.transform(df[label_col]) # Encode the labels column 
    df = df.drop([label_col], axis = 1) # Drop the categorical label column
    new_df = pd.get_dummies(df) # Convert categorical variable except the labels 
    new_df[label_col] = label # Add the encoded labels column 
    
    
    return new_df



#Data Visualization:
def visualize_data(df):
    """
    Visualizing our DataSet by plotting a simple scatter of 2 different features one plotted along the x-axis 
    and the other plotted along the y-axis
    
    @Author : Hamid MASSAOUD
    """
    num_col = df.select_dtypes(include=['float64']).columns # get Numerical columns 
    if 'id' in num_col : 
        df = df.drop(['id'], axis='columns') 
    fig, axes = plt.subplots(nrows=int(len(num_col)/2), ncols=int(len(num_col)/2), figsize=(20,10))
    fig.tight_layout()

    plots = [(i, j) for i in range(len(num_col)) for j in range(len(num_col)) if i!=j]
    colors = ['g', 'y']
    labels = ['0', '1']

    for i, ax in enumerate(axes.flat):
        for j in range(2):
            x = df.columns[plots[i][0]]
            y = df.columns[plots[i][1]]
            ax.scatter(df[df[df.columns[-1]]==j][x], df[df[df.columns[-1]]==j][y], color=colors[j])
            ax.set(xlabel=x, ylabel=y)

    fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
    fig.tight_layout()
    plt.show()



#Getting insight about features correlation
def data_correlation(df):
    """
    Compare the correlation between features and remove one of two features 
    that have a correlation higher than 0.9
    
    @Author : Hamid MASSAOUD
    """
    corr_matrix = df.corr()
    plt.figure(figsize=(20,10))
    sns.heatmap(corr_matrix,annot=True, cmap="YlGnBu")
    plt.title('Correlation heatmap for the DataSet')
    plt.show()
    
    columns = np.full((corr_matrix.shape[0],), True, dtype=bool)
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[0]):
            if abs(corr_matrix.iloc[i,j]) >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = df.columns[columns]
    
    



#Splitting the data and applying PCA
def split_data(df, pca_bool = True):
    """
    Takes as argument the dataset and a boolean to specify whether to apply the PCA or not
    Returns the training set and the test set

    @Author : Hamid MASSAOUD
    """
    X = df.iloc[:, :-1].values # get the features 
    labels = df.iloc[:,-1].values # get the labels   
    
    if pca_bool : 
        
        #Getting the number of component  
        cov_matrix = np.cov(X) # Computing Covariance of features 
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix) # Eigenvalue and its eigenvector 
        total = sum(eig_vals) 
        var = [(i / total)*100 for i in sorted(eig_vals, reverse=True)] # Variance captured by each component
        cum_var = np.cumsum(var) # Cumulative variance
        # Getting the number of component where 95% of our dataSet is being caputured.   
        for i in range(len(X)):       
            if cum_var[i]//95==1:  # Getting the component where the cumulative variance is equal to 95%    
                num = i
                break

        pca = PCA(n_components=num+1) # Apply PCA 
        pca.fit(X) # fit PCA
        X = pca.transform(X) # Transform X

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33) # Split dataSet
    return X_train, X_test, y_train, y_test

#Model Selection & Training :
def select_train_model(X_train, y_train, scoring_met, n_folds=10):
    """
    Takes as argument the training set and a scoring method used to determine the best model (Ex : 'f1', 'recall' etc.) alongside its hyperparameters.  
    Returns the best model found and its hyperparameters

    @Author : Hatim EL MALKI
    """
    
    pipe = Pipeline([('classifier' , DecisionTreeClassifier())])
    
    param_grid = [
                 {'classifier' : [DecisionTreeClassifier()],
                  'classifier__criterion':["gini", "entropy"],
                  'classifier__max_depth':[2,4,6,8,10]},
        
                 {'classifier': [GaussianNB()]},
                 
                 {'classifier' : [SVC()],
                  'classifier__C': [0.1, 1, 10, 100, 1000],  
                  'classifier__gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                  'classifier__kernel': ['poly', 'sigmoid', 'rbf']},

                 {'classifier' : [KNeighborsClassifier()],
                  'classifier__n_neighbors':[3,5,11,19],
                  'classifier__weights':['uniform', 'distance'],
                  'classifier__metric':['euclidean', 'manhattan']},
        
                 {'classifier' : [LogisticRegression()],
                  'classifier__penalty' : ['l1', 'l2'],
                  'classifier__C' : np.logspace(-4, 4, 20),
                  'classifier__solver' : ['liblinear'],
                  'classifier__max_iter' : [1000]},
            
    ]
        
    kf = KFold(shuffle=True, n_splits=n_folds)
    grid_search = GridSearchCV(pipe, param_grid, scoring=scoring_met, n_jobs = -1, cv = kf)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_



#Testing the chosen model:
def test_model(X_train, X_test, y_train, y_test, scoring_met, n_folds=10):
    """
	@Author : Hamid MASSAOUD
    """
    print('################################# Best Model Chosen by GridSearch #################################')
    model = select_train_model(X_train,y_train, scoring_met, n_folds)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    print(f'The accuracy of the chosen model is {accuracy: .2f}')
    print(f'The precision of the chosen model is {precision: .2f}')
    print(f'The recall of the chosen model is {recall: .2f}')
    print(clf_report)
    print('\n')