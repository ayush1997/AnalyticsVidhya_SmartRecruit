import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
# from sklearn.metrics.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from random import randint
from scipy.stats import pointbiserialr, spearmanr
from sklearn.svm import SVC
from sklearn import preprocessing

def feature_selection():
    df = pd.read_csv("dataset/train_new.csv")
    df = df.dropna()
    print df.describe()
    y = df["Business_Sourced"]
    print y.shape
    df = df.drop(['ID','Business_Sourced'],axis=1)

    param_df=df.columns.values
    print df
    print param_df
    # param_df=['Applicant_Gender','Applicant_Occupation','Manager_Joining_Designation'
    # ,'Manager_Current_Designation','Manager_Status','Manager_Gender'
    # ,'Manager_exp','App_age','Manager_age']

    scores = []
    scoreCV =[]
    # df = df[np.isfinite(df['EPS'])]
    # for j in range(5):
        # scores = []
        # scoreCV=[]
        # for i in range(0,len(param_df)-1):
    for i in range(len(param_df)):
        print i
        # print df[:,0:i+1:]
        # X = df.ix[:,0:(i+1)]
        X = df.ix[:,i:(i+1)]

        X = preprocessing.scale(X)
        print X.shape

        # print X
        clf = LogisticRegression()
        # clf = RandomForestClassifier(n_estimators=1000)
        # clf = DecisionTreeClassifier(criterion='gini')
        scoreCV = cross_validation.cross_val_score(clf, X, y, cv=3)

        print np.mean(scoreCV)
        scores.append(np.mean(scoreCV))

    plt.figure(figsize=(15,5))
    plt.plot(range(1,len(scores)+1),scores, '.-')
    plt.axis("tight")
    plt.title('Feature Selection', fontsize=14)
    plt.xlabel('# Features', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.show()

feature_selection()
