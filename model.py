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
from sklearn.metrics import roc_auc_score
from sklearn.metrics.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import csv as csv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from xgboost.sklearn import XGBClassifier

col = ['Applicant_Gender','Applicant_Occupation','Applicant_Qualification'
,'Manager_Status','Manager_Gender','Manager_Num_Application'
,'Manager_Business','Manager_Business2','App_age'
,'Manager_age']

def feature_selection():
    df = pd.read_csv("dataset/train_new.csv")
    # df = df.dropna()
    print df.describe()
    y = df["Business_Sourced"]
    print y.shape
    df = df.drop(['ID','Business_Sourced'],axis=1)

    param_df=df.columns.values
    print df
    print param_df

    scores = []
    scoreCV =[]

    # for j in range(5):
        # scores = []
        # scoreCV=[]
        # for i in range(0,len(param_df)-1):
    for i in range(len(param_df)):
        print i
        # print df[:,0:i+1:]
        # X = df.ix[:,0:(i+1)]
        X = df.ix[:,i:(i+1)]

        # X = preprocessing.scale(X)
        print X.shape

        # print X
        # clf = LogisticRegression()
        clf = RandomForestClassifier(n_estimators=2000)
        # clf = DecisionTreeClassifier(criterion='gini')
        scoreCV = cross_validation.cross_val_score(clf, X, y, cv=3,scoring="roc_auc")

        print np.mean(scoreCV)
        scores.append(np.mean(scoreCV))

    plt.figure(figsize=(15,5))
    plt.plot(range(1,len(scores)+1),scores, '.-')
    plt.axis("tight")
    plt.title('Feature Selection', fontsize=14)
    plt.xlabel('# Features', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.show()


def model():

    df = pd.read_csv("dataset/train_new.csv")
    df_test = pd.read_csv("dataset/test_new.csv")
    ID =  df_test["ID"]
    # print df.describe()
    print df.columns.values


    Y = df["Business_Sourced"]

    # df_test = df_test.drop(["ID"],axis=1)
    df_test = df_test.drop(["ID",'Applicant_Occupation','Manager_Gender','Manager_Num_Application'
,'Manager_Business','Manager_Business2','Manager_age',"Applicant_Qualification"],axis=1)

    x_final = df_test
    print x_final.shape

    print Y.shape
    df = df.drop(['ID','Business_Sourced','Applicant_Occupation','Manager_Gender','Manager_Num_Application'
,'Manager_Business','Manager_Business2','Manager_age',"Applicant_Qualification"],axis=1)

    print df.columns.values


    # X = df
    X = df
    # X = preprocessing.scale(X)
    # X = StandardScaler().fit_transform(X)
    # print X.columns.values

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

    # print X_train
    pipeline=  Pipeline([
                        # ('clf',DecisionTreeClassifier())
                        # ('clf',RandomForestClassifier(criterion="entropy"))
                        # ('clf',LogisticRegression())
                        # ('clf',SGDClassifier( penalty="l2"))
                        # ('clf',SVC())
                        ('clf',XGBClassifier())

                        ])

    parameters={
        # 'clf__n_estimators':([50]),           # 500
        'clf__max_depth':([6]),
        # 'clf__min_samples_split':([3]),
        # 'clf__min_samples_leaf':([2])

        # 'clf__loss':('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
        # 'clf__n_iter':(100,200,50)

        # 'clf__gamma':([0.3]),
        # 'clf__C':([30]),
        # 'clf__kernel':(['poly']),
        # 'clf__degree':([3])

    }


    grid_search = GridSearchCV(pipeline,parameters,n_jobs=2,verbose=1,scoring="roc_auc")

    grid_search.fit(X,Y)

    print "Best score:",grid_search.best_score_
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):
        print (param_name,best_parameters[param_name])


    # pred_val = cross_validation.cross_val_predict(clf, X, y, cv=3,scoring="roc_auc")

    prediction = grid_search.predict(X_test).tolist()
    # print prediction
    # print "Accuracy score",accuracy_score(Y_test,prediction)
    # print "ra",roc_auc_score(Y_test,prediction)
    # print classification_report(Y_test,prediction)

    final_prediction = grid_search.predict(x_final).tolist()
    # print final_prediction

    with open("sample_submission2.csv", "wb") as predictions_file:
        # predictions_file = open("myfirstforest.csv", "wb")
        open_file_object = csv.writer(predictions_file, delimiter=',')
        open_file_object.writerow(["ID","Business_Sourced"])
        open_file_object.writerows(zip(ID,final_prediction))
        # predictions_file.close()
    print "Done"

def final():
    df = pd.read_csv("dataset/train_new.csv")
    df_test = pd.read_csv("dataset/test_new.csv")
    ID =  df_test["ID"]
    # print df.describe()
    print df.columns.values


    Y = df["Business_Sourced"]

    # df_test = df_test.drop(["ID"],axis=1)
    df_test = df_test.drop(["ID",'Applicant_Occupation'
,'Manager_Gender','Manager_Num_Application'
,'Manager_Business','Manager_Business2','Manager_age'],axis=1)

    x_final = df_test
    print x_final.shape

    print Y.shape
    df = df.drop(['ID','Business_Sourced','Applicant_Occupation'
,'Manager_Gender','Manager_Num_Application'
,'Manager_Business','Manager_Business2','Manager_age'],axis=1)

    X = df

    print df.columns.values

    # xgb_model = xgboost.XGBClassifier(objective="multi:softprob", nthread=-1)
    #
    # clf = GridSearchCV(
    #     xgb_model,
    #     {
    #         'max_depth': [1, 2, 3],
    #         'n_estimators': [4, 5, 6],
    #         'learning_rate': [0.1, 0.2],
    #     },
    #     cv=10,
    #     verbose=10,
    #     n_jobs=1,
    #     scoring="roc_auc"
    # )




    # clf.fit(x, y)
    # clf = RandomForestClassifier(criterion="entropy",n_estimators=2000,max_depth=275,min_samples_split=3,min_samples_leaf=2)
    clf = clf.fit(X,Y)

    final_prediction = clf.predict(x_final).tolist()
    print final_prediction

    with open("sample_submission2.csv", "wb") as predictions_file:
        # predictions_file = open("myfirstforest.csv", "wb")
        open_file_object = csv.writer(predictions_file, delimiter=',')
        open_file_object.writerow(["ID","Business_Sourced"])
        open_file_object.writerows(zip(ID,final_prediction))
        # predictions_file.close()
    print "Done"







# feature_selection()
model()
# final()
