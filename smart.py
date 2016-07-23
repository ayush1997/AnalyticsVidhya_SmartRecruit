import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from datetime import datetime
import numpy as np

columns = ['ID','Office_PIN','Application_Receipt_Date','Applicant_City_PIN',
'Applicant_Gender','Applicant_BirthDate','Applicant_Marital_Status',
'Applicant_Occupation','Applicant_Qualification','Manager_DOJ',
'Manager_Joining_Designation','Manager_Current_Designation',
'Manager_Grade','Manager_Status','Manager_Gender','Manager_DoB',
'Manager_Num_Application','Manager_Num_Coded','Manager_Business',
'Manager_Num_Products','Manager_Business2','Manager_Num_Products2',
'Business_Sourced']

def load_train(filename,dest):
    df =  pd.read_csv(filename)
    # print df.describe()


    print df["Applicant_Marital_Status"].describe()
    print df["Applicant_Marital_Status"].unique()
    # print df["Applicant_City_PIN"].describe()

    # df.loc[df["Applicant_Gender"]=="M","Applicant_Gender"]=0
    # df.loc[df["Applicant_Gender"]=="F","Applicant_Gender"]=1
    #
    #
    # # print df.columns.values
    # # print df[df["Applicant_Gender"]=="M"][df["Business_Sourced"]==0].describe()
    # print df[df["Applicant_Gender"]=="M"].describe()


    f = ["Applicant_Gender","Applicant_Marital_Status","Applicant_Qualification","Applicant_Occupation","Manager_Joining_Designation","Manager_Current_Designation","Manager_Grade","Manager_Status","Manager_Gender"]
    for k in f:
        a = df[k].unique()
        for i,j in enumerate(a, start=0):
            df.loc[df[k]==j,k]=i

    # print df.head(100)

    df["Manager_exp"] = df.apply(ret_days, axis=1)
    df["App_age"] = df.apply(ret_app_age, axis=1)
    df["Manager_age"] = df.apply(ret_man_age, axis=1)

    # print df.describe()

    print df["App_age"].mean()
    df['App_age'] = df['App_age'].fillna(df["App_age"].mean())
    print df["Manager_age"].mean()
    df['Manager_age'] = df['Manager_age'].fillna(df["Manager_age"].mean())
    df['Manager_exp'] = df['Manager_exp'].fillna(df["Manager_exp"].mean())
    df['Applicant_Qualification'] = df['Applicant_Qualification'].fillna(1)
    df['Applicant_Gender'] = df['Applicant_Gender'].fillna(0)
    df['Manager_Gender'] = df['Manager_Gender'].fillna(0)
    df['Manager_Status'] = df['Manager_Status'].fillna(0)
    df['Applicant_Occupation'] = df['Applicant_Occupation'].fillna(2)

    c = ['Manager_Num_Application','Manager_Num_Coded','Manager_Business','Manager_Num_Products','Manager_Business2','Manager_Num_Products2']
    for i in c:
        df[i] = df[i].fillna(df[i].mean())

    df = df.drop(["Applicant_BirthDate","Application_Receipt_Date","Manager_DoB","Manager_DOJ","Applicant_City_PIN","Manager_Grade","Office_PIN","Applicant_Marital_Status","Manager_Num_Coded","Manager_Num_Products","Manager_Num_Products2","Manager_Current_Designation","Manager_Joining_Designation","Manager_exp"],axis=1)

    print df.columns.values
    print df.describe()
    df.to_csv(dest,index=False)

    # for i in columns:
    #     print i
    #     print df[i].describe()
    #     print df[i].unique().size

def ret_days(df):
    try:
        # print df
        date_for = "%m/%d/%Y"
        a = datetime.strptime(df["Application_Receipt_Date"],date_for)
        # print a
        b = datetime.strptime(df["Manager_DOJ"],date_for)
        # print b
        d = a-b
        # print (d.days)/365.0
        return (d.days)/365.0
    except:
        return None

def ret_app_age(df):
    try:
        # print df
        date_for = "%m/%d/%Y"
        a = datetime.strptime(df["Application_Receipt_Date"],date_for)
        # print a
        b = datetime.strptime(df["Applicant_BirthDate"],date_for)
        # print b
        d = a-b
        # print (d.days)/365.0
        return (d.days)/365.0
    except:
        return None
def ret_man_age(df):
    try:
        # print df
        date_for = "%m/%d/%Y"
        a = datetime.strptime(df["Application_Receipt_Date"],date_for)
        # print a
        b = datetime.strptime(df["Manager_DoB"],date_for)
        # print b
        d = a-b
        # print (d.days)/365.0
        return (d.days)/365.0
    except:
        return None

def correlation():
    df =  pd.read_csv("dataset/train_new.csv")
    # df = df.dropna(axis=0,how="any")
    print df.describe()
    # print df.head()
    param=[]
    correlation=[]
    abs_corr=[]
    covariance = []
    columns = ["Applicant_Gender","App_age","Applicant_Occupation","Applicant_Qualification","Manager_age","Manager_Status","Manager_Gender","Manager_Business","Manager_Business2","Manager_Num_Application"]
    for c in columns:
        #Check if binary or continuous

        if len(df[c].unique())<=12:
            corr = spearmanr(df['Business_Sourced'],df[c])[0]
            print "spear",c,corr
            y = df['Business_Sourced']
            x = df[c]
            X = np.vstack((y,x))
            covar = np.cov(X)
        else:
            corr = pointbiserialr(df['Business_Sourced'],df[c])[0]
            print "point",c,corr
            y = df['Business_Sourced']
            x = df[c]
            X = np.vstack((y,x))
            covar = np.cov(X)
        param.append(c)
        correlation.append(corr)
        abs_corr.append(abs(corr))
        # covariance.append(covar[0][1])
    print covariance

def ana_test():
    df = pd.read_csv("dataset/test_new.csv")
    print df.describe()
    # for i in columns[:-1]:
    #     print i
    #     print df[i].describe()
    #     print df[i].unique().size
if __name__ == "__main__":
    # load_train("dataset/Test_wyCirpO.csv","dataset/test_new.csv")
    # load_train("dataset/Train_pjb2QcD.csv","dataset/train_new.csv")
    # correlation()
    ana_test()
