import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from datetime import datetime

def load_train():
    df =  pd.read_csv("dataset/Train_pjb2QcD.csv")
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

    columns = ['ID','Office_PIN','Application_Receipt_Date','Applicant_City_PIN',
    'Applicant_Gender','Applicant_BirthDate','Applicant_Marital_Status',
    'Applicant_Occupation','Applicant_Qualification','Manager_DOJ',
    'Manager_Joining_Designation','Manager_Current_Designation',
    'Manager_Grade','Manager_Status','Manager_Gender','Manager_DoB',
    'Manager_Num_Application','Manager_Num_Coded','Manager_Business',
    'Manager_Num_Products','Manager_Business2','Manager_Num_Products2',
    'Business_Sourced']

    f = ["Applicant_Gender","Applicant_Marital_Status","Applicant_Qualification","Applicant_Occupation","Manager_Joining_Designation","Manager_Current_Designation","Manager_Grade","Manager_Status","Manager_Gender"]
    for k in f:
        a = df[k].unique()
        for i,j in enumerate(a, start=0):
            df.loc[df[k]==j,k]=i

    # print df.head(100)

    df["Manager_exp"] = df.apply(ret_days, axis=1)
    df["App_age"] = df.apply(ret_app_age, axis=1)
    df["Manager_age"] = df.apply(ret_man_age, axis=1)

    df = df.drop(["Applicant_BirthDate","Application_Receipt_Date","Manager_DoB","Manager_DOJ"],axis=1)

    print df.head()
    df.to_csv("dataset/rain_new.csv",index=False)

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
        b = datetime.strptime(df["Applicant_BirthDate"],date_for)
        # print b
        d = a-b
        # print (d.days)/365.0
        return (d.days)/365.0
    except:
        return None

def correlation():
    df =  pd.read_csv("dataset/Train_pjb2QcD.csv")

    columns = ["Applicant_Gender"]
    for c in columns:
        #Check if binary or continuous
        if len(df[c].unique())<=2:
            corr = spearmanr(df['Survived'],df[c])[0]
            y = df['Survived']
            x = df[c]
            X = np.vstack((y,x))
            covar = np.cov(X)
        else:
            corr = pointbiserialr(df['Survived'],df[c])[0]
            print corr
            y = df['Survived']
            x = df[c]
            X = np.vstack((y,x))
            covar = np.cov(X)
        param.append(c)
        correlation.append(corr)
        abs_corr.append(abs(corr))
        covariance.append(covar[0][1])
    print covariance

if __name__ == "__main__":
    load_train()
