# MACHINE LEARNING EXERCISE WITH PYTHON

#import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import copy
import random

#get the file
path="C:/Users/fabnatal/Documents/Python Scripts"
input_file_name="Churn_Modelling.csv"

#read in the data
datos=pd.read_csv(path+"/"+input_file_name)

print("The shape of the input dataset is "+str(datos.shape))
print(datos.head)
print(datos.dtypes)

#dummy variables
print(datos["Geography"].unique())

mylist=list()
for i in range(len(datos["Geography"])):
    if datos["Geography"][i]=="France":
        mylist.append(1)
    else:
        mylist.append(0)
datos["France"]=mylist

datos["Spain"]=np.where(datos["Geography"]=="Spain",1,0)

datos=datos.drop(columns="Geography")
del(mylist,i)

print(datos["Gender"].unique())
datos["Female"]=np.where(datos["Gender"]=="Female",1,0)
datos=datos.drop(columns="Gender")

clientes=datos.iloc[:,[0,1]]
datos=datos.drop(columns=["Name","Surname"])

#split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
datos[datos.columns[datos.columns!="Exited"]], datos['Exited'],
test_size=0.25,shuffle=True,random_state=123)

print("X_train shape: "+str(X_train.shape))
print("X_test shape: "+str(X_test.shape))
print("y_train shape: "+str(y_train.shape))
print("y_test shape: "+str(y_test.shape))

#target rates
def print_rate(df,target_name):
    if len(df.shape)==2:
        conteo=(df[target_name].value_counts() / df[target_name].value_counts().sum())
        resultado=("The target rate is "+str(round(conteo[1],3)))
    if len(df.shape)==1:
        conteo=(df.value_counts() / df.value_counts().sum())
        resultado=("The target rate is "+str(round(conteo[1],3)))
    return resultado

print(print_rate(datos,"Exited"))
print(print_rate(y_train,"Exited"))
print(print_rate(y_test,"Exited"))



################### train/test #####################

methods_dict={"method":["KNN","LogReg","Tree"],
             "command":["KNeighborsClassifier()","LogisticRegression()","DecisionTreeClassifier(random_state=123)"]}


#predictions is a DataFrame with two columns (named 0 and 1, in this order)
#target is an list with the real values of the target variable in the test dataset
def lift(predictions,target,quantiles):
    predictions["real"]=target
    random.seed(123)
    predictions["aleatory"]=random.sample(range(len(predictions)),len(predictions))
    predictions=predictions.iloc[:,1:4]
    predictions=predictions.sort_values(by=[1,"aleatory"],ascending=False)
    bins=pd.cut(np.array(range(len(predictions))),quantiles,right=True,labels=False)
    predictions["bins"]=bins
    results=predictions.groupby(by="bins",axis=0,as_index=False).sum()
    results["sample_size"]=predictions.groupby(by="bins",axis=0,as_index=False).count().iloc[:,1]
    results["real_cumsum"]=np.cumsum(results["real"])
    results["real_cumsum_rate"]=(results["real_cumsum"])/sum(target)
    mean_rate=sum(target)/len(results)
    results["mean_rate"]=mean_rate
    results["lift"]=results["real"]/mean_rate
    mylist=[]
    for i in range(len(results)):
        mylist.append((results["real_cumsum"][i])/(mean_rate*(i+1)))
    results["lift_cum"]=mylist
    results=results.drop(columns=[1,"aleatory"])
    return results



#arg1 is a DataFrame created with the function lift
def plot_lift(arg1):
    fig,ax=plt.subplots(1,2)
    positions=(np.arange(len(arg1["lift_cum"])))+1
    ax[0].set_xlim(right=max(arg1["lift_cum"])+1)
    ax[0].barh(y=positions,width=arg1["lift_cum"])
    ax[0].set_xlabel("Cumulated lift")
    ax[0].set_ylabel("Bins")
    ax[0].set_title("Lift")
    for y, x in enumerate(arg1["lift_cum"]):
        ax[0].text(x+0.1, y+1, str(round(x,2)))
    ax[1].set_xlim(right=max(arg1["real_cumsum_rate"])+1)
    ax[1].barh(y=positions,width=arg1["real_cumsum_rate"])
    ax[1].set_xlabel("Cumulated captured target")
    ax[1].set_title("Captured target")
    for y, x in enumerate(arg1["real_cumsum_rate"]):
        ax[1].text(x+0.1, y+1, str(round(x,2)))
    plt.show()

    
def train_test_models(summary_models):
    for i in range(len(summary_models["command"])):
        newStr=(summary_models["method"][i])
        print("Train/test with {foo}".format(foo=newStr))
        clf=eval(methods_dict["command"][i])
        clf.fit(X_train, y_train)
        pred=pd.DataFrame(clf.predict_proba(X_test))
        results=lift(predictions=pred,target=list(y_test),quantiles=10)
        plot_lift(results)

train_test_models(methods_dict)

methods_dict_2=copy.deepcopy(methods_dict)
methods_dict_2["command"][2]=methods_dict_2["command"][2].replace(")",",max_depth=5)")
print(methods_dict_2)

train_test_models(methods_dict_2)
