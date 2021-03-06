#import relevant modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import copy

#input file
path="C:/Users/fabnatal/Documents/Python Scripts"
input_file_name="Churn_Modelling.csv"

datos=pd.read_csv(path+"/"+input_file_name)

print("The shape of the input dataset is "+str(datos.shape))
print(datos.head)
print(datos.dtypes)

#see the unique values of the categorical variables
print(datos["Geography"].unique())
print(datos["Gender"].unique())

#make dummy variables
mylist=list()
for i in range(len(datos["Geography"])):
    if datos["Geography"][i]=="France":
        mylist.append(1)
    else:
        mylist.append(0)
datos["France"]=mylist
del(mylist,i)

#another way to build the dummy variable
datos["Spain"]=np.where(datos["Geography"]=="Spain",1,0)

datos=datos.drop(columns="Geography")

datos["Female"]=np.where(datos["Gender"]=="Female",1,0)
datos=datos.drop(columns="Gender")

clients=datos.iloc[:,[0,1]]
datos=datos.drop(columns=["Name","Surname"])

print("The shape of the input dataset is "+str(datos.shape))
print(datos.head)
print(datos.dtypes)

#split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
datos[datos.columns[datos.columns!="Exited"]], datos['Exited'],
test_size=0.25,shuffle=True,random_state=123)

print("X_train shape: "+str(X_train.shape))
print("X_test shape: "+str(X_test.shape))
print("y_train shape: "+str(y_train.shape))
print("y_test shape: "+str(y_test.shape))

#define a function to calculate the target rates
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

#prepare a pipeline to train and test three models.
#First, we will define a dictionary which will contain the names of the three algorithms and the corresponding python commands. 
#We will update this dictionary with accuracy values obtained from the train and test processes. 
#Moreover, we will define a function to visually compare the performances of the three methods.
methods_dict={"method":["KNN","LogReg","Tree"],
             "command":["KNeighborsClassifier()","LogisticRegression()","DecisionTreeClassifier(random_state=123)"],
             "train_accuracy":[],
             "test_accuracy":[]}

def train_test_models(summary_models):
    for i in range(len(summary_models["command"])):
        newStr=(summary_models["method"][i])
        print("Train/test with {foo}".format(foo=newStr))
        clf=eval(summary_models["command"][i])
        clf.fit(X_train, y_train)
        train_score=round(clf.score(X_train, y_train),5)
        y_pred=clf.predict(X_test)
        test_score=round((sum(y_pred==y_test) / len(y_test)),5)
        summary_models["train_accuracy"].append(train_score)
        summary_models["test_accuracy"].append(test_score)

train_test_models(methods_dict)
#"https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/"

def see_accuracy(summary_models):
    fig,ax=plt.subplots()
    ax.plot(summary_models["train_accuracy"],color="b",linestyle="--",marker="o")
    ax.plot(summary_models["test_accuracy"],color="r",marker="v")
    ax.set_xticks([0,1,2],minor=False)
    ax.set_xticklabels(summary_models["method"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy")
    plt.show()

see_accuracy(methods_dict)

#introduce the argument "max_depth" to limit the depth of the tree, and set it equal to 5 (this is arbitrary).
#First, we will create a copy of the dictionary we created earlier, and we will update its elemets. 
#Then we will use the fuctions defined above to train and test the models, and to display the results.
methods_dict_2=copy.deepcopy(methods_dict)
methods_dict_2["command"][2]=methods_dict_2["command"][2].replace(")",",max_depth=5)")
methods_dict_2["train_accuracy"]=[]
methods_dict_2["test_accuracy"]=[]
print(methods_dict_2)

train_test_models(methods_dict_2)

see_accuracy(methods_dict_2)

#compare the results on a same plot sharing the scale of the y axis:
fig,ax=plt.subplots(1,2,sharey=True)
ax[0].plot(methods_dict["train_accuracy"],color="b",linestyle="--",marker="o")
ax[0].plot(methods_dict["test_accuracy"],color="r",marker="v")
ax[0].set_title("'Overfitted Tree' method")
ax[0].set_xticks([0,1,2],minor=False)
ax[0].set_xticklabels(methods_dict["method"])
ax[0].set_xlabel("Method")
ax[0].set_ylabel("Accuracy")
ax[1].plot(methods_dict_2["train_accuracy"],color="b",linestyle="--",marker="o")
ax[1].plot(methods_dict_2["test_accuracy"],color="r",marker="v")
ax[1].set_title("'Pruned Tree' method")
ax[1].set_xticks([0,1,2],minor=False)
ax[1].set_xticklabels(methods_dict["method"])
ax[1].set_xlabel("Method")
plt.show()
