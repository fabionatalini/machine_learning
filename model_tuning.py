import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import os

#set the working directory and introduce the name of the input file
path="C:\\Users\\fabnatal\\Documents\\Python Scripts"
file_name="Churn_Modelling.csv"

#read in the data
f=open(os.path.join(path,file_name))
datos=pd.read_csv(f)

#have a look at the data
print("The shape of the input dataset is "+str(datos.shape))
print(datos.head)
print(datos.dtypes)

#"Geography" and "Gender" are categorical variables, 
#so we will construct binary dummy variables from them. 
#Moreover, the variables "Name" and "Surname" are useless in the model, 
#so we can get rid of them.

#see the unique values of the categorical variables
print(datos["Geography"].unique())
print(datos["Gender"].unique())
#make dummy variables
datos["France"]=np.where(datos["Geography"]=="France",1,0)
datos["Spain"]=np.where(datos["Geography"]=="Spain",1,0)
datos["Female"]=np.where(datos["Gender"]=="Female",1,0)
#drop variables
datos=datos.drop(columns="Geography")
datos=datos.drop(columns="Gender")
datos=datos.drop(columns=["Name","Surname"])

#have a look at the data again
print("The shape of the input dataset is "+str(datos.shape))
print(datos.head)
print(datos.dtypes)

#split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
datos[datos.columns[datos.columns!="Exited"]], datos['Exited'],
test_size=0.25,shuffle=True,random_state=123)
print("X_train shape: {foo1} rows and {foo2} columns".format(foo1=X_train.shape[0],foo2=X_train.shape[1]))
print("X_test shape: {foo1} rows and {foo2} columns".format(foo1=X_test.shape[0],foo2=X_test.shape[1]))
print("y_train shape: {foo1} rows and 0 columns".format(foo1=y_train.shape[0]))
print("y_test shape: {foo1} rows and 0 columns".format(foo1=y_test.shape[0]))


#Let's focus on two parameters of the DecisionTreeClassifier() algorithm:
#"max_depth" and "min_samples_split" 
#The version of scikit-learn we are using is
print(sklearn.__version__)
#The scikit-learn documentation correspondign to our sklearn verison describes max_depth as
#"maximum depth of the tree. If None, then nodes are expanded until all leaves are pure 
#or until all leaves contain less than min_samples_split samples"
#On the other hand, "min_samples_split" is defined as
#"minimum number of samples required to split an internal node"
# The default value of min_samples_split is 2. So let's set the min_samples_split equal to 500.
#In this case we can set max_depth equal to 
print(X_train.shape[0] / 500)

#Now let's build a loop to train the model with all the possibel values of max_depth
train_accuracy=[]
test_accuracy=[]

for i in np.arange(1,16):
    print("Train/test with max_depth="+str(i))
    clf=DecisionTreeClassifier(random_state=123,max_depth=i)
    clf.fit(X_train, y_train)
    train_score=round(clf.score(X_train, y_train),5)
    train_accuracy.append(train_score)
    y_pred=clf.predict(X_test)
    test_score=round((sum(y_pred==y_test) / len(y_test)),5)
    test_accuracy.append(test_score)

#Let's see how the accuracy of the model changes depending on the max_depth
fig,ax=plt.subplots()
ax.plot(train_accuracy,label="Train accuracy",color="b",linestyle="--")
ax.plot(test_accuracy,label="Test accuracy",color="r",linestyle="--")
ax.set_xticks(np.arange(0,len(train_accuracy)))
ax.set_xticklabels(np.arange(0,len(train_accuracy))+1)
ax.set_xlabel("Tree max_depth")
ax.set_ylabel("Accuracy")
ax.legend(loc="upper center")

foo=test_accuracy.index(max(test_accuracy))
print("The max_depth with the highest test accuracy is {}".format(foo+1))

#interestingly, the accuracy on the train set increases with max_depth
#but the accuracy on the test set starts decreasing when max_depth is equal to 6
#This indicates that the model is overfitted when max_depth is greater than 5





