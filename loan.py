from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox,simpledialog,filedialog

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
root=Tk()
root.title("Breast Cancer Prediction")
#root.geometry('800x600')
root.config(bg="#FAD7A0")

 
def upload():
	global train_data
	train_data=askopenfilename(initialdir="dataset")
	label=Label(root,text="data is loaded",font='bold')
	label.grid(row=2,column=0)
	return train_data
	
def data():
	global dframe
	dframe=pd.read_csv(train_data)
	dframe=pd.DataFrame(dframe,columns=['ApplicantIncome','LoanAmount','Credit_History','Loan_Status'])
	label=Label(root,text="data is readed",font='bold')
	label.grid(row=2,column=1)

def stats():
	global null
	global dframe
	dframe=dframe.fillna(np.mean(dframe))
	null=dframe.isnull().sum()
	label=Label(root,text=str(null))
	label.grid(row=2,column=2)

def individuals():
	global x,y
	x=dframe.iloc[:,:-1].values
	y=dframe.iloc[:,-1].values
	y=y.reshape(-1,1)
	return x,y
def encoding():
	global y
	lb=LabelEncoder()
	y=lb.fit_transform(y)
	return y
def model_selection():
	global x_train,x_test,y_train,y_test
	x,y=individuals()
	y=encoding()
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
	return x_train,x_test,y_train,y_test

def eda():
	global totals,trains,tests
	global x_train,x_test,y_train,y_test
	x_train,x_test,y_train,y_test=model_selection()
	totals=len(dframe)
	trains=len(x_train)
	tests=len(x_test)
	label=Label(root,text=("totalsamples are:"+str(totals),("\ntraining samples are:"+str(trains)),("\n testing samples are:"+str(tests))))
	label.grid(row=2,column=3)
'''
def preprocessing():
	global ms
	global x_train,x_test,y_train,y_test
	x_train,y_train,x_test,y_test=model_selection()
	ms=MinMaxScaler()
	x1_train=ms.fit_transform(x_train)
	x1_test=ms.transform(x_test)
	return x1_train,x1_test
'''
def logistic():
	global train_data,dframe
	global x_train,x_test,y_train,y_test
	x_train,x_test,y_train,y_test=model_selection()
	lr=LogisticRegression()
	lr.fit(x_train,y_train)
	predl=lr.predict(x_test)
	cf=confusion_matrix(y_test,predl)
	accurate=accuracy_score(y_test,predl)
	a=accurate*100
	r=np.round(a,2)
	label=Label(root,text="cf: "+str(cf),font=15)
	label.grid(row=5,column=0)
	label=Label(root,text="accuracy: "+str(r),font=15)
	label.grid(row=6,column=0)
	#label=Label(root,text="predictions are"+str(predl))
	#label.grid(row=7,column=0)
def random():
	global train_data,dframe
	global x_train,x_test,y_train,y_test
	x_train,x_test,y_train,y_test=model_selection()
	rf=RandomForestClassifier()
	rf.fit(x_train,y_train)
	predr=rf.predict(x_test)
	cf=confusion_matrix(y_test,predr)
	accurate=accuracy_score(y_test,predr)
	a=accurate*100
	r=np.round(a,2)
	label=Label(root,text="cf: "+str(cf),font=15)
	label.grid(row=2,column=4,pady=10)
	label=Label(root,text="accuracy: "+str(r),font=15)
	label.grid(row=3,column=4,pady=10)
def knn(): 
	global x_train,x_test,y_train,y_test
	x_train,x_test,y_train,y_test=model_selection()
	knn=KNeighborsClassifier()
	knn.fit(x_train,y_train)
	predk=knn.predict(x_test)
	cf=confusion_matrix(y_test,predk)
	accuratek=accuracy_score(y_test,predk)
	a=accuratek*100
	r=np.round(a,2)
	label=Label(root,text="cf: "+str(cf),font=15)
	label.grid(row=5,column=1)
	label=Label(root,text="accuracy: "+str(r),font=15)
	label.grid(row=6,column=1)
	#label=Label(root,text="predictions are"+str(predk))
	#label.grid(row=7,column=1)
def Decision():
	global x_train,x_test,y_train,y_test
	x_train,x_test,y_train,y_test=model_selection()
	dt=DecisionTreeClassifier()
	dt.fit(x_train,y_train)
	predd=dt.predict(x_test)
	cf=confusion_matrix(y_test,predd)
	accurated=accuracy_score(y_test,predd)
	a=accurated*100
	r=np.round(a,2)
	label=Label(root,text="cf: "+str(cf),font=15)
	label.grid(row=5,column=2)
	label=Label(root,text="accuracy: "+str(r),font=15)
	label.grid(row=6,column=2)
	#label=Label(root,text="predictions are"+str(predd))
	#label.grid(row=7,column=2)
def naivebayes():
	global nb,predn
	global x_train,x_test,y_train,y_test
	x_train,x_test,y_train,y_test=model_selection()
	nb=GaussianNB()
	nb.fit(x_train,y_train)
	predn=nb.predict(x_test)
	cf=confusion_matrix(y_test,predn)
	accuraten=accuracy_score(y_test,predn)
	a=accuraten*100
	r=np.round(a,2)
	label=Label(root,text="cf: "+str(cf),font=15)
	label.grid(row=5,column=3)
	label=Label(root,text="accuracy: "+str(r),font=15)
	label.grid(row=6,column=3)
	#label=Label(root,text="predictions are"+str(predn))
	#label.grid(row=7,column=3)
def svc():
	global x_train,x_test,y_train,y_test
	x_train,x_test,y_train,y_test=model_selection()
	sv=SVC()
	sv.fit(x_train,y_train)
	preds=sv.predict(x_test)
	cf=confusion_matrix(y_test,preds)
	accurates=accuracy_score(y_test,preds)
	a=accurates*100
	r=np.round(a,2)
	label=Label(root,text="cf: "+str(cf),font=15)
	label.grid(row=5,column=4)
	label=Label(root,text="accuracy "+str(r),font=15)
	label.grid(row=6,column=4)
	#label=Label(root,text="predictions are"+str(preds))
	#label.grid(row=7,column=4)

label=Label(root,text="Bank Loan Prediction",font=('bold',20),fg="red")
label.grid(row=0,column=2)
myButton=Button(root,text="Upload Dataset",font='bold',command=upload,width=17,bg='orange')
myButton.grid(row=1,column=0,pady=10)
myButton=Button(root,text="Data",width=17,command=data,bg='orange',font='bold')
myButton.grid(row=1,column=1,pady=10)
myButton=Button(root,text="Null values",width=17,command=stats,bg='orange',font='bold')
myButton.grid(row=1,column=2,pady=10)
myButton=Button(root,text="EDA",width=17,command=eda,bg='orange',font='bold')
myButton.grid(row=1,column=3,pady=10)

myButton=Button(root,text="RandomForest",width=17,command=random,bg='orange',font='bold')
myButton.grid(row=1,column=4,pady=10)
myButton=Button(root,text="Logistic regression",width=17,font='bold',bg="orange",command=logistic)
myButton.grid(row=4,column=0,pady=10,padx=10)
myButton=Button(root,text="knn algorithm",width=17,font='bold',bg="orange",command=knn)
myButton.grid(row=4,column=1,pady=10,padx=10)
myButton=Button(root,text="Decision Tree",width=17,font='bold',bg="orange",command=Decision)
myButton.grid(row=4,column=2,pady=10,padx=10)
myButton=Button(root,text="Naive bayes",width=17,font='bold',bg="orange",command=naivebayes)
myButton.grid(row=4,column=3,pady=10,padx=10)
myButton=Button(root,text="SVM",width=17,font='bold',bg="orange",command=svc)
myButton.grid(row=4,column=4,pady=10,padx=10)
label=Label(root,text="New Prediction",font=('bold',16),fg='red')
label.grid(row=8,column=2,padx=10,pady=10)


def clear1(event):
	rad.delete(0,END)
def clear2(event):
	tex.delete(0,END)
def clear3(event):
	area.delete(0,END)
def clear4(event):
	per.delete(0,END)
def clear5(event):
	smooth.delete(0,END)
def predict():
	global nb,predn
	global x_train,x_test,y_train,y_test
	x_train,x_test,y_train,y_test=model_selection()
	Income=float(income.get())
	amount=float(loanamount.get())
	credith=float(credit.get())
	nb=GaussianNB()
	nb.fit(x_train,y_train)
	predn=nb.predict(x_test)
	newp=nb.predict([[Income,amount,credith]])
	label=Label(root,text="Loan status is: "+str(newp),font=('bold',16),fg='red')
	label.grid(row=11,column=2,padx=10,pady=10)

income=Entry(root,width=17,bg="orange")
income.grid(row=9,column=0,padx=10,pady=10)
income.insert(0,'ApplicantIncome')
loanamount=Entry(root,width=17,bg="orange")
loanamount.grid(row=9,column=2,padx=10,pady=10)
loanamount.insert(0,'LoanAmount')
credit=Entry(root,width=17,bg="orange")
credit.grid(row=9,column=4,padx=10,pady=10)
credit.insert(0,'Credit_History')
myButton=Button(root,text="PREDICTION",width=14,font='bold',bg="orange",command=predict)
myButton.grid(row=10,column=2,pady=10,padx=10)
root.mainloop()