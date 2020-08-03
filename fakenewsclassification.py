from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox,simpledialog,filedialog

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('all')
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
root=Tk()
root.title("Fake News Classification")
#root.geometry('800x600')
root.config(bg="#FF00CC")

def upload():
	global train_data
	train_data=askopenfilename(initialdir="dataset")
	label=Label(root,text="data is loaded",font='bold')
	label.grid(row=2,column=0)
	return train_data

def data():
	global dframe
	dframe=pd.read_csv(train_data)
	label=Label(root,text="data is readed",font='bold')
	label.grid(row=2,column=2)

def stats():
	global null
	null=dframe.isnull().sum()
	label=Label(root,text=str(null))
	label.grid(row=4,column=0)

def individuals():
	global x,y
	x=dframe['Headline']
	y=dframe['Label']
	return x,y

def grammer():
	global corpus
	global x,y
	global X
	x,y=individuals()
	wl=WordNetLemmatizer()
	cv=CountVectorizer()
	sw=stopwords.words('english')
	corpus=[]
	for i in range(0,len(x)):
		review=re.sub('[^a-zA-Z]',' ',x[i])
		review=review.lower()
		review=review.split()
		review=[wl.lemmatize(item) for item in review if item not in sw]
		review=' '.join(review)
		corpus.append(review)
	X=cv.fit_transform(corpus).toarray()
	return X,y

def model_selection():
	global x_train,x_test,y_train,y_test  
	global X
	global y
	X,y=grammer()
	x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
	return x_train,x_test,y_train,y_test

def Naivebayes():
	global train_data,dframe
	global x_train,x_test,y_train,y_test
	global X
	global y
	X,y=grammer()
	x_train,x_test,y_train,y_test=model_selection()
	model=MultinomialNB()
	model.fit(x_train,y_train)
	predn=model.predict(x_test)
	cf=confusion_matrix(y_test,predn)
	accurate=accuracy_score(y_test,predn)
	a=accurate*100
	r=np.round(a,2)
	label=Label(root,text="cf: "+str(cf),font=15)
	label.grid(row=4,column=2)
	label=Label(root,text="accuracy: "+str(r),font=15)
	label.grid(row=5,column=2)

label=Label(root,text="Fake News Classification",font=('bold',20),fg="red")
label.grid(row=0,column=1)

myButton=Button(root,text="Upload Dataset",font='bold',command=upload,width=17,bg='orange')
myButton.grid(row=1,column=0,pady=10)
myButton=Button(root,text="Data",width=17,command=data,bg='orange',font='bold')
myButton.grid(row=1,column=2,pady=10)
myButton=Button(root,text="Null values",width=17,command=stats,bg='orange',font='bold')
myButton.grid(row=3,column=0,pady=10)

myButton=Button(root,text="MultinomailNB",command=Naivebayes,width=17,bg='orange',font='bold')
myButton.grid(row=3,column=2,pady=10)

root.mainloop()