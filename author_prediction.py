# -*- coding: utf-8 -*-

import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
import numpy as np


filename = 'chat_all.csv'
corpus = pd.read_csv(filename, sep=',', encoding="utf-8")
corpus = corpus.fillna('')

filter = corpus['Text_stemmed'] != '[]' 
corpus_clean = corpus[filter]

corpus_sample_Name2 = corpus_clean[corpus_clean.Name=='Name2'].sample(n=8000, random_state=42)
corpus_sample_Name1 = corpus_clean[corpus_clean.Name=='Name1'].sample(n=8000, random_state=42)
frames = [corpus_sample_Name2, corpus_sample_Name1]
corpus_sample = pd.concat(frames)

text = list(corpus_sample.Text_stemmed.values)
labels = list(corpus_sample.Name.values)

#Encode the labes to int
le = preprocessing.LabelEncoder()
le.fit(['Name1','Name2'])
labels_int = le.transform(labels)

#vectorize all tokens with TF-IDF measure and get n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X = vectorizer.fit_transform(text) 
X = X.toarray()

#split into train and test datasets: X stands for features, y stands for labels
X_train, X_test, y_train, y_test = train_test_split(X, labels_int, test_size=0.25, random_state=42)
#get words selected as feauters
feature_names = vectorizer.get_feature_names()

n_train = len(X_train)
n_test = len(X_test)

X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_test = X_test[:n_test]
y_test = y_test[:n_test] 

from sklearn.feature_selection import SelectKBest, chi2
#feature selection with chi2 statistics
ch2 = SelectKBest(chi2, k=7000)
X_train_new = ch2.fit_transform(X_train, y_train)
X_test_new = ch2.transform(X_test)
X_train_new.shape
X_test_new.shape

feature_names_ch2 = [feature_names_chi2[i] for i in ch2.get_support(indices=True)]

"""
#first we do gridsearch over hyperparameters of SVM classifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#the grids of parameters
parameters = {'kernel':['rbf'], 'C':[10, 100], 'gamma': [0.5, 0.75, 1]}
svc = svm.SVC(gamma="scale")
#define classifier
clf = GridSearchCV(svc, parameters, cv=5, n_jobs=4, verbose=True)
clf.fit(X_train, y_train)
#add the result of grid search to Dataframe
grid_result = pd.DataFrame(clf.cv_results_)
print(grid_result)
file_name = "GRID_whatsapp_author.csv"
grid_result.to_csv(file_name, encoding='utf-8', index=False)
df_C= grid_result.loc[grid_result['rank_test_score'] == 1] 
C = list(df_C.param_C.values)
gamma = list(df_C.param_gamma.values)



#next, cross-validation training with the best parameters; here we can tune the model, change features, parameters etc. 
import io
from sklearn.model_selection import cross_val_predict
clf = SVC(gamma=gamma, C=C, kernel='rbf')
clf.fit(X_train_new, y_train)
y_pred = cross_val_predict(clf, X_train_new, y_train, cv=10)
target_names=['Name1', 'Name2']
cl_report = classification_report(y_train, y_pred, target_names=target_names)
cm = confusion_matrix(y_train, y_pred)
print(cm)
#convert clssification report to pandas dataframe
report_df = pd.read_fwf(io.StringIO(cl_report), sep="\s+")

"""
#train and test SVM model with best parameters selected after cross-validation
clf = SVC(gamma=1, C=10, kernel='rbf')
clf.fit(X_train_new, y_train)
y_pred = clf.predict(X_test_new)

accuracy = accuracy_score(y_test, y_pred)
f_score_macro = f1_score(y_test, y_pred, average='macro')
f_score_micro = f1_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='macro') 
recall = recall_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)
target_names=['Name1', 'Name2']
    
print(cm)
print('f-score macro:', f_score_macro)
print('f-score micro:', f_score_micro)
print('precision: ', precision)
print('recall: ', recall)
print('accuracy', accuracy)

    
