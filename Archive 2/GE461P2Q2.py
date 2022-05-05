from cgi import test
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import matplotlib.image as mpimg
import numpy as np
import seaborn as sn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection._split import train_test_split
from sklearn import *
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random

#obtaining data from matlab file
data=loadmat('digitss.mat')

#splitting data into training and testing data
train_digits,test_digits,train_label,test_label=train_test_split(data['digits'],data['labels'], test_size=0.5,random_state=42,shuffle=True)
sc = StandardScaler()

#standardizing the data
train_digits = sc.fit_transform(train_digits)
test_digits = sc.transform(test_digits)
x = random.randint(1,9)
print(x)

#fitting the data to LDA
lda = LDA(n_components=x,store_covariance=True)
train_digits = lda.fit_transform(train_digits, train_label.ravel())
test_digits = lda.transform(test_digits)

#eigenvalues and eigenvectors plots
eigenvalues,eigenvectors = np.linalg.eig(train_digits.T.dot(train_digits))
samplemean = np.mean(train_digits,axis=0)
img=mpimg.imsave('samplemean.png',samplemean.reshape(1,x))
img = mpimg.imread('samplemean.png')
imgplot = plt.imshow(img)
plt.show()
eigenvectors=np.real(eigenvectors)
show_vectors=np.dot(train_digits.T.dot(train_digits) ,eigenvectors)
vimg=mpimg.imsave('show_vectors.png',show_vectors.reshape(x,x))
vimg = mpimg.imread('show_vectors.png')
vimgplot = plt.imshow(vimg)
plt.show()

#creating a classifier using Gaussian Process Classifier with kernel as RBF 
kernel = 1.0 * RBF(1.0)
classifier = GaussianProcessClassifier(kernel=kernel, random_state=0, n_jobs = -1).fit(train_digits, train_label.ravel())
#predicting the data
classifier.fit(train_digits, train_label.ravel())
y_pred = classifier.predict(test_digits)
x_pred=classifier.predict(train_digits)
cm = confusion_matrix(test_label, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(test_label, y_pred)))
print('Accuracy' + str(accuracy_score(train_label, x_pred)))

df = pd.read_excel (r'önemliLDA.xlsx')
plt.scatter(df['components'], df['train eror'])
plt.xlabel('Number of Components')
plt.ylabel('Training Error')
plt.show()
plt.scatter(df['components'], df['test error'])
plt.xlabel('Number of Components')
plt.ylabel('Testing Error')
plt.show()


