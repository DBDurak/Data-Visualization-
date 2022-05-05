from cgi import test
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import matplotlib.image as mpimg
import numpy as np
import seaborn as sn
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection._split import train_test_split
import random

#loading data
data = loadmat('digitss.mat')

#splitting data
train_digits,test_digits,train_label,test_label=train_test_split(data['digits'],data['labels'], test_size=0.5,random_state=42,shuffle=True)
#PCA calculations
x = random.randint(1,200)
print(x)
pca = PCA(n_components=x)
trained = pca.fit(train_digits)

#eigenvalues and eigenvectors of the PCA 
eigenvalues,eigenvectors = np.linalg.eig(trained.get_covariance())
eigenvalues = np.real(eigenvalues)

#plotting eigenvalues
plt.plot(eigenvalues)
plt.title('Eigenvalues of the PCA')
plt.xlabel('Components')
plt.ylabel('Eigenvalues')
plt.show()

#plotting sample mean
samplemean = np.mean(train_digits,axis=0)
img=mpimg.imsave('samplemean.png',samplemean.reshape(20,20))
img = mpimg.imread('samplemean.png')
imgplot = plt.imshow(img)
plt.show()

#plotting eigenvectors
eigenvectors=np.real(eigenvectors)
show_vectors=np.dot(trained.components_,eigenvectors)
vimg=mpimg.imsave('show_vectors.png',show_vectors.reshape(400,x))
vimg = mpimg.imread('show_vectors.png',show_vectors.reshape(400,x))
vimgplot = plt.imshow(vimg)
plt.show()

#training the model
train_transform = trained.transform(train_digits)
test_transform = trained.transform(test_digits)
print(train_label.ravel())
kernel = 4.0 * RBF(4.0)

#Gaussian Process Classifier
classifier = GaussianProcessClassifier(kernel=kernel,random_state=0, n_jobs = -1).fit(train_transform, train_label.ravel())
#predicting the test data
print(classifier.score(train_transform, train_label.ravel()))
print(classifier.score(test_transform, test_label.ravel()))

#printing the error plots for the test data and training data from excell file using pandas
df = pd.read_excel (r'önemli data.xlsx')
print (df)
plt.scatter(df['comp'], df['traerror'])
plt.xlabel('Number of Components')
plt.ylabel('Training Error')
plt.show()
plt.scatter(df['comp'], df['teeror'])
plt.xlabel('Number of Components')
plt.ylabel('Testing Error')
plt.show()
