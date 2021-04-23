# Lab 6

## Question 1: A tree with no depth restriction (an infinite maximum depth) will always attempt to create only pure nodes. True/False
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

## Question 2: Classification Tree models can be helpful if your data illustrates non-linearity. True/False
This is true - this is exactly what we want to do!

## Question 3: In the RandomForestClassifier, what does the n_estimators define?
according to the write up of RandomForestClassifier, n_estimators is 'the number of trees in the forest' so the correct answer would be 'the number of trees to include in the forest.'

## Question 4: In the lab, based on the test dataset, which model predicted the most false negatives? Use a max depth of 5 for your trees, and the random state = 1693. For the random forest classifier, n_estimators should be 1000.
Using the code from Lecture 25, I just altered the code a bit to select the answer of Naive Bayers. However, I believe I used the wrong X and Y data when doing this problem, so my answer was wrong.

## Question 5: In the lab, what would be considered a "False Positive"?
This is correct because a malignant tumor would be a 'positive' test result and if a system identified a tumor as malignant when it actually was benign, then it is a false positive.

## Question 6: Naive Bayes classifiers are probabilistic. True/False
According to our notes and lecture, we learned that the Naive Bayes classifier is a member of the family of 'probabilistic classifiers.'

## Question 7: The root node of a classification tree is...
Based on our notes, it is a node which contains all data points.

## Question 8: What is soft classification?
A probability-based classification, where an observation is assigned a probability of class membership for each class - this is its definition!.

## Question 9: What is the Posterior Probability in a Bayesian approach to classification?
The probability you are solving for, such as the the probability that the response variable belongs to a particular class given the data. It is the probability we are solving for each class!

## Question 10: In a two-class classification the confusion matrix helps with determining
When you have two classes, you are able to see true positives and negatives as well as false positives and negatives. This is the correct answer as a result.

## Question 11: The biological function of the "Axon" is represented by what element of a neural network?
When looking at a brain and functions of its parts, it is clear that axons work as outputs from each neuron along the synapse.

## Question 12: Match each term to it's correct definition, in the context of a neural network.
Neuron -            Node in which an activation function is applied.
Hidden Layer -      A set of neurons that are applied at the same time, going left to right in the network.
Cost Function -     Measurement of how close predictions are to true values.
Back propagation -  Information on accuracy being used to adjust weights or other model paramters
Gradient Descent -  Method for reducing cost function to a minimum.

## Question 13: What is an advantage of stochastic gradient descent, as contrasted to traditional gradient descent?
From Lecture 26, we learned that for determining the optimal weights for the neurons (which means training the network) we have to use an efficient optimization algorithm that is not going to get “stuck” in a sub-optimal local minima. A promising choice for this challenging situation is the Stochastic Gradient Descent algorithm which for computing the vector for updating the weights will use a random subset of observations at a time. So, the correct answer is 'it is less likely to get stuck at suboptimal local minima'

## Question 14: If we retain only two input features, such as "mean radius" and "mean texture" and apply the Gaussian Naive Bayes model for classification, then the average accuracy determined on a 10-fold cross validation with random_state = 1693 is (do not use the % notation, just copy the first 4 decimals)
```markdown
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as CM
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

def validation(X,y,k,model):
  AC = []
  pipe = Pipeline([('scale',scale),('Classifier',model)])
  kf = KFold(n_splits=k,shuffle=True,random_state=1693)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test = X[idxtest,:]
    y_test = y[idxtest]
    pipe.fit(X_train,y_train)
    AC.append(acc(ytest,model.predict(Xstest)))
  return np.mean(AC)
  
dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)

df = df.drop(columns=['mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error','concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'])

X = df.values
y = dat.target

model = GaussianNB()
scale = StandardScaler()

validation(X,y,10,model)
```
As a result, we see that the answer is 0.89821.

## Question 15: From the data retain only two input features, such as "mean radius" and "mean texture" and apply the Random Froest model for classification with 100 trees, max depth of 7 and random_state=1693; The average accuracy determined on a 10-fold cross validation with the same random state is (do not use the % notation, just copy the first 4 decimals)
```markdown
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as CM
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

def validation(X,y,k,model):
  AC = []
  pipe = Pipeline([('scale',scale),('Classifier',model)])
  kf = KFold(n_splits=k,shuffle=True,random_state=1693)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test = X[idxtest,:]
    y_test = y[idxtest]
    pipe.fit(X_train,y_train)
    AC.append(acc(ytest,model.predict(Xstest)))
  return np.mean(AC)
  
dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)

df = df.drop(columns=['mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error','concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'])

X = df.values
y = dat.target

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=1693, max_depth=7, n_estimators = 100)
scale = StandardScaler()

validation(X,y,10,model)
```
As a result, we see that the answer is 0.9321.

## Question 16: From the data retain only two input features, such as "mean radius" and "mean texture" we build an Artificial Neural Network for classification that has three hidden layers with 16, 8 and 4 neurons respectively. Assume that the neurons in the hidden layer have the rectified linear activation ('relu') and the kernel initializer uses the random normal distribution. Assume the output layer has only one neuron with 'sigmoid' activation. You will compile the model with model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) and fit the model with 150 epochs and validation_split=0.25,batch_size=10,shuffle=False. The average accuracy determined on a 10-fold cross validation (random state=1693) is closer to
```markdown
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score as acc

model = Sequential()

model.add(Dense(16,kernel_initializer='random_normal', input_dim=2, activation='relu')) 
model.add(Dense(8,kernel_initializer='random_normal', activation='relu'))
model.add(Dense(4,kernel_initializer='random_normal', activation='relu'))

model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

from sklearn.model_selection import KFold
kf = KFold(n_splits=10,shuffle=True,random_state=1693)

AC = []
for idxtrain, idxtest in kf.split(X):
  Xtrain = X[idxtrain,:]
  Xtest  = X[idxtest,:]
  ytrain = y[idxtrain]
  ytest  = y[idxtest]
  Xstrain = scale.fit_transform(Xtrain)
  Xstest  = scale.transform(Xtest)
  model.fit(Xstrain, ytrain, epochs=150, validation_split=0.25,batch_size=10, shuffle=False)
  AC.append(acc(ytest,model.predict_classes(Xstest)))
  print(acc(ytest,model.predict_classes(Xstest)))

import numpy as np
np.mean(AC)
```
As a result, we see that the answer is 0.8928.
