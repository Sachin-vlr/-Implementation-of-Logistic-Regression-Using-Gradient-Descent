# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression

2.Set variables for assigning dataset values.
 
3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SACHIN.C
RegisterNumber: 212222230125
```

```PYTHON
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data =np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0], X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0], X[y==0][:,1],label="Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1/(1+np.exp(-z))
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()
def costFuction(theta, X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return j,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFuction(theta,X_train,y)
print(j)
print(grad)
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFuction(theta,X_train,y)
print(j)
print(grad)
def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
printf=(res.fun)
print(res.x)
def plotDecisionBoundary(theta, x, y):
  x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
  y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
  x_plot = np.c_[xx.ravel(), yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot, theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0], x[y==1][:,1], label= "admitted")
  plt.scatter(x[y==0][:,0], x[y==0][:,1], label= "not admitted")
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:

### Array value of X:
![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/9ba9cd42-aacc-4be0-ac80-30cdb6a1c084)

### Array value of Y:
![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/b295ede5-3a93-45e2-b4a0-532979d58a1b)

### Score graph:
![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/07100620-afaf-4f13-97d0-f75d22851b28)

### Sigmoid function:
![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/f5c0d695-6745-4e15-bd6f-135195e483b2)

### X_train_grad value:
![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/e9e660c0-4e32-4425-98d1-efd9d084bb5e)

### Y_train_grad value:
![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/9499bc2e-873a-4a56-96d8-bc180c8e08df)

### res.x:
![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/8976fefe-570c-4935-85da-c39e56a2b3c5)

### Decision boundry:

### Probability value:
![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/c909d5b9-3276-45a3-a068-ca91a8c41151)

### Probability value of mean:
![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/853a7f09-a40f-413f-b1d4-8bc9fbd65b86)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

