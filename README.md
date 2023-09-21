# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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
![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/9ba9cd42-aacc-4be0-ac80-30cdb6a1c084)

![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/0b004b5d-e7c2-42ee-b40a-0aa6e9f71095)

![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/c07925ac-8977-4e0e-9786-af299bc7562d)

![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/e6690b3a-d84d-4928-b4e1-d1930d2b2d16)

![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/1def83c8-1968-40f3-a08d-75b439c25806)

![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/969dd66c-0a32-4e2c-840d-7734fbd78d1e)

![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/d955a448-dc79-4f2f-af4a-4e543217542c)

![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/c80f230c-0cbc-43cb-8e53-5ab901633198)

![image](https://github.com/Sachin-vlr/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113497666/90d378f1-15b0-47aa-8419-a19b60890bb8)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

