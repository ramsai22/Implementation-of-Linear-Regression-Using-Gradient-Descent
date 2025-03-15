# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations of gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
Program to implement the linear regression using gradient descent.
Developed by:Paida Ram Sai
RegisterNumber:212223110034
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01, num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(x).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
x=(data.iloc[1:,:-2].values)
print(x)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_Scaled=scaler.fit_transform(x1)
y1_Scaled=scaler.fit_transform(y)
print(x1_Scaled)
print(y1_Scaled)
theta=linear_regression(x1_Scaled,y1_Scaled)
new_data=np.array([165349.2,136897.8,471781.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")
```
## Output:
<img width="305" alt="image" src="https://github.com/user-attachments/assets/30999bee-d34c-438c-bb34-ab22046ed8d2" />


### X Values:

<img width="281" alt="image" src="https://github.com/user-attachments/assets/59fceb4a-f67d-4d03-9324-a7b519fe0c77" />


### Y Values:

<img width="254" alt="image" src="https://github.com/user-attachments/assets/47cae3bc-d51e-4eda-8485-520a1c884176" />

### XScaled values:

<img width="397" alt="image" src="https://github.com/user-attachments/assets/32c9f58e-af48-440f-a453-c99e9160d274" />

### YScaled values:

<img width="140" alt="image" src="https://github.com/user-attachments/assets/e7f3866d-9593-4247-b26f-dc65d6889d3a" />

### Predicted Value:

<img width="379" alt="image" src="https://github.com/user-attachments/assets/0bae5b4a-1372-426a-bddd-4c10d50d22ff" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
