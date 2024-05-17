import pandas as pd
file=pd.read_excel(r"C:\Users\Chandra\Desktop\coding\hours_score.ods")
inputs =file.loc[:,'hours'].values.reshape(-1,1)
outputs=file.loc[:,'score'].values.reshape(-1,1)
print(inputs)

#train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,outputs,test_size=0.2,random_state=2)
print("x_train shape:", len(x_train))
print("x_test shape:", len(x_test))
print("y_train shape:", len(y_train))
print("y_test shape:", len(y_test))

#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
poly_model =LinearRegression()
poly_model.fit(x_train, y_train)
y_predict=poly_model.predict(x_test)
print("values:",y_predict)

df = pd.DataFrame({'actual': y_test.flatten(), 'predict': y_predict.flatten()})
print(df)

print("mean squared error using linear regression:",mean_squared_error(y_test,y_predict))
print("R^2 Score:", r2_score(y_test,y_predict))

# Cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(LinearRegression(), inputs, outputs, cv=5, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
print("Cross-Validated MSE using cross validation:", mse_scores.mean())

#Standard Scaler
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)
lrs=LinearRegression().fit(x_train, y_train)
yps=lrs.predict(x_test)
print("mean squared error using standard scaler :",mean_squared_error(y_test,yps))
print("R^2 Score:", r2_score(y_test,yps))

import matplotlib.pyplot as plt
# Plotting the distribution of scores
plt.plot(inputs,outputs,'o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()
line = poly_model.coef_ * inputs + poly_model.intercept_
plt.scatter(inputs, outputs)
plt.plot(inputs, line)
plt.show()

import numpy as np
hours=9.25
o=poly_model.predict(np.array(hours).reshape(-1,1)).flatten()
print("hours=",hours)
print("predicted score=",o)
