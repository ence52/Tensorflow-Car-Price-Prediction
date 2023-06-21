import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
carsDf=pd.read_excel("merc.xlsx")
carsDf=carsDf.drop("transmission",axis=1)
#Describe dataframe
print(carsDf.describe())
print("*"*100)

fig,axes=plt.subplots(nrows=3,ncols=2)

#Graphical analysis for price
sbn.distplot(carsDf["price"],ax=axes[0,0])
axes[0,0].set_title("Price DistPlot")
sbn.scatterplot(x="mileage",y="price",data=carsDf,ax=axes[0,1])
axes[0,1].set_title("Price ScatterPlot")

#Correlation 
print(carsDf.corr())
print("*"*100)
print(carsDf.corr()["price"].sort_values())

#Detecting unusually expensive cars and droping them
print("*"*100)
carsDf=carsDf.sort_values("price",ascending=False).iloc[131:]
print(carsDf)
#After dropping expensives data
sbn.distplot(carsDf["price"],ax=axes[1,0])
axes[1,0].set_title("Price distplot (dropped)")
#Dropping weird data
carsDf= carsDf[carsDf.year!=1970]
print(carsDf.count())



#---------------------------------------------------------CREATING MODEL---------------------------------------------------------

#Define x and y 
y=carsDf["price"].values
x=carsDf.drop("price",axis=1).values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)

#Scaling data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Preparing the model
model = Sequential()
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x=x_train, y=y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300)

#Results
lossData=pd.DataFrame(model.history.history)
lossData.plot(ax=axes[1,1])
axes[1,1].set_title("Loss")

# Prediction 
prediction=model.predict(x_test)

#Determine Mean Absolute Error
print(f"Mean absolute error:  {mean_absolute_error(y_test,prediction)}")
plt.scatter(y_test,prediction)
axes[2,1].set_title("y_test and prediction scatterplot")

#Dropping a car from data and trying to predict its price
print("*"*100)
print(carsDf.iloc[3])
newCar = carsDf.drop("price",axis=1).iloc[3]
newCar=scaler.transform(newCar.values.reshape(-1,5))
print(f"Predicted price: {model.predict(newCar)}")
print("*"*100)

fig.tight_layout()
plt.show()