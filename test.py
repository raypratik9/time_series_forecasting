import pandas as pd
import csv
from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from matplotlib import style
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 10}) 
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)



from sklearn.metrics import mean_squared_error
local_path='D:\\3rd_year\\Objective1\\pythonscript\\'
#data_xls = pd.read_excel(local_path+'AirQualityUCI2.xlsx', 'AirQualityUCI', index_col=None)
#data_xls.to_csv(local_path+'tempfile1.csv', encoding='utf-8')

print("hhhh")
series = pd.read_csv(local_path+'tempfile1.csv',encoding='utf-8', header=0, index_col=0)
X = series['CO(GT)']
#print(X)
"""with open(local_path+'tempfile2.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     writer.writerow(["'Data'","'Value'"])
     j = 0   
     for i in X:
        writer.writerow(["'str(value)'",X[j]])
        j=j+1   
"""              

series = pd.read_csv(local_path+'tempfile1.csv',encoding='utf-8', header=0, index_col=0)
X = series.values
print(X)
print (series['CO(GT)'])

#Missing values or outliers are found out and replaced with the medians

print (series['CO(GT)'].isnull())
missing_values=["n/a","na","-200"]
series = read_csv("tempfile1.csv", na_values=missing_values)
print (series['CO(GT)'].isnull())
median = series['CO(GT)'].median()
print("median:"," ", median)
series['CO(GT)'].fillna(median, inplace=True)
print(series['CO(GT)'])
X = series.values
print('Here')
print(X)
y = series['CO(GT)']
print("hellllo")

my_list = []

for i in range(len(y)):
    my_list.append(y[i])
    
#print(my_list)

train, test = my_list[1:len(X)-100], my_list[len(X)-100:]
test1= test
k=1
# using list comprehension 
# adding K to each element 
test2 = [x + k for x in test1] 

#print(train)
#6
#25

# train autoregression
### AR Model#####
model1 = AR(train)
model_fit1 = model1.fit()
print('Lag: %s' % model_fit1.k_ar)
print('Coefficients: %s' % model_fit1.params)
# make predictions
predictions1 = model_fit1.predict(start=len(train), end=len(train)+len(test2)-1, dynamic=False)
for i in range(len(predictions1)):
	print('predicted=%f, expected=%f' % (predictions1[i], test2[i]))
error1 = mean_squared_error(test2, predictions1)
print('Test MSE in AR: %.3f' % error1)

# ARIMA model
# train autoregression
model2 = ARIMA(train, order=(10,1,1))
model_fit2 = model2.fit()
print('Lag: %s' % model_fit2.k_ar)
print('Coefficients: %s' % model_fit2.params)
# make predictions
predictions2 = model_fit2.predict(start=len(train), end=len(train)+len(test2)-1, dynamic=False)
for i in range(len(predictions2)):
	print('predicted=%f, expected=%f' % (predictions2[i], test2[i]))
error2 = mean_squared_error(test2, predictions2)
print('Test MSE ARIMA: %.3f' % error2)


#MA model
# train autoregression
model3 = ARMA(train,order=(0,1))
model_fit3 = model3.fit()
print('Lag: %s' % model_fit3.k_ar)
print('Coefficients: %s' % model_fit3.params)
# make predictions
predictions3 = model_fit3.predict(start=len(train), end=len(train)+len(test2)-1, dynamic=False)
for i in range(len(predictions3)):
	print('predicted=%f, expected=%f' % (predictions3[i], test2[i]))
error3 = mean_squared_error(test2, predictions3)
print('Test MSE in MA: %.3f' % error3)
# plot results
#pyplot.plot(test)

#ARMA model
# train autoregression
#for k in range(5)
model4 = ARMA(train, order=(2,1))
model_fit4 = model4.fit()
print('Lag: %s' % model_fit4.k_ar)
print('Coefficients: %s' % model_fit4.params)
# make predictions
predictions4 = model_fit4.predict(start=len(train), end=len(train)+len(test2)-1, dynamic=False)
for i in range(len(predictions4)):
	print('predicted=%f, expected=%f' % (predictions4[i], test2[i]))
error4 = mean_squared_error(test2, predictions4)
print('Test MSE in ARMA: %.3f' % error4)
# plot results
pyplot.grid()
pyplot.plot(test2, linestyle='--', label ='Expected Value', linewidth= 2)
pyplot.plot(predictions1 ,color='red',marker="o", label='AR Model', linewidth= 2)
pyplot.plot(predictions2, color='green', marker="^", label='ARIMA Model', linewidth= 2)
pyplot.plot(predictions3, color='blue', marker="+", label='MA Model', linewidth= 2)
pyplot.plot(predictions4, color='black', marker="v", label='ARMA Model', linewidth= 2)
pyplot.ylabel('CO Index')
pyplot.xlabel('Observations')
pyplot.ylim(-2,6)
pyplot.xlim(0,100)
pyplot.legend()

#xlim(right=10)
#xlim(left=1)
#yplot.show()
pyplot.savefig("out13.png")


