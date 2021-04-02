# Filling large gaps in time series using forecasting

## Context

This notebook has the pourpose to show an easy approach to fill large gaps in time series, mantainign a certain veridicity and data validity. The approach consist in apply a forecasting in both sides of the gap, and combine the two prediction using interpolation. As shown in the specific case of this notebook, we can apply this approach also to attach consequential files, concerning same data entity. Moreover this specific case we are going to use esponential smoothing with seasonality, indulging the characteristic of our time series, but the approach can obviously be repeated with the most suitable forecasting tecnique for each single case.

## Libraries 




```python
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
plt.rcParams['figure.dpi']= 300  #resolution
```

## Data

Data are divided into three separated text files. Every dataset is composed by the same columns.
Surveys have been recorded minutes by minutes, and there are no missing rows. 
The dirst columns of the txt files contains the date time, that is parsed by Pandas and used as index. In fact is important to have the index weel formatted.



```python
df1 = pd.read_csv('datatest.txt',index_col=1,parse_dates=True)
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Light</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
      <th>Occupancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-02-02 14:19:00</th>
      <td>140</td>
      <td>23.7000</td>
      <td>26.272</td>
      <td>585.200000</td>
      <td>749.200000</td>
      <td>0.004764</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-02-02 14:19:59</th>
      <td>141</td>
      <td>23.7180</td>
      <td>26.290</td>
      <td>578.400000</td>
      <td>760.400000</td>
      <td>0.004773</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-02-02 14:21:00</th>
      <td>142</td>
      <td>23.7300</td>
      <td>26.230</td>
      <td>572.666667</td>
      <td>769.666667</td>
      <td>0.004765</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-02-02 14:22:00</th>
      <td>143</td>
      <td>23.7225</td>
      <td>26.125</td>
      <td>493.750000</td>
      <td>774.750000</td>
      <td>0.004744</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-02-02 14:23:00</th>
      <td>144</td>
      <td>23.7540</td>
      <td>26.200</td>
      <td>488.600000</td>
      <td>779.000000</td>
      <td>0.004767</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.index
```




    DatetimeIndex(['2015-02-02 14:19:00', '2015-02-02 14:19:59',
                   '2015-02-02 14:21:00', '2015-02-02 14:22:00',
                   '2015-02-02 14:23:00', '2015-02-02 14:23:59',
                   '2015-02-02 14:25:00', '2015-02-02 14:25:59',
                   '2015-02-02 14:26:59', '2015-02-02 14:28:00',
                   ...
                   '2015-02-04 10:34:00', '2015-02-04 10:34:59',
                   '2015-02-04 10:36:00', '2015-02-04 10:37:00',
                   '2015-02-04 10:38:00', '2015-02-04 10:38:59',
                   '2015-02-04 10:40:00', '2015-02-04 10:40:59',
                   '2015-02-04 10:41:59', '2015-02-04 10:43:00'],
                  dtype='datetime64[ns]', length=2665, freq=None)



Let's procede in cleaning our data... First and last id of the index are used to create an ordered index.


```python
#First dataset
df1 = pd.read_csv('datatest.txt',index_col=1,parse_dates=True)
df1.set_index(pd.date_range(start=df1.index[0], end=df1.index[-1], freq='min'), inplace=True)
df1.drop(['date','Occupancy','Light'], axis=1, inplace=True)

#Second dataset
df2=pd.read_csv(r'datatraining.txt', index_col=1,parse_dates=True)
df2.set_index(pd.date_range(start=df2.index[0], end=df2.index[-1], freq='min'), inplace=True)
df2.drop(['date','Occupancy','Light'], axis=1, inplace=True)

#Third dataset
df3= pd.read_csv('datatest2.txt',index_col=1,parse_dates=True)
df3.set_index(pd.date_range(start=df3.index[0], end=df3.index[-1], freq='min'), inplace=True)
df3.drop(['date','Occupancy','Light'], axis=1, inplace=True)

df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-02-02 14:19:00</th>
      <td>23.7000</td>
      <td>26.272</td>
      <td>749.200000</td>
      <td>0.004764</td>
    </tr>
    <tr>
      <th>2015-02-02 14:20:00</th>
      <td>23.7180</td>
      <td>26.290</td>
      <td>760.400000</td>
      <td>0.004773</td>
    </tr>
    <tr>
      <th>2015-02-02 14:21:00</th>
      <td>23.7300</td>
      <td>26.230</td>
      <td>769.666667</td>
      <td>0.004765</td>
    </tr>
    <tr>
      <th>2015-02-02 14:22:00</th>
      <td>23.7225</td>
      <td>26.125</td>
      <td>774.750000</td>
      <td>0.004744</td>
    </tr>
    <tr>
      <th>2015-02-02 14:23:00</th>
      <td>23.7540</td>
      <td>26.200</td>
      <td>779.000000</td>
      <td>0.004767</td>
    </tr>
  </tbody>
</table>
</div>




```python

for i,col in enumerate(df1.columns):
    #plt.subplot(2,2,i+1)
    plt.plot(df1[col], label='ts1')
    plt.plot(df2[col], label='ts2')
    plt.plot(df3[col], label='ts3')
    plt.ylabel(col)
    plt.xticks(rotation = 30)
    #plt.legend()
    plt.show()
    


```


    
![png](output_7_0.png)
    



    
![png](output_7_1.png)
    



    
![png](output_7_2.png)
    



    
![png](output_7_3.png)
    


## Forecasting

Now we are ready to start with the forecasting. Example id given using *Humidity* column.
The implementation is provided by the library *statsmodel*. Method used is exponential smoothing with seasonality to cover daily periods. Records in a day are sixty (minutes in an hour) times 24 (hours), and for this reason season periods are of 1440.


```python
ts1 = df1['Humidity'].copy()
ts2 = df2['Humidity'].copy()
ts3 = df3['Humidity'].copy()

seasonal_periods = 60 * 24

one = timedelta(minutes=1)
```

### Forward 
Here the selected forecasting tecnique is applied in a standard way, fitting it in the second time series in order to fill the hole betweeen itself and the third one time series.


```python

from statsmodels.tsa.holtwinters import ExponentialSmoothing

es = ExponentialSmoothing(ts2,  seasonal_periods=seasonal_periods,seasonal='add').fit()
pred23 = es.predict(start=ts2.index[-1]+one, end=ts3.index[0]-one)

```

    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:429: FutureWarning: After 0.13 initialization must be handled at model creation
      FutureWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    

### Backward 

While here time series are reverted, in order to apply the forecasting in the opposite side. In order to make it working, till the forecasting is made to works only forward (that's makes sense), it's importatn to reverse also all index... so we need to understand also how many record consist in teh gap ot fill it.



```python
ts3r = ts3[::-1]
ts2r = ts2[::-1]

indexr = pd.date_range(start=ts2.index[0], end=ts3.index[-1], freq='min')
ts2r.index = indexr[-len(ts2r):]
ts3r.index = indexr[:len(ts3r)]
```


```python


es = ExponentialSmoothing(ts3r,  seasonal_periods=seasonal_periods,seasonal='add').fit()


```

    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    


```python
pred32 = es.predict(start=ts3r.index[-1]+one, end=ts2r.index[0]-one)
```


```python
pred32.index = pred23.index.copy()
```

### Interpolation 

Now we have the predicitons in the two directions, only needs to be interpolated. Result are shown below.


```python
l = len(pred23)
pred = pd.Series([(pred32[i] * i + pred23[i] * (l -i) )/ l for i in range(l)], index=pred23.index.copy())

```


```python
fig = plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.plot(ts2, color='C1', label='from dataset')
plt.plot(ts3, color='C1')
plt.plot(pred23, color='C4', label='predicted')
plt.title('Forward')
plt.legend()
plt.xticks(rotation = 30) 

plt.subplot(1,3,2)
plt.plot(ts2,  color='C1', label='from dataset')
plt.plot(ts3, color='C1')
plt.plot(pred32, color='C4', label='predicted')
plt.title('Backward')
plt.legend()
plt.xticks(rotation = 30) 

plt.subplot(1,3,3)
plt.plot(ts2,  color='C1', label='from dataset')
plt.plot(ts3,  color='C1')
plt.plot(pred, color='C4', label='predicted')
plt.title('Interpolation')
plt.legend()
plt.xticks(rotation = 30) 


plt.show()
```


    
![png](output_19_0.png)
    


## The method in a function 
Here is defined a function that given two time series returns the time series between them;
wi will apply it on the first gap .


```python
def fillgap(firstTS, secondTS, seasonal_periods = 60 * 24):
    
    #PREPARATION
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    one = timedelta(minutes=1)
    secondTSr = secondTS[::-1].copy()
    firstTSr = firstTS[::-1].copy()
    indexr = pd.date_range(start=firstTS.index[0], end=secondTS.index[-1], freq='min')
    firstTSr.index = indexr[-len(firstTSr):]
    secondTSr.index = indexr[:len(secondTSr)]
    
    #FORWARD    
    es = ExponentialSmoothing(firstTS,  seasonal_periods=seasonal_periods,seasonal='add').fit()
    forwardPrediction = es.predict(start=firstTS.index[-1]+one, end=secondTS.index[0]-one)
    
    #BACKWARD
    es = ExponentialSmoothing(secondTSr,  seasonal_periods=seasonal_periods,seasonal='add').fit()
    backwardPrediction = es.predict(start=secondTSr.index[-1]+one, end=firstTSr.index[0]-one)
    
    #INTERPOLATION
    l = len(forwardPrediction)
    interpolation = pd.Series([(backwardPrediction[i] * i + forwardPrediction[i] * (l -i) )/ l for i in range(l)], index=forwardPrediction.index.copy())
  
    return interpolation


```


```python
ts1 = df1['Humidity'].copy()
ts2 = df2['Humidity'].copy()
gap = fillgap(ts1,ts2)
```

    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    


```python
ts = pd.concat([ts1,gap,ts2,pred,ts3])

plt.plot(ts, color='C1', label='from dataset')
plt.plot(gap, color='C4')
plt.plot(pred, color='C4', label='predicted')
plt.ylabel('Humidity')
plt.title('Prediction')
plt.legend()
plt.xticks(rotation = 20)

plt.show()
```


    
![png](output_23_0.png)
    


## Application and resolution in the dataset 
Here the mothos is apply fro the two gaps and for each column of the dataset, in order to creating a new one long dataset without interruption.


```python
#First dataset
df1 = pd.read_csv('datatest.txt',index_col=1,parse_dates=True)
df1.set_index(pd.date_range(start=df1.index[0], end=df1.index[-1], freq='min'), inplace=True)
df1.drop(['date','Occupancy','Light'], axis=1, inplace=True)

#Second dataset
df2=pd.read_csv(r'datatraining.txt', index_col=1,parse_dates=True)
df2.set_index(pd.date_range(start=df2.index[0], end=df2.index[-1], freq='min'), inplace=True)
df2.drop(['date','Occupancy','Light'], axis=1, inplace=True)

#Third dataset
df3= pd.read_csv('datatest2.txt',index_col=1,parse_dates=True)
df3.set_index(pd.date_range(start=df3.index[0], end=df3.index[-1], freq='min'), inplace=True)
df3.drop(['date','Occupancy','Light'], axis=1, inplace=True)

df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-02-02 14:19:00</th>
      <td>23.7000</td>
      <td>26.272</td>
      <td>749.200000</td>
      <td>0.004764</td>
    </tr>
    <tr>
      <th>2015-02-02 14:20:00</th>
      <td>23.7180</td>
      <td>26.290</td>
      <td>760.400000</td>
      <td>0.004773</td>
    </tr>
    <tr>
      <th>2015-02-02 14:21:00</th>
      <td>23.7300</td>
      <td>26.230</td>
      <td>769.666667</td>
      <td>0.004765</td>
    </tr>
    <tr>
      <th>2015-02-02 14:22:00</th>
      <td>23.7225</td>
      <td>26.125</td>
      <td>774.750000</td>
      <td>0.004744</td>
    </tr>
    <tr>
      <th>2015-02-02 14:23:00</th>
      <td>23.7540</td>
      <td>26.200</td>
      <td>779.000000</td>
      <td>0.004767</td>
    </tr>
  </tbody>
</table>
</div>




```python
gap1 = pd.DataFrame()
gap2 = pd.DataFrame()

for col in df1.columns:
    ts1 = df1[col].copy()
    ts2 = df2[col].copy()
    ts3 = df3[col].copy()
    gap1[col] = fillgap(ts1,ts2)
    gap2[col] = fillgap(ts2,ts3)
    
```

    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:429: FutureWarning: After 0.13 initialization must be handled at model creation
      FutureWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    C:\Users\carlo\Anaconda3\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    


```python
df = pd.DataFrame(columns = [col for col in df1.columns])
df = df.append(df1)
df = df.append(gap1)
df = df.append(df2)
df = df.append(gap2)
df = df.append(df3)

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>CO2</th>
      <th>HumidityRatio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-02-02 14:19:00</th>
      <td>23.7000</td>
      <td>26.2720</td>
      <td>749.200000</td>
      <td>0.004764</td>
    </tr>
    <tr>
      <th>2015-02-02 14:20:00</th>
      <td>23.7180</td>
      <td>26.2900</td>
      <td>760.400000</td>
      <td>0.004773</td>
    </tr>
    <tr>
      <th>2015-02-02 14:21:00</th>
      <td>23.7300</td>
      <td>26.2300</td>
      <td>769.666667</td>
      <td>0.004765</td>
    </tr>
    <tr>
      <th>2015-02-02 14:22:00</th>
      <td>23.7225</td>
      <td>26.1250</td>
      <td>774.750000</td>
      <td>0.004744</td>
    </tr>
    <tr>
      <th>2015-02-02 14:23:00</th>
      <td>23.7540</td>
      <td>26.2000</td>
      <td>779.000000</td>
      <td>0.004767</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2015-02-18 09:15:00</th>
      <td>20.8150</td>
      <td>27.7175</td>
      <td>1505.250000</td>
      <td>0.004213</td>
    </tr>
    <tr>
      <th>2015-02-18 09:16:00</th>
      <td>20.8650</td>
      <td>27.7450</td>
      <td>1514.500000</td>
      <td>0.004230</td>
    </tr>
    <tr>
      <th>2015-02-18 09:17:00</th>
      <td>20.8900</td>
      <td>27.7450</td>
      <td>1521.500000</td>
      <td>0.004237</td>
    </tr>
    <tr>
      <th>2015-02-18 09:18:00</th>
      <td>20.8900</td>
      <td>28.0225</td>
      <td>1632.000000</td>
      <td>0.004279</td>
    </tr>
    <tr>
      <th>2015-02-18 09:19:00</th>
      <td>21.0000</td>
      <td>28.1000</td>
      <td>1864.000000</td>
      <td>0.004321</td>
    </tr>
  </tbody>
</table>
<p>22741 rows × 4 columns</p>
</div>




```python
for i,col in enumerate(df.columns):
    plt.plot(df[col],  color='C1', label='from dataset')
    plt.plot(gap1[col], color='C4')
    plt.plot(gap2[col], color='C4', label='predicted')
    plt.ylabel(col)
    plt.legend()
    plt.xticks(rotation = 20) 
    plt.show()
```


    
![png](output_28_0.png)
    



    
![png](output_28_1.png)
    



    
![png](output_28_2.png)
    



    
![png](output_28_3.png)
    



```python
df.to_csv('ts_with-gaps.csv')
df1.to_csv('ts1.csv')
df2.to_csv('ts2.csv')
df3.to_csv('ts3.csv')
gap1.to_csv('gap1.csv')
gap2.to_csv('gap2.csv')
```

# Thanks for attention
