# Fibonacci-Sequence-using-lstm
Here we constuct LSTM model to predict next series
Following package needs to be installed
```
!pip install numpy
!pip install tensorflow
```

The graph between original data and predicted data is shown below:
```
plt.plot(index, timeseries_data, label='Training Series')
plt.plot(p_index, next_series, label='Predicted Series')
plt.xlabel("index")
plt.ylabel("value")
plt.legend(loc='upper left')
```
![download (1)](https://user-images.githubusercontent.com/45820805/201526588-4b5a5ae1-90a2-42dd-89d6-7daed1ec8400.png)


