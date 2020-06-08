import numpy as np
from sklearn.utils import shuffle

Y = np.array([0]*100 + [1]*100)
y_values = shuffle(Y)
print("y_values : \n",y_values)
boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
print("boundaries (Array elements index where there is a change to the next element): ",boundaries)
