import numpy as np
from scipy import stats
import math

# Function to compute the 95% confidence interval
def confidence_interval(data, confidence=95):
    data_array = np.array(data)
    mean = np.mean(data_array)
    n = len(data_array)
    std_err = stats.sem(data_array)  # Standard error of the mean
    if confidence==95:
      Z=1.960
    elif confidence==99:
      Z=2.576
    else:
      print("wrong confidence level")
      exit()
    margin_of_error = Z * std_err/math.sqrt(n)


    return margin_of_error
