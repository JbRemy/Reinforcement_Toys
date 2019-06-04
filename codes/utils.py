import io
import os
from IPython.display import HTML
import numpy as np
import sys

def ipython_info():
    ip = False
    if 'ipykernel' in sys.modules:
        ip = 'notebook'
    elif 'IPython' in sys.modules:
        ip = 'terminal'
    return ip

def rolling_average(serie, T):
    rolling_average = []
    for _ in range(T,len(serie)):
        rolling_average.append(np.mean(serie[_-T:_])) 
    
    return rolling_average




