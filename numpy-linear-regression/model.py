import pandas as pd
import numpy as np

df = pd.read_csv('data/winequality_red.csv', sep=',', header=None)
print(df.values)