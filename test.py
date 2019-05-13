import os
import math
from collections import OrderedDict
import pandas as pd
import numpy as np
from collections import defaultdict
from tabulate import tabulate
import argparse

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colorbar, colors
from matplotlib.colors import Normalize

fig, ax = plt.subplots()

df = pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]), columns=['a', 'b'])

x = df['a'].values.tolist()
y = df.index.values.tolist()

df.plot(
    kind='barh',
    width=1,
    legend=False,
    ax=ax,
    color='red',
    zorder=2)
    #sharex=True,
    #sharey=True)

plt.scatter(x,y)
ax.step(
    x=x,
    y=y,
    zorder=2,
    color='magenta')

plt.show()
