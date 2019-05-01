import numpy as np
import matplotlib.pyplot as plt
mat = np.random.random((10,10))
plt.imshow(mat, origin="lower", cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()