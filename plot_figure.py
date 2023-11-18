import matplotlib.pyplot as plt
import numpy as np
x = list(range(1, 7))
fig, axes = plt.subplots(1, 2)

ax1 = axes[0]
ax2 = axes[1]
ax1.scatter(x, x, c=[1,2])
ax2.scatter(x, x, s=30, c=x)
plt.show()