#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# Create plot
plt.plot(y, 'r')

# Set limit from 0 to 10
plt.xlim(0, 10)

plt.show()
