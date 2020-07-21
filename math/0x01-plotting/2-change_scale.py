#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# Set title and X and Y label
plt.title('Exponential Decay of C-14')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')

# Create plot
plt.plot(x, y)

# Setup Y scale
plt.yscale('log')

# Setup X limit from 0 to 28650
plt.xlim(0, 28650)

plt.show()
