#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create histogram plot
plt.hist(student_grades, bins=10, edgecolor='black')

# Setup title and X and Y label
plt.title('Project A')
plt.xlabel('Grades')
plt.ylabel('Number of Students')

# Setup X lim and ticks
plt.xlim(0)
plt.xticks(np.arange(0, 101, 10))

# Setup Y lim
plt.ylim(0, 30)

plt.show()
