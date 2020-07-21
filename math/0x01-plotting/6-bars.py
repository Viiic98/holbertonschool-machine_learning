#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Create figure and the subplot that's going to contain the bars
fig, ax = plt.subplots()

# Set title of the plot and Y label of subplot
plt.title('Number of Fruit per Person')
ax.set_ylabel('Quantity of Fruit')

# Set Y limit and set Y ticks to ten
plt.ylim(0, 80)
plt.yticks(np.arange(0, 90, 10))

# Name of every person
names = ["Farrah", "Fred", "Felicia"]

# Create every bar
plt.bar(names, fruit[0], width=0.5, align='center', color='r', label='apples')
plt.bar(names, fruit[1], width=0.5,
        align='center', color='#FFFF00',
        label='bananas', bottom=fruit[0])
plt.bar(names, fruit[2], width=0.5,
        align='center', color='#ff8000',
        label='oranges', bottom=fruit[0]+fruit[1])
plt.bar(names, fruit[3], width=0.5,
        align='center', color='#ffe5b4',
        label='peaches', bottom=fruit[0]+fruit[1]+fruit[2])

# Setup legends of fruits
# Handles is a list with every bar
# labels is a list with the label of every bar
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

plt.show()
