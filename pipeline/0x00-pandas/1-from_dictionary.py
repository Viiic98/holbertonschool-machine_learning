#!/usr/bin/env python3
""" From dictionary """
import pandas as pd

dic = {'First': [0.0, 0.5, 1.0, 1.5], 'Second': ['one', 'two',
                                                 'three', 'four']}
rows = ['A', 'B', 'C', 'D']
df = pd.DataFrame(dic, index=rows)
