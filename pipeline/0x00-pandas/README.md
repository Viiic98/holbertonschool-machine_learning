# Pandas

## Download Pandas 0.24.x
- pip install --user pandas

## Datasets
For this project, we will be using the [coinbase](https://drive.google.com/file/d/16MgiuBfQKzXPoWFWi2w-LKJuZ7LgivpE/view) and [bitstamp](https://drive.google.com/file/d/15A-rLSrfZ0td7muSrYHy0WX9ZqrMweES/view) datasets, as seen previously in 0x0E. Time Series Forecasting

## Tasks

### [From Numpy](./0-from_numpy.py)
- creates a pd.DataFrame from a np.ndarray

### [From Dictionary](./1-from_dictionary.py)
- python script that created a pd.DataFrame from a dictionary

### [From File](./2-from_file.py)
- loads data from a file as a pd.DataFrame

### [Rename](./3-rename.py)
- Rename the column Timestamp to Datetime
- Convert the timestamp values to datatime values
- Display only the Datetime and Close columns

### [To Numpy](./4-array.py)
- take the last 10 rows of the columns High and Close and convert them into a numpy.ndarray

### [Slice](./5-slice.py)
- slice the pd.DataFrame along the columns High, Low, Close, and Volume_BTC, taking every 60th row

### [Flip it and Switch it](./6-flip_switch.py)
- alter the pd.DataFrame such that the rows and columns are transposed and the data is sorted in reverse chronological order

### [Sort](./7-high.py)
- sort the pd.DataFrame by the High price in descending order

### [Prune](./8-prune.py)
- remove the entries in the pd.DataFrame where Close is NaN

### [Fill](./9-fill.py)
- fill in the missing data points in the pd.DataFrame

### [Indexing](./10-index.py)
- index the pd.DataFrame on the Timestamp column

### [Concat](./11-concat.py)
- index the pd.DataFrames on the Timestamp columns and concatenate them

### [Hierarchy](./12-hierarchy.py)
- rearrange the MultiIndex levels such that timestamp is the first level

### [Analyze](./13-analyze.py)
- calculate descriptive statistics for all columns in pd.DataFrame except Timestamp

### [Visualize](./14-visualize.py)
- visualize the pd.DataFrame
