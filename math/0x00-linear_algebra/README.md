# LINEAR ALGEBRA

![Matrix](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/54daaf81421a9b894688.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210213%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210213T021449Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=015e7e475e9163980fa50c1f95aeaa03ae602f16e81941d24c0a40a43b6df953)

## INSTALLING GUIDES

### Installing pip 19.1
```
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
rm get-pip.py
```
To check that pip has been successfully downloaded, use pip -V. Your output should look like:
```
pip -V
pip 19.1.1 from /usr/local/lib/python3.5/dist-packages/pip (python 3.5)
```

### Installing numpy 1.15, scipy 1.3, and pycodestyle 2.5
```
pip install --user numpy==1.15
pip install --user scipy==1.3
pip install --user pycodestyle==2.5
```
To check that all have been successfully downloaded, use pip list.

## Tasks

### [Slice Me Up](./0-slice_me_up.py)
- Complete the following source code (found below):
	- arr1 should be the first two numbers of arr
	- arr2 should be the last five numbers of arr
	- arr3 should be the 2nd through 6th numbers of arr
	- You are not allowed to use any loops or conditional statements
	- Your program should be exactly 8 lines

### [Trim Me Down](./1-trim_me_down.py)
- Complete the following source code (found below):
	- the_middle should be a 2D matrix containing the 3rd and 4th columns of matrix
	- You are not allowed to use any conditional statements
	- You are only allowed to use one for loop
	- Your program should be exactly 6 lines

### [Size Me Please](./2-size_me_please.py)
- Write a function def matrix_shape(matrix): that calculates the shape of a matrix:
	- You can assume all elements in the same dimension are of the same type/shape
	- The shape should be returned as a list of integers

### [Flip Me Over](./3-flip_me_over.py)
- Write a function def matrix_transpose(matrix): that returns the transpose of a 2D matrix, matrix:
	- You must return a new matrix
	- You can assume that matrix is never empty
	- You can assume all elements in the same dimension are of the same type/shape

### [Line Up](./4-line_up.py)
- Write a function def add_arrays(arr1, arr2): that adds two arrays element-wise:
	- You can assume that arr1 and arr2 are lists of ints/floats
	- You must return a new list
	- If arr1 and arr2 are not the same shape, return None

### [Across The Planes](./5-across_the_planes.py)
- Write a function def add_matrices2D(mat1, mat2): that adds two matrices element-wise:
	- You can assume that mat1 and mat2 are 2D matrices containing ints/floats
	- You can assume all elements in the same dimension are of the same type/shape
	- You must return a new matrix
	- If mat1 and mat2 are not the same shape, return None

### [Howdy Partner](./6-howdy_partner.py)
- Write a function def cat_arrays(arr1, arr2): that concatenates two arrays:
	- You can assume that arr1 and arr2 are lists of ints/floats
	- You must return a new list

### [Gettin’ Cozy](./7-gettin_cozy.py)
- Write a function def cat_matrices2D(mat1, mat2, axis=0): that concatenates two matrices along a specific axis:
	- You can assume that mat1 and mat2 are 2D matrices containing ints/floats
	- You can assume all elements in the same dimension are of the same type/shape
	- You must return a new matrix
	- If the two matrices cannot be concatenated, return None

### [Ridin’ Bareback](./8-ridin_bareback.py)
- Write a function def mat_mul(mat1, mat2): that performs matrix multiplication:
	- You can assume that mat1 and mat2 are 2D matrices containing ints/floats
	- You can assume all elements in the same dimension are of the same type/shape
	- You must return a new matrix
	- If the two matrices cannot be multiplied, return None

### [Let The Butcher Slice It](9-let_the_butcher_slice_it.py)
- Complete the following source code (found below)
	- mat1 should be the middle two rows of matrix
	- mat2 should be the middle two columns of matrix
	- mat3 should be the bottom-right, square, 3x3 matrix of matrix
	- You are not allowed to use any loops or conditional statements
	- Your program should be exactly 10 lines

#### [I’ll Use My Scale](10-ill_use_my_scale.py)
- Function def np_shape(matrix): that calculates the shape of a numpy.ndarray
	- You are not allowed to use any loops or conditional statements
	- You are not allowed to use try/except statements
	- The shape should be returned as a tuple of integers

### [The Western Exchange](11-the_western_exchange.py)
- Function def np_transpose(matrix): that transposes matrix
	- You can assume that matrix can be interpreted as a numpy.ndarray
	- You are not allowed to use any loops or conditional statements
	- You must return a new numpy.ndarray

### [Bracing The Elements](./12-bracin_the_elements.py)
- Function def np_elementwise(mat1, mat2): that performs element-wise addition, subtraction, multiplication, and division
	- You can assume that mat1 and mat2 can be interpreted as numpy.ndarrays
	- You should return a tuple containing the element-wise sum, difference, product, and quotient, respectively
	- You are not allowed to use any loops or conditional statements
	- You can assume that mat1 and mat2 are never empty

### [Cat's Got Your Tongue](./13-cats_got_your_tongue.py)
- Function def np_cat(mat1, mat2, axis=0) that concatenates two matrices along a specific axis
	- You can assume that mat1 and mat2 can be interpreted as numpy.ndarrays
	- You must return a new numpy.ndarray
	- You are not allowed to use any loops or conditional statements
	- You may use: import numpy as np
	- You can assume that mat1 and mat2 are never empty

### [Saddle Up](./14-saddle_up.py)
- Function def np_matmul(mat1, mat2): that performs matrix multiplication:
	- You can assume that mat1 and mat2 are numpy.ndarrays
	- You are not allowed to use any loops or conditional statements
	- You may use: import numpy as np
	- You can assume that mat1 and mat2 are never empty
