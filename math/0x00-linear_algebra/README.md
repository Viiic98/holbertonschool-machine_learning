# LINEAR ALGEBRA

### INSTALLING GUIDES

#### Installing pip 19.1
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

#### Installing numpy 1.15, scipy 1.3, and pycodestyle 2.5
```
pip install --user numpy==1.15
pip install --user scipy==1.3
pip install --user pycodestyle==2.5
```
To check that all have been successfully downloaded, use pip list.

#### Slice Me Up [0-slice_me_up.py](./0-slice_me_up.py)
- Complete the following source code (found below):
	- arr1 should be the first two numbers of arr
	- arr2 should be the last five numbers of arr
	- arr3 should be the 2nd through 6th numbers of arr
	- You are not allowed to use any loops or conditional statements
	- Your program should be exactly 8 lines

#### Trim Me Down [1-trim_me_down.py](./1-trim_me_down.py)
- Complete the following source code (found below):
	- the_middle should be a 2D matrix containing the 3rd and 4th columns of matrix
	- You are not allowed to use any conditional statements
	- You are only allowed to use one for loop
	- Your program should be exactly 6 lines

#### Size Me Please [2-size_me_please.py](./2-size_me_please.py)
- Write a function def matrix_shape(matrix): that calculates the shape of a matrix:
	- You can assume all elements in the same dimension are of the same type/shape
	- The shape should be returned as a list of integers

#### Flip Me Over [3-flip_me_over.py](./3-flip_me_over.py)
- Write a function def matrix_transpose(matrix): that returns the transpose of a 2D matrix, matrix:
	- You must return a new matrix
	- You can assume that matrix is never empty
	- You can assume all elements in the same dimension are of the same type/shape

#### Line Up [4-line_up.py](./4-line_up.py)
- Write a function def add_arrays(arr1, arr2): that adds two arrays element-wise:
	- You can assume that arr1 and arr2 are lists of ints/floats
	- You must return a new list
	- If arr1 and arr2 are not the same shape, return None

#### Across The Planes [5-across_the_planes.py](./5-across_the_planes.py)
- Write a function def add_matrices2D(mat1, mat2): that adds two matrices element-wise:
	- You can assume that mat1 and mat2 are 2D matrices containing ints/floats
	- You can assume all elements in the same dimension are of the same type/shape
	- You must return a new matrix
	- If mat1 and mat2 are not the same shape, return None

#### Howdy Partner [6-howdy_partner.py](./6-howdy_partner.py)
- Write a function def cat_arrays(arr1, arr2): that concatenates two arrays:
	- You can assume that arr1 and arr2 are lists of ints/floats
	- You must return a new list

#### Gettin’ Cozy [7-gettin_cozy.py](./7-gettin_cozy.py)
- Write a function def cat_matrices2D(mat1, mat2, axis=0): that concatenates two matrices along a specific axis:
	- You can assume that mat1 and mat2 are 2D matrices containing ints/floats
	- You can assume all elements in the same dimension are of the same type/shape
	- You must return a new matrix
	- If the two matrices cannot be concatenated, return None
