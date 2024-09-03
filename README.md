*Created for Large Scale Computing labs*
### Problem definition:
A two-dimensional array TAB[N][N] with the elements of a row (of length N) placed at consecutive addresses (the element of the array TAB[i][j] is accessed as TAB[iN+j]). For the input array TAB, you need to calculate the output array OUT[N-2R][N-2R] (where N>2R) containing the sums of elements within a radius R. Each element of the output array is the sum of (2R+1)(2*R+1) values.

- Input: an array of size (n+2r) x (n+2r)

- Output: an array of size n x n; we have n tasks, each element is the sum of neighboring elements.

- Neighborhood: the parameter R is the number of components to be included in this sum; for example, r=1 => 9 elements; with r=2 => 25 elements, and so on.

The goal of the project is to compare the performance of computations performed on a GPU with those carried out on a CPU.
