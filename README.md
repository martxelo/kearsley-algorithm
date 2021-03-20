# Kearsley algorithm

This module provides a class to perform the Kearsley algorithm for structural comparisons. The class calculates the rotation and translation that minimizes the root mean squared deviations for two sets of 3D points.

Original paper by Simon K. Kearsley: _On the orthogonal transformation used for structural comparisons_ https://doi.org/10.1107/S0108767388010128.

## Usage

Given a set of 3D points ```u```:
```
>>> import numpy as np
>>> u = np.mgrid[0:10, 0:10, 0:10].reshape(3, -1).T
>>> u
array([[0, 0, 0],
       [0, 0, 1],
       [0, 0, 2],
       ...,
       [9, 9, 7],
       [9, 9, 8],
       [9, 9, 9]])
```
Let's create another set ```v``` rotating and translating the first set:
```
>>> r = Rotation.from_euler('xyz', [90, 45, 120])
>>> v = r.apply(u) + np.array([[10, 15, -5]])
>>> v
array([[ 10.        ,  15.        ,  -5.        ],
       [ 10.20864378,  14.0507568 ,  -5.23538292],
       [ 10.41728757,  13.1015136 ,  -5.47076585],
       ...,
       [ 23.22544314,  11.79211492, -10.07908724],
       [ 23.43408692,  10.84287172, -10.31447016],
       [ 23.6427307 ,   9.89362852, -10.54985308]])
```
Both sets of points have the same structure, so the RMSD is close to zero.
```
>>> from kearsley import Kearsley
>>> k = Kearsley()
>>> rmsd = k.fit(u, v)
>>> rmsd
8.529922399520072e-08
```
It is possible to apply the transformation to the second set of points.
```
>>> v_transform = k.transform(v)
>>> v_transform
array([[-2.27527525e-15, -2.70751951e-15, -9.50313659e-16],
       [-2.08166817e-15, -3.06699111e-15,  1.00000000e+00],
       [-1.88737914e-15, -3.38618023e-15,  2.00000000e+00],
       ...,
       [ 9.00000000e+00,  9.00000000e+00,  7.00000000e+00],
       [ 9.00000000e+00,  9.00000000e+00,  8.00000000e+00],
       [ 9.00000000e+00,  9.00000000e+00,  9.00000000e+00]])
```
It is also possible to fit and transform with one command.
```
>>> v_transform, rmsd = k.fit_transform(u, v)
>>> rmsd
8.529922399520072e-08
>>> v_transform
array([[-2.27527525e-15, -2.70751951e-15, -9.50313659e-16],
       [-2.08166817e-15, -3.06699111e-15,  1.00000000e+00],
       [-1.88737914e-15, -3.38618023e-15,  2.00000000e+00],
       ...,
       [ 9.00000000e+00,  9.00000000e+00,  7.00000000e+00],
       [ 9.00000000e+00,  9.00000000e+00,  8.00000000e+00],
       [ 9.00000000e+00,  9.00000000e+00,  9.00000000e+00]])
```
The values of ```v_transform``` are close to the values of ```u```.
```
>>> np.allclose(u, v_transform)
True
```
There are two attributes:

- Kearsley.rot: a scipy Rotation instance.
- Kearsley.trans: a ndarray with shape (3,) with the translation.


## Applications

- Compare a set of measured points with their theoretical positions.
- In robotics compare two sets of points measured in different coordinate systems and get the transformation between both coordinate systems. 
- It is possible to use it in a 2D space fixing the third coordinate to zero.

## Notes

Check [Scipy Rotation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) to have all the info about Rotation instance.