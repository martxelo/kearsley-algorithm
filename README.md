# Kearsley algorithm

This module provides a class to perform the Kearsley algorithm for structural comparisons. The class calculates the rotation and translation that minimizes the root mean squared deviations for two sets of 3D points.

Original paper by Simon K. Kearsley: _On the orthogonal transformation used for structural comparisons_ https://doi.org/10.1107/S0108767388010128.

## Usage

Given two sets of 3D points ```u``` and ```v```:
```
>>> u, v = read_data()
>>> u
array([[0, 0, 0],
       [0, 0, 1],
       [0, 0, 2],
       ...,
       [9, 9, 7],
       [9, 9, 8],
       [9, 9, 9]])
>>> v
array([[ 30.50347534, -20.16089091,  -7.42752623],
       [ 30.77704903, -21.02339348,  -7.27823201],
       [ 31.3215374 , -21.99452332,  -7.15703548],
       ...,
       [ 42.05988643, -23.50924264, -15.59516355],
       [ 42.27217891, -24.36478643, -15.59064995],
       [ 42.66080502, -25.27318759, -15.386241  ]])
```
It is possible to calculate the rotation and translation that minimize the root mean squared deviation:
```
>>> from kearsley import Kearsley
>>> k = Kearsley()
>>> rmsd = k.fit(u, v)
>>> rmsd
0.10003430497284149
```
The rotation and translation are the attributes of the class:
```
>>> k.rot.as_matrix()
array([[ 0.05552838, -0.04405506, -0.99748471],
       [ 0.91956342,  0.39147652,  0.03390061],
       [ 0.38899835, -0.9191329 ,  0.06224948]])
>>> k.trans
array([ 30.46560753, -20.15086287,  -7.34422276])
```
Once fitted you can apply the transformation to ```v``` or to other set of points:
```
>>> v_transform = k.transform(v)
>>> v_transform
array([[ 0.08563846,  0.02807207,  0.01876202],
       [-0.01009153, -0.0529479 ,  0.92722971],
       [-0.05796549,  0.07167779,  2.03917659],
       ...,
       [ 9.0219524 ,  9.067236  ,  7.08333594],
       [ 9.06692944,  8.9276801 ,  7.95255679],
       [ 8.92463409,  8.93635832,  8.95139744]])
```
It is also possible to fit and transform with one command, in this case the transformation is applied to the second set of points:
```
>>> v_transform, rmsd = k.fit_transform(u, v)
>>> rmsd
0.10003430497284149
>>> v_transform
array([[ 0.08563846,  0.02807207,  0.01876202],
       [-0.01009153, -0.0529479 ,  0.92722971],
       [-0.05796549,  0.07167779,  2.03917659],
       ...,
       [ 9.0219524 ,  9.067236  ,  7.08333594],
       [ 9.06692944,  8.9276801 ,  7.95255679],
       [ 8.92463409,  8.93635832,  8.95139744]])
```
The rmsd is the expected:
```
>>> np.sqrt(np.sum((u - v_transform)**2)/len(u))
0.10003430497298871
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