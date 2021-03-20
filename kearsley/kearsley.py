import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation


class Kearsley():
    '''
    A Kearsley transformation.

    Algorithm to minimize the sum of the squared distances between two sets of points.

    Author: Simon K. Kearsley

    Paper: "On the orthogonal transformation used for structural comparisons"
    https://doi.org/10.1107/S0108767388010128

    Attributes
    ----------
    rot: Rotation
        A scipy.spatial.transform Rotation in 3 dimensions.
    trans: ndarray, shape (3,)
        The 3D translation.

    Examples
    ----------
    >>> import numpyas np
    >>> from scipy.spatial.transform import Rotation
    >>> from kearsley import Kearsley
    
    Create a set of points
    >>> u = np.mgrid[0:3, 0:3, 0:3].reshape(3, -1).T
    >>> u
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 0, 2],
           [0, 1, 0],
           [0, 1, 1],
           [0, 1, 2],
           ...,
           [2, 1, 0],
           [2, 1, 1],
           [2, 1, 2],
           [2, 2, 0],
           [2, 2, 1],
           [2, 2, 2]])

    Create a rotation
    >>> r = Rotation.from_quat([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)])

    Rotate the points
    >>> v = r.apply(u)

    Fit both sets of points
    >>> k = Kearsley()
    >>> rmsd = k.fit()
    >>> rmsd
    0.0

    >>> k.rot.as_matrix()
    array([[ 0.00000000e+00,  1.00000000e+00, -1.23358114e-17],
           [-1.00000000e+00,  0.00000000e+00,  1.23358114e-17],
           [ 1.23358114e-17,  1.23358114e-17,  1.00000000e+00]])

    >>> k.trans
    array([0., 0., 0.])

    >>> np.allclose(u, k.transform(v))
    True

    Rotate the points with some noise (values may change)
    >>> v = r.apply(u) + np.random.random((27, 3)) - 0.5
    >>> rmsd = k.fit(u, v)
    >>> rmsd
    0.46749750347988245

    >>> k.rot.as_matrix()
    array([[-0.04192624,  0.99812318,  0.04463537],
           [-0.99649536, -0.03853783, -0.07424172],
           [-0.07238224, -0.04759161,  0.99624086]])

    >>> k.trans
    array([ 0.04735894,  0.14588084, -0.01272672])
    '''

    def __init__(self):

        self.rot = Rotation.from_quat([0, 0, 0, 1])
        self.trans = np.zeros(3)

    def _kearsley_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Calculates the Kearsley matrix.

        Parameters
        ----------
        x: ndarray, shape (N, 3)
            Input array with 3D points, centered at zero.
        y: ndarray, shape (N, 3)
            Input array with 3D points, centered at zero.
        
        Returns
        ----------
        K: ndarray, shape (4, 4)
            The Kearsley matrix.
        '''
        # diff and sum quantities
        d, s = x - y, x + y

        # extract columns to simplify notation
        d0, d1, d2 = d[:,0], d[:,1], d[:,2]
        s0, s1, s2 = s[:,0], s[:,1], s[:,2]

        # fill kearsley matrix
        K = np.zeros((4, 4))
        K[0,0] = np.dot(d0, d0) + np.dot(d1, d1) + np.dot(d2, d2)
        K[1,0] = np.dot(s1, d2) - np.dot(d1, s2)
        K[2,0] = np.dot(d0, s2) - np.dot(s0, d2)
        K[3,0] = np.dot(s0, d1) - np.dot(d0, s1)
        K[1,1] = np.dot(s1, s1) + np.dot(s2, s2) + np.dot(d0, d0)
        K[2,1] = np.dot(d0, d1) - np.dot(s0, s1)
        K[3,1] = np.dot(d0, d2) - np.dot(s0, s2)
        K[2,2] = np.dot(s0, s0) + np.dot(s2, s2) + np.dot(d1, d1)
        K[3,2] = np.dot(d1, d2) - np.dot(s1, s2)
        K[3,3] = np.dot(s0, s0) + np.dot(s1, s1) + np.dot(d2, d2)

        return K

    def transform(self, u: np.ndarray) -> np.ndarray:
        '''
        Transforms a list of 3D points with a rotation and a translation.

        Parameters
        ----------
        u: ndarray, shape (N, 3)
            Input array with 3D points.
        
        Returns
        ----------
        array: ndarray, shape (N, 3)
            Input points transformed.

        Raises
        ----------
        ValueError
            If the input points have not the correct shape
        '''
        if len(u.shape) != 2:
            raise ValueError('Input array must have 2 dimensions')

        if u.shape[1] != 3:
            raise ValueError('Input array must have 3 columns')

        return self.rot.apply(u - self.trans)

    def fit(self, u: np.ndarray, v: np.ndarray) -> np.float:
        '''
        Calculates the rotation and translation that best fits both sets of points.

        Parameters
        ----------
        u: ndarray, shape (N, 3)
            Input array with 3D points.
        v: ndarray, shape (N, 3)
            Input array with 3D points.
        
        Returns
        ----------
        rmsd: float
            The root mean squared deviation.

        Raises
        ----------
        ValueError
            If the input points have not the correct shape, or don't have the same number of points.
        '''
        if len(u.shape) != 2 or len(v.shape) != 2:
            raise ValueError('Input arrays must have 2 dimensions')

        if u.shape[1] != 3 or v.shape[1] != 3:
            raise ValueError('Input arrays must have 3 columns')

        if u.shape[0] != v.shape[0]:
            raise ValueError('Both sets of points must have the same number of points')
        
        # centroids
        centroid_u = u.mean(axis=0)
        centroid_v = v.mean(axis=0)

        # center both sets of points
        x, y = u - centroid_u, v - centroid_v

        # calculate Kearsley matrix
        K = self._kearsley_matrix(x, y)

        # diagonalize K
        eig_vals, eig_vecs = la.eigh(K)

        # first eig_vec minimizes the rmsd
        q = eig_vecs[:,0]
        q = np.roll(q, shift=3)

        # calculate rotation and translation
        self.rot = Rotation.from_quat(q).inv()
        self.trans = centroid_v - self.rot.inv().apply(centroid_u)

        # calculate rmsd
        eig_val = np.abs(eig_vals[0])
        rmsd = np.sqrt(eig_val/u.shape[0])

        return rmsd

    def fit_transform(self, u: np.ndarray, v: np.ndarray) -> (np.ndarray, np.float):
        '''
        Calculates the rotation and translation that best fits both sets of points and
        applies the transformation to the second set.

        Parameters
        ----------
        u: ndarray, shape (N, 3)
            Input array with 3D points.
        v: ndarray, shape (N, 3)
            Input array with 3D points.
        
        Returns
        ----------
        array: ndarray, shape (N, 3)
            Input points transformed.
        rmsd: float
            The root mean squared deviation.

        Raises
        ----------
        ValueError
            If the input points have not the correct shape, or don't have the same number of points.
        '''
        rmsd = self.fit(u, v)

        return self.transform(v), rmsd


