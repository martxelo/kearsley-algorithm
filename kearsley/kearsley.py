

import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation


class Kearsley():

    def __init__(self):

        self.rot = Rotation.from_quat([0, 0, 0, 1])
        self.trans = np.zeros(3)

    def _centroid(self, u):

        return u.mean(axis=0)

    def _kearsley_matrix(self, x, y):

        d, s = x - y, x + y

        d0, d1, d2 = d[:,0], d[:,1], d[:,2]
        s0, s1, s2 = s[:,0], s[:,1], s[:,2]

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

    def transform(self, u):

        return self.rot.apply(u - self.trans)

    def fit(self, u, v):

        centroid_u = self._centroid(u)
        centroid_v = self._centroid(v)

        x, y = u - centroid_u, v - centroid_v

        K = self._kearsley_matrix(x, y)

        eig_vals, eig_vecs = la.eigh(K)

        q = eig_vecs[:,0]
        q = np.roll(q, shift=3)

        self.rot = Rotation.from_quat(q).inv()
        self.trans = centroid_v - self.rot.inv().apply(centroid_u)

        print('RMSD =', np.sqrt(eig_vals[0]/u.shape[0]))




