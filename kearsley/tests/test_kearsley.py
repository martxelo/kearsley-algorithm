import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import pytest

from kearsley.kearsley import Kearsley


@pytest.mark.parametrize(
    'u, message_expected',
    [
        (np.zeros(3), 'Input array must have 2 dimensions'),
        (np.zeros((3,3,3)), 'Input array must have 2 dimensions'),
        (np.zeros((3,2)), 'Input array must have 3 columns'),
        (np.zeros((10,4)), 'Input array must have 3 columns'),
    ],
)
def test_kearsley_transform_failure(u, message_expected):
    
    with pytest.raises(ValueError, match=message_expected):
        k = Kearsley()
        k.transform(u)


@pytest.mark.parametrize(
    'u, v, message_expected',
    [
        (np.zeros(3), np.zeros(5), 'Input arrays must have 2 dimensions'),
        (np.zeros((3,3,3)), np.zeros((3,3,3)), 'Input arrays must have 2 dimensions'),
        (np.zeros((3,3)), np.zeros((3,3,3)), 'Input arrays must have 2 dimensions'),
        (np.zeros((3,2)), np.zeros((3,2)), 'Input arrays must have 3 columns'),
        (np.zeros((3,2)), np.zeros((3,3)), 'Input arrays must have 3 columns'),
        (np.zeros((10,3)), np.zeros((9,3)), 'Both sets of points must have the same number of points'),
    ],
)
def test_kearsley_fit_failure(u, v, message_expected):
    
    with pytest.raises(ValueError, match=message_expected):
        k = Kearsley()
        k.fit(u, v)


@pytest.mark.parametrize(
    'v, rot, trans, expected_u',
    [
        (np.array([[ 0.,  0.,  1.],
                   [ 0.,  0.,  2.],
                   [-1.,  0.,  1.],
                   [-1.,  0.,  2.],
                   [ 0.,  1.,  1.],
                   [ 0.,  1.,  2.],
                   [-1.,  1.,  1.],
                   [-1.,  1.,  2.]]),
        np.array([[ 0.,  1.,  0.],
                  [-1.,  0., -0.],
                  [-0.,  0.,  1.]]),
        np.array([0., 0., 1.]),
        np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]])),
    ],
)
def test_kearsley_transform(v, rot, trans, expected_u):

    k = Kearsley()
    k.rot = Rotation.from_matrix(rot)
    k.trans = trans

    u = k.transform(v)

    assert np.allclose(u, expected_u)


@pytest.mark.parametrize(
    'u, v, expected_rot, expected_trans',
    [
        (np.array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 1],
                   [1, 1, 0],
                   [1, 1, 1]]),
        np.array([[ 0.,  0.,  1.],
                  [ 0.,  0.,  2.],
                  [-1.,  0.,  1.],
                  [-1.,  0.,  2.],
                  [ 0.,  1.,  1.],
                  [ 0.,  1.,  2.],
                  [-1.,  1.,  1.],
                  [-1.,  1.,  2.]]),
        np.array([[ 0.,  1.,  0.],
                  [-1.,  0., -0.],
                  [-0.,  0.,  1.]]),
        np.array([0., 0., 1.])),
    ],
)
def test_kearsley_fit(u, v, expected_rot, expected_trans):

    k = Kearsley()
    k.fit(u, v)

    assert np.allclose(k.rot.as_matrix(), expected_rot)
    assert np.allclose(k.trans, expected_trans)