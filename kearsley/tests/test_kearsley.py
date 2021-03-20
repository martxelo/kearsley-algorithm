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
    'rnd_trans, rnd_angles',
    [
        (np.random.random(3)*10, np.random.random(3)),
    ],
)
def test_kearsley_fit(rnd_trans, rnd_angles):

    rnd_rot = Rotation.from_euler('xyz', rnd_angles)

    u = np.mgrid[0:10, 0:10, 0:10].reshape(3, -1).T
    v = rnd_rot.apply(u) + rnd_trans

    k = Kearsley()
    v_trans, rmsd = k.fit_transform(u, v, return_rmsd=True)

    assert np.isclose(rmsd, 0.0, atol=1e-5)
    assert np.allclose(u, v_trans)