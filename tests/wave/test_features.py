from cogpy.wave.features import *

def test_positive_boundaries():
    x = np.array([1, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0])
    boundaries = positive_boundaries(x)
    assert np.all(boundaries == np.array([[0, 1], [2, 5], [7, 10]])), print(boundaries)
