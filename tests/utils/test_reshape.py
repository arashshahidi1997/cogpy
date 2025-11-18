from cogpy.utils.reshape import *


# %% test
def test_unflush_axes():
    # Test case 1
    X1 = np.random.rand(10, 20, 30, 40, 50)
    num_axes1 = 3
    dst_axes1 = 1
    expected_output1 = (10, 30, 40, 50, 20)

    moved_array1 = unflush_axes(X1, num_axes1, dst_axes1)
    assert moved_array1.shape == expected_output1

    # Test case 2
    X2 = np.random.rand(5, 6, 7, 8, 9, 10)
    num_axes2 = 2
    dst_axes2 = 4
    expected_output2 = (5, 6, 7, 8, 9, 10)

    moved_array2 = unflush_axes(X2, num_axes2, dst_axes2)
    assert moved_array2.shape == expected_output2

    # Test case 3
    X3 = np.random.rand(3, 4, 5)
    num_axes3 = 1
    dst_axes3 = 0
    expected_output3 = (5, 3, 4)

    moved_array3 = unflush_axes(X3, num_axes3, dst_axes3)
    assert moved_array3.shape == expected_output3

    print("All test cases pass")


def test_reshape_axis():
    original_shape = (30, 64, 20, 10)
    arr = np.random.rand(*original_shape)
    reshaped_array = reshape_axes(arr, axes=1, shape=(8, 8))
    assert (reshaped_array.reshape(-1) == arr.reshape(-1)).all()
    assert reshaped_array.shape == (30, 8, 8, 20, 10)


def test_reshape_axes():
    # Test case 1
    original_shape1 = (30, 64, 20, 10)
    arr1 = np.random.rand(*original_shape1)
    reshaped_shape1 = (8, 8)
    axes1 = 1
    expected_output1 = (30, 8, 8, 20, 10)

    reshaped_array1 = reshape_axes(arr1, axes1, reshaped_shape1)
    assert np.array_equal(reshaped_array1.reshape(-1), arr1.reshape(-1))
    assert reshaped_array1.shape == expected_output1

    # Test case 2
    original_shape2 = (100, 50, 30, 20)
    arr2 = np.random.rand(*original_shape2)
    reshaped_shape2 = (5, 3, 2)
    axes2 = 2
    expected_output2 = (100, 50, 5, 3, 2, 20)

    reshaped_array2 = reshape_axes(arr2, axes2, reshaped_shape2)
    assert np.array_equal(reshaped_array2.reshape(-1), arr2.reshape(-1))
    assert reshaped_array2.shape == expected_output2

    # Test case 3
    original_shape3 = (50, 40, 30, 20, 10)
    arr3 = np.random.rand(*original_shape3)
    reshaped_shape3 = (5, 4)
    axes3 = 3
    expected_output3 = (50, 40, 30, 5, 4, 10)

    reshaped_array3 = reshape_axes(arr3, axes3, reshaped_shape3)
    assert np.array_equal(reshaped_array3.reshape(-1), arr3.reshape(-1))
    assert reshaped_array3.shape == expected_output3

    print("All test cases pass")
