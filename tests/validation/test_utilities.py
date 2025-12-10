
from ...list_utils.validation.type_checks import is_strict_container
from .mock_depth import _iter_test_is_depth_at_least, get_max_depth, is_depth_at_least

def test_is_strict_container():
    assert is_strict_container([1, 2, 3]) is True
    assert is_strict_container((1, 2)) is True
    assert is_strict_container({1, 2, 3}) is True
    assert is_strict_container({'a': 1, 'b': 2}) is True

    assert is_strict_container("hello") is False
    assert is_strict_container(b"bytes") is False
    assert is_strict_container(42) is False
    assert is_strict_container(3.14) is False
    assert is_strict_container(None) is False

def test_get_max_depth():
    assert get_max_depth(42) == 0
    assert get_max_depth("hello") == 0
    assert get_max_depth("hello", str_depth=1) == 1
    assert get_max_depth([1, 2, 3]) == 1
    assert get_max_depth([[1, 2], [3, 4]]) == 2
    assert get_max_depth([[[1]], [[2]]]) == 3
    assert get_max_depth((1, (2, (3,)))) == 3
    assert get_max_depth({1, 2, 3}) == 1
    assert get_max_depth({1: 'a', 2: 'b'}) == 1
    assert get_max_depth({'a': [1, 2], 'b': [3, 4]}) == 1  # Iterating over keys by default
    assert get_max_depth({'a': [1, 2], 'b': [3, 4]}, depth_of_dict_values=True) == 2  # Iterating over values
    assert get_max_depth([{'a': [1, 2]}, {'b': [3, 4]}]) == 2  # Dict keys are strings
    assert get_max_depth([{'a': [1, 2]}, {'b': [3, 4]}], depth_of_dict_values=True) == 3  # Dict values are lists
    assert get_max_depth([["hello", 1]], str_depth=1) == 3
    assert get_max_depth([{"a": [["b"]]}], depth_of_dict_values=True) == 4

def test_is_depth_at_least():
    assert is_depth_at_least(42, 0) is True
    assert is_depth_at_least(42, 1) is False
    assert is_depth_at_least([1, 2, 3], 1) is True
    assert is_depth_at_least([[1, 2], [3, 4]], 2) is True
    assert is_depth_at_least([[1, 2], [3, 4]], 3) is False
    assert is_depth_at_least({'a': [1, 2], 'b': [3, 4]}, 2, depth_of_dict_values=True) is True
    assert is_depth_at_least({'a': [1, 2], 'b': [3, 4]}, 2, depth_of_dict_values=False) is False

def test_iter_test_stops_early():
    #_iter_test_is_depth_at_least returns value, iterations
    result, iterations = _iter_test_is_depth_at_least([[1, 2], [3, 4]], 2)
    assert result is True
    assert iterations < 10  # Should stop early
    result, iterations = _iter_test_is_depth_at_least([[1], [[[[[1]]]]]], 2)
    assert result is True
    assert iterations == 3  # Should stop after checking first item of outer list
    result, iterations = _iter_test_is_depth_at_least(1, 0)
    assert result is True
    assert iterations == 1  # Should stop immediately
    result, iterations = _iter_test_is_depth_at_least(range(4), 1)
    assert result is False
    assert iterations == 1  # Should stop after first iteration