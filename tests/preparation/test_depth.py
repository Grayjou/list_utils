import pytest
from ...list_utils.preparation.depth import (
    ensure_uniform_depth,
    UnwrapPolicy,
    shed_layer,
    add_list_layer,
)


# ============================================================================
# Tests for shed_layer
# ============================================================================

def test_shed_layer_single_item():
    assert shed_layer([1]) == 1
    assert shed_layer((1,)) == 1
    assert shed_layer({1}) == 1


def test_shed_layer_multiple_items_ignore_extra():
    assert shed_layer([1, 2, 3], ignore_extra=True) == 1
    assert shed_layer((1, 2), ignore_extra=True) == 1


def test_shed_layer_multiple_items_no_ignore():
    with pytest.raises(ValueError, match="Cannot unpack: container has"):
        shed_layer([1, 2, 3], ignore_extra=False)


def test_shed_layer_empty_container():
    with pytest.raises(ValueError, match="empty container"):
        shed_layer([])


def test_shed_layer_non_container():
    with pytest.raises(ValueError, match="non-container"):
        shed_layer(42)


# ============================================================================
# Tests for add_list_layer
# ============================================================================

def test_add_list_layer():
    assert add_list_layer(1) == [1]
    assert add_list_layer([1, 2]) == [[1, 2]]
    assert add_list_layer("hello") == ["hello"]
    assert add_list_layer([[1]]) == [[[1]]]


# ============================================================================
# Tests for ensure_uniform_depth - Inside-Out Mode
# ============================================================================

class TestInsideOutMode:
    """Tests for inside_out=True depth normalization."""
    
    def test_depth_zero(self):
        """At depth 0, return as-is."""
        assert ensure_uniform_depth(42, 0, inside_out=True) == 42
        assert ensure_uniform_depth([1, 2], 0, inside_out=True) == [1, 2]
        assert ensure_uniform_depth("hello", 0, inside_out=True) == "hello"
    
    def test_wrap_non_container(self):
        """Non-containers get wrapped to target depth."""
        assert ensure_uniform_depth(42, 1, inside_out=True) == [42]
        assert ensure_uniform_depth(42, 2, inside_out=True) == [[42]]
        assert ensure_uniform_depth(42, 3, inside_out=True) == [[[42]]]
        assert ensure_uniform_depth("hello", 2, inside_out=True) == [["hello"]]
    
    def test_shallow_list_gets_wrapped(self):
        """Lists with insufficient depth get wrapped from inside."""
        result = ensure_uniform_depth([1, 2, 3], 2, inside_out=True)
        assert result == [[1], [2], [3]]
    
    def test_nested_wrapping(self):
        """Each element gets wrapped independently."""
        result = ensure_uniform_depth([[1], [2]], 3, inside_out=True)
        assert result == [[[1]], [[2]]]
    
    def test_mixed_depth_elements(self):
        """Elements with different depths all get normalized."""
        result = ensure_uniform_depth([1, [2], [[3]]], 3, inside_out=True)
        # Inside-out: each element individually wrapped to depth 2 (since container adds 1)
        # 1 -> needs 2 wraps -> [[1]]
        # [2] -> is container, recurse with depth 2 -> [2] needs 1 wrap -> [2]
        # [[3]] -> is container, recurse with depth 2 -> [3] is container, recurse with depth 1 -> 3 needs 0 wraps -> 3
        assert result == [[[1]], [[2]], [[3]]]
    
    def test_empty_container(self):
        """Empty containers get wrapped appropriately."""
        result = ensure_uniform_depth([], 2, inside_out=True)
        assert result == []
    
    def test_nested_empty_containers(self):
        """Nested empty containers."""
        result = ensure_uniform_depth([[], []], 3, inside_out=True)
        assert result == [[], []]


# ============================================================================
# Tests for ensure_uniform_depth - Outside-In Mode (STRICT policy)
# ============================================================================

class TestOutsideInModeStrict:
    """Tests for inside_out=False with STRICT unwrap policy."""
    
    def test_depth_zero(self):
        """At depth 0, return as-is for non-containers."""
        assert ensure_uniform_depth(42, 0, inside_out=False) == 42
        assert ensure_uniform_depth("hello", 0, inside_out=False) == "hello"
    
    def test_wrap_shallow_object(self):
        """Objects with insufficient depth get wrapped."""
        result = ensure_uniform_depth("aaa", 2, inside_out=False)
        assert result == [["aaa"]]
        
        result = ensure_uniform_depth(42, 3, inside_out=False)
        assert result == [[[42]]]
    
    def test_wrap_shallow_list(self):
        """Lists with insufficient depth get wrapped."""
        result = ensure_uniform_depth([1, 2, 3], 2, inside_out=False)
        assert result == [[1, 2, 3]]
    
    def test_unwrap_single_item_container(self):
        """Containers with single items can be unwrapped."""
        result = ensure_uniform_depth([[1]], 1, inside_out=False)
        assert result == [1]
        
        result = ensure_uniform_depth([[[1]]], 1, inside_out=False)
        assert result == [1]
    
    def test_unwrap_multiple_layers(self):
        """Multiple layers can be unwrapped if uniform."""
        result = ensure_uniform_depth([[[[1]]]], 2, inside_out=False)
        assert result == [[1]]
    
    def test_unwrap_fails_with_multiple_items(self):
        """STRICT policy fails when trying to unwrap containers with >1 items."""
        with pytest.raises(ValueError, match="single-item container"):
            ensure_uniform_depth([[1], [2]], 1, inside_out=False)
        
        with pytest.raises(ValueError, match="single-item container"):
            ensure_uniform_depth([[1, 2, 3]], 0, inside_out=False)
    
    def test_exact_depth_no_change(self):
        """Objects already at exact depth remain unchanged."""
        result = ensure_uniform_depth([1, 2, 3], 1, inside_out=False)
        assert result == [1, 2, 3]
        
        result = ensure_uniform_depth([[1], [2]], 2, inside_out=False)
        assert result == [[1], [2]]


# ============================================================================
# Tests for ensure_uniform_depth - Outside-In Mode (IGNORE_EXTRA policy)
# ============================================================================

class TestOutsideInModeIgnoreExtra:
    """Tests for inside_out=False with IGNORE_EXTRA unwrap policy."""
    
    def test_unwrap_keeps_first_item(self):
        """IGNORE_EXTRA keeps only the first item when unwrapping."""
        result = ensure_uniform_depth(
            [[1], [2], [3]], 1, 
            inside_out=False, 
            unwrap_policy=UnwrapPolicy.IGNORE_EXTRA
        )
        assert result == [1]
    
    def test_unwrap_multiple_layers_first_item(self):
        """Multiple unwraps take first item at each level."""
        result = ensure_uniform_depth(
            [[[1, 2], [3, 4]], [[5, 6]]], 1,
            inside_out=False,
            unwrap_policy=UnwrapPolicy.IGNORE_EXTRA
        )
        assert result == [1, 2]
    
    def test_ignore_extra_with_nested_structure(self):
        """Complex nested structures use first item at each unwrap."""
        result = ensure_uniform_depth(
            [[[1]], [[2]], [[3]]], 2,
            inside_out=False,
            unwrap_policy=UnwrapPolicy.IGNORE_EXTRA
        )
        assert result == [[1]]
    
    def test_empty_container_fails(self):
        """Empty containers still fail even with IGNORE_EXTRA."""
        with pytest.raises(ValueError, match="empty container"):
            ensure_uniform_depth(
                [[]], 0,
                inside_out=False,
                unwrap_policy=UnwrapPolicy.IGNORE_EXTRA
            )


# ============================================================================
# Tests for ensure_uniform_depth - Outside-In Mode (MERGE policy)
# ============================================================================

class TestOutsideInModeMerge:
    """Tests for inside_out=False with MERGE unwrap policy."""
    
    def test_merge_flattens_containers(self):
        """MERGE policy concatenates inner containers."""
        result = ensure_uniform_depth(
            [[1], [2, 3]], 1,
            inside_out=False,
            unwrap_policy=UnwrapPolicy.MERGE
        )
        assert result == [1, 2, 3]
    
    def test_merge_with_non_containers(self):
        """MERGE appends non-containers as-is."""
        result = ensure_uniform_depth(
            [[1, 2], 3, [4]], 0,
            inside_out=False,
            unwrap_policy=UnwrapPolicy.MERGE
        )
        assert result == [1, 2, 3, 4]
    
    def test_merge_multiple_layers(self):
        """Multiple unwraps with MERGE flatten progressively."""
        result = ensure_uniform_depth(
            [[[1], [2]], [[3], [4]]], 1,
            inside_out=False,
            unwrap_policy=UnwrapPolicy.MERGE
        )
        assert result == [1, 2, 3, 4]
    
    def test_merge_mixed_types(self):
        """MERGE handles mixed container types."""
        result = ensure_uniform_depth(
            [[[1, 2]], [[3]], [4]], 1,
            inside_out=False,
            unwrap_policy=UnwrapPolicy.MERGE
        )
        assert result == [1, 2, 3, 4]


# ============================================================================
# Tests for depth_of_dict_values parameter
# ============================================================================

class TestDictValuesDepth:
    """Tests for depth_of_dict_values parameter."""
    
    def test_dict_keys_default(self):
        """By default, dicts iterate over keys."""
        # Dict with string keys has depth 1
        result = ensure_uniform_depth(
            {'a': [1, 2], 'b': [3, 4]}, 2,
            inside_out=False
        )
        assert result == [{'a': [1, 2], 'b': [3, 4]}]
    
    def test_dict_values_explicit(self):
        """With depth_of_dict_values=True, iterate over values."""
        # Dict values are lists, so depth is 2
        result = ensure_uniform_depth(
            {'a': [1, 2], 'b': [3, 4]}, 3,
            inside_out=False,
            depth_of_dict_values=True
        )
        assert result == [{'a': [1, 2], 'b': [3, 4]}]
    
    def test_nested_dict_with_dict_values(self):
        """Nested dicts with depth_of_dict_values."""
        # With depth_of_dict_values=True: List(1) -> Dict(2) -> List(3)
        # Actual max depth is 3, so it matches exactly
        result = ensure_uniform_depth(
            [{'a': [1, 2]}, {'b': [3, 4]}], 4,
            inside_out=False,
            depth_of_dict_values=True
        )
        # Should wrap the whole thing since depth 3 < 4
        assert result == [[{'a': [1, 2]}, {'b': [3, 4]}]]


# ============================================================================
# Tests for edge cases and error handling
# ============================================================================

class TestEdgeCases:
    """Edge cases and error conditions."""
    
    def test_negative_depth_raises_error(self):
        """Negative depth values should raise ValueError."""
        with pytest.raises(ValueError, match="depth must be >= 0"):
            ensure_uniform_depth([1, 2], -1)
    
    def test_depth_zero_with_container(self):
        """Depth 0 with container in outside-in mode."""
        # Should try to unwrap, but will fail with STRICT policy
        with pytest.raises(ValueError, match="single-item container"):
            ensure_uniform_depth([1, 2, 3], 0, inside_out=False)
    
    def test_very_deep_nesting(self):
        """Handle very deep nesting levels."""
        deep = 1
        for _ in range(10):
            deep = [deep]
        
        result = ensure_uniform_depth(deep, 5, inside_out=False)
        # Should unwrap from 10 to 5
        expected = 1
        for _ in range(5):
            expected = [expected]
        assert result == expected
    
    def test_tuple_as_container(self):
        """Tuples are treated as containers."""
        result = ensure_uniform_depth((1, 2, 3), 2, inside_out=False)
        assert result == [(1, 2, 3)]
    
    def test_set_as_container(self):
        """Sets are treated as containers."""
        result = ensure_uniform_depth({1, 2}, 2, inside_out=False)
        assert result == [{1, 2}]
    
    def test_bytes_not_container(self):
        """Bytes are not treated as containers."""
        result = ensure_uniform_depth(b"hello", 2, inside_out=False)
        assert result == [[b"hello"]]
    
    def test_string_not_container(self):
        """Strings are not treated as containers."""
        result = ensure_uniform_depth("hello", 2, inside_out=False)
        assert result == [["hello"]]


# ============================================================================
# Tests for complex scenarios
# ============================================================================

class TestComplexScenarios:
    """Complex real-world scenarios."""
    
    def test_normalize_args_to_uniform_depth(self):
        """Simulate normalizing function arguments to uniform depth."""
        # Different arg formats should normalize to same depth
        args1 = "a"
        args2 = ["a"]
        
        result1 = ensure_uniform_depth(args1, 1, inside_out=False)
        result2 = ensure_uniform_depth(args2, 1, inside_out=False)
        
        assert result1 == ["a"]
        assert result2 == ["a"]
    
    def test_batch_processing_depth_normalization(self):
        """Normalize batch data to consistent depth."""
        single_item = 42
        batch_items = [1, 2, 3]
        nested_batch = [[1, 2], [3, 4]]
        
        # All should become depth 2
        assert ensure_uniform_depth(single_item, 2, inside_out=False) == [[42]]
        assert ensure_uniform_depth(batch_items, 2, inside_out=False) == [[1, 2, 3]]
        assert ensure_uniform_depth(nested_batch, 2, inside_out=False) == [[1, 2], [3, 4]]
    
    def test_inside_out_preserves_structure(self):
        """Inside-out mode preserves heterogeneous structures."""
        mixed = [1, [2, 3], [[4]]]
        result = ensure_uniform_depth(mixed, 3, inside_out=True)
        
        # Each element independently wrapped to depth 2 (depth-1 for children)
        # 1 -> [[1]]
        # [2, 3] -> recurse on 2 and 3 each with depth 2 -> [2] and [3] -> [[2], [3]]
        # [[4]] -> recurse on [4] with depth 2 -> [4] recurse on 4 with depth 1 -> [4] -> [[4]]
        assert result == [[[1]], [[2], [3]], [[4]]]
    
    def test_outside_in_with_merge_flattens(self):
        """Outside-in with MERGE can flatten irregular structures."""
        irregular = [[[1]], [[2, 3]], [[4]]]
        result = ensure_uniform_depth(
            irregular, 1,
            inside_out=False,
            unwrap_policy=UnwrapPolicy.MERGE
        )
        assert result == [1, 2, 3, 4]
    
    def test_recursive_structure_normalization(self):
        """Deeply nested recursive structures."""
        data = [[[[[1]]]]]
        
        # Unwrap to depth 2
        result = ensure_uniform_depth(data, 2, inside_out=False)
        assert result == [[1]]
        
        # Wrap to depth 7
        result = ensure_uniform_depth(data, 7, inside_out=False)
        assert result == [[[[[[[1]]]]]]]


# ============================================================================
# Tests for consistency between modes
# ============================================================================

class TestModeConsistency:
    """Test consistency between inside-out and outside-in modes."""
    
    def test_same_result_for_wrapping(self):
        """Both modes should produce same result when only wrapping."""
        data = 42
        
        result_inside = ensure_uniform_depth(data, 3, inside_out=True)
        result_outside = ensure_uniform_depth(data, 3, inside_out=False)
        
        assert result_inside == result_outside == [[[42]]]
    
    def test_different_behavior_for_containers(self):
        """Modes differ for containers needing depth increase."""
        data = [1, 2, 3]
        
        # Inside-out wraps each element
        result_inside = ensure_uniform_depth(data, 2, inside_out=True)
        assert result_inside == [[1], [2], [3]]
        
        # Outside-in wraps whole container
        result_outside = ensure_uniform_depth(data, 2, inside_out=False)
        assert result_outside == [[1, 2, 3]]


# ============================================================================
# Tests for str_depth and bytes_depth parameters
# ============================================================================

class TestStringAndBytesDepth:
    """Test handling of strings and bytes with configurable depth."""
    
    def test_string_depth_zero_atomic(self):
        """With str_depth=0, strings are atomic (depth 0)."""
        result = ensure_uniform_depth("hello", 2, inside_out=False, str_depth=0)
        assert result == [["hello"]]
    
    def test_string_depth_one_sequence(self):
        """With str_depth=1, strings are sequences (depth 1)."""
        result = ensure_uniform_depth("hello", 1, inside_out=False, str_depth=1)
        assert result == "hello"
        
        result = ensure_uniform_depth("hello", 2, inside_out=False, str_depth=1)
        assert result == ["hello"]
    
    def test_bytes_depth_zero_atomic(self):
        """With bytes_depth=0, bytes are atomic (depth 0)."""
        result = ensure_uniform_depth(b"hello", 2, inside_out=False, bytes_depth=0)
        assert result == [[b"hello"]]
    
    def test_bytes_depth_one_sequence(self):
        """With bytes_depth=1, bytes are sequences (depth 1)."""
        result = ensure_uniform_depth(b"hello", 1, inside_out=False, bytes_depth=1)
        assert result == b"hello"
        
        result = ensure_uniform_depth(b"hello", 2, inside_out=False, bytes_depth=1)
        assert result == [b"hello"]
    
    def test_string_inside_out_mode(self):
        """String depth handling in inside-out mode."""
        # str_depth=0: string is atomic, wrap to depth 2
        result = ensure_uniform_depth("test", 2, inside_out=True, str_depth=0)
        assert result == [["test"]]
        
        # str_depth=1: string is at depth 1, wrap once more for depth 2
        result = ensure_uniform_depth("test", 2, inside_out=True, str_depth=1)
        assert result == ["test"]
    
    def test_bytes_inside_out_mode(self):
        """Bytes depth handling in inside-out mode."""
        # bytes_depth=0: bytes is atomic, wrap to depth 2
        result = ensure_uniform_depth(b"test", 2, inside_out=True, bytes_depth=0)
        assert result == [[b"test"]]
        
        # bytes_depth=1: bytes is at depth 1, wrap once more for depth 2
        result = ensure_uniform_depth(b"test", 2, inside_out=True, bytes_depth=1)
        assert result == [b"test"]
    
    def test_mixed_strings_and_containers(self):
        """Test lists containing strings with str_depth."""
        data = ["a", "b", "c"]
        
        # With str_depth=0, strings are atomic (depth 0), list adds 1 -> depth 1, wrap to 2
        result = ensure_uniform_depth(data, 2, inside_out=False, str_depth=0)
        assert result == [["a", "b", "c"]]
        
        # With str_depth=1, strings are at depth 1, list adds 1 -> depth 2, already correct
        result = ensure_uniform_depth(data, 2, inside_out=False, str_depth=1)
        assert result == ["a", "b", "c"]


# ============================================================================
# Tests for ERROR_ON_EXTRA policy
# ============================================================================

class TestErrorOnExtraPolicy:
    """Test the ERROR_ON_EXTRA unwrap policy (alias of STRICT)."""
    
    def test_error_on_extra_single_item(self):
        """ERROR_ON_EXTRA accepts single-item containers."""
        result = ensure_uniform_depth(
            [[1]], 1, inside_out=False,
            unwrap_policy=UnwrapPolicy.ERROR_ON_EXTRA
        )
        assert result == [1]
    
    def test_error_on_extra_multiple_items(self):
        """ERROR_ON_EXTRA errors on multiple items."""
        with pytest.raises(ValueError, match="Expected single-item container"):
            ensure_uniform_depth(
                [[1], [2]], 1, inside_out=False,
                unwrap_policy=UnwrapPolicy.ERROR_ON_EXTRA
            )
    
    def test_error_on_extra_empty_container(self):
        """ERROR_ON_EXTRA errors on empty containers."""
        with pytest.raises(ValueError, match="Cannot reduce depth|found 0 items"):
            ensure_uniform_depth(
                [[]], 0, inside_out=False,
                unwrap_policy=UnwrapPolicy.ERROR_ON_EXTRA
            )
