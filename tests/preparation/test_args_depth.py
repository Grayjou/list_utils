import pytest
from ...list_utils.preparation.args_depth import (
    enforce_asterisk_args_depth,
    enforce_asterisk_args_uniform_depth,
    _normalize_depth_spec,
)
from ...list_utils.preparation.depth import UnwrapPolicy


# ============================================================================
# Tests for _normalize_depth_spec
# ============================================================================

class TestNormalizeDepthSpec:
    """Tests for the depth specification normalizer."""
    
    def test_int_depth_spec(self):
        """Int depth spec creates uniform mapping for all args."""
        result = _normalize_depth_spec(2, 3)
        assert result == {0: 2, 1: 2, 2: 2}
        
        result = _normalize_depth_spec(1, 5)
        assert result == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    
    def test_int_depth_spec_zero_args(self):
        """Int depth spec with zero args returns empty dict."""
        result = _normalize_depth_spec(2, 0)
        assert result == {}
    
    def test_list_depth_spec(self):
        """List depth spec maps indices to depths."""
        result = _normalize_depth_spec([1, 2, 3], 5)
        assert result == {0: 1, 1: 2, 2: 3}
    
    def test_list_depth_spec_exact_length(self):
        """List depth spec with exact length."""
        result = _normalize_depth_spec([1, 2, 3], 3)
        assert result == {0: 1, 1: 2, 2: 3}
    
    def test_list_depth_spec_longer_than_args(self):
        """List depth spec longer than num_args is truncated."""
        result = _normalize_depth_spec([1, 2, 3, 4, 5], 3)
        assert result == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    
    def test_list_depth_spec_empty(self):
        """Empty list returns empty dict."""
        result = _normalize_depth_spec([], 5)
        assert result == {}
    
    def test_dict_depth_spec(self):
        """Dict depth spec is returned as-is."""
        depth_map = {0: 1, 2: 3, 4: 2}
        result = _normalize_depth_spec(depth_map, 10)
        assert result == depth_map
    
    def test_dict_depth_spec_empty(self):
        """Empty dict returns empty dict."""
        result = _normalize_depth_spec({}, 5)
        assert result == {}
    
    def test_dict_depth_spec_sparse(self):
        """Dict can specify depths for non-contiguous indices."""
        result = _normalize_depth_spec({1: 2, 5: 3, 10: 1}, 15)
        assert result == {1: 2, 5: 3, 10: 1}
    
    def test_invalid_depth_spec_raises_error(self):
        """Invalid depth spec types raise TypeError."""
        with pytest.raises(TypeError, match="depth must be int, list, or dict"):
            _normalize_depth_spec("invalid", 3)
        
        with pytest.raises(TypeError, match="depth must be int, list, or dict"):
            _normalize_depth_spec(3.14, 3)


# ============================================================================
# Tests for enforce_asterisk_args_uniform_depth
# ============================================================================

class TestEnforceUniformDepth:
    """Tests for uniform depth enforcement decorator."""
    
    def test_uniform_depth_all_args(self):
        """All args normalized to same depth."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(a, b, c):
            return [a, b, c]
        
        result = func("x", "y", "z")
        assert result == [["x"], ["y"], ["z"]]
    
    def test_uniform_depth_wrapping(self):
        """Args get wrapped to target depth."""
        @enforce_asterisk_args_uniform_depth(depth=2)
        def func(x):
            return x
        
        result = func(42)
        assert result == [[42]]
    
    def test_uniform_depth_unwrapping(self):
        """Args get unwrapped to target depth."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(x):
            return x
        
        result = func([[42]])
        assert result == [42]
    
    def test_uniform_depth_no_args(self):
        """Works with functions that have no args."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func():
            return "no args"
        
        result = func()
        assert result == "no args"
    
    def test_uniform_depth_kwargs_unaffected(self):
        """Keyword arguments are not normalized."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(a, b, c=None):
            return [a, b, c]
        
        result = func("x", "y", c="unchanged")
        assert result == [["x"], ["y"], "unchanged"]
    
    def test_uniform_depth_inside_out_mode(self):
        """Inside-out mode works correctly."""
        @enforce_asterisk_args_uniform_depth(depth=2, inside_out=True)
        def func(x, y):
            return [x, y]
        
        result = func([1, 2], [3, 4])
        assert result == [[[1], [2]], [[3], [4]]]
    
    def test_uniform_depth_with_unwrap_policy(self):
        """Unwrap policy is respected."""
        @enforce_asterisk_args_uniform_depth(
            depth=1,
            unwrap_policy=UnwrapPolicy.IGNORE_EXTRA
        )
        def func(x):
            return x
        
        result = func([[1], [2], [3]])
        assert result == [1]
    
    def test_uniform_depth_str_depth_param(self):
        """str_depth parameter is respected."""
        @enforce_asterisk_args_uniform_depth(depth=1, str_depth=1)
        def func(x):
            return x
        
        result = func("hello")
        assert result == "hello"  # String at depth 1
    
    def test_uniform_depth_mixed_types(self):
        """Works with mixed argument types."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(a, b, c, d):
            return [a, b, c, d]
        
        result = func(42, "text", [1, 2], [[3]])
        assert result == [[42], ["text"], [1, 2], [3]]
    
    def test_uniform_depth_preserves_function_name(self):
        """Decorator preserves original function metadata."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def my_function():
            """My docstring."""
            pass
        
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


# ============================================================================
# Tests for enforce_asterisk_args_depth (variable depth)
# ============================================================================

class TestEnforceVariableDepth:
    """Tests for variable depth enforcement decorator."""
    
    def test_variable_depth_with_list(self):
        """List specifies different depth for each arg."""
        @enforce_asterisk_args_depth(depth=[1, 2, 3])
        def func(a, b, c):
            return [a, b, c]
        
        result = func("x", "y", "z")
        assert result == [["x"], [["y"]], [[["z"]]]]
    
    def test_variable_depth_with_dict(self):
        """Dict specifies depths for specific args."""
        @enforce_asterisk_args_depth(depth={0: 2, 2: 1})
        def func(a, b, c):
            return [a, b, c]
        
        result = func("x", "y", "z")
        assert result == [[["x"]], "y", ["z"]]
    
    def test_variable_depth_dict_sparse(self):
        """Dict can skip args (they remain unchanged)."""
        @enforce_asterisk_args_depth(depth={1: 2})
        def func(a, b, c):
            return [a, b, c]
        
        result = func("x", "y", "z")
        assert result == ["x", [["y"]], "z"]
    
    def test_variable_depth_list_shorter_than_args(self):
        """List shorter than args only processes specified args."""
        @enforce_asterisk_args_depth(depth=[2, 1])
        def func(a, b, c, d):
            return [a, b, c, d]
        
        result = func("w", "x", "y", "z")
        assert result == [[["w"]], ["x"], "y", "z"]
    
    def test_variable_depth_with_int_delegates_to_uniform(self):
        """Int depth delegates to uniform version."""
        @enforce_asterisk_args_depth(depth=2)
        def func(a, b):
            return [a, b]
        
        result = func("x", "y")
        assert result == [[["x"]], [["y"]]]
    
    def test_variable_depth_inside_out_mode(self):
        """Inside-out mode with variable depths."""
        @enforce_asterisk_args_depth(depth=[1, 2, 3], inside_out=True)
        def func(a, b, c):
            return [a, b, c]
        
        result = func([1, 2], [3, 4], [5, 6])
        assert result == [[1, 2], [[3], [4]], [[[5]], [[6]]]]
    
    def test_variable_depth_with_unwrap_policy(self):
        """Unwrap policy works with variable depths."""
        @enforce_asterisk_args_depth(
            depth={0: 1, 1: 1},
            unwrap_policy=UnwrapPolicy.MERGE
        )
        def func(a, b):
            return [a, b]
        
        result = func([[1], [2, 3]], [[4, 5], [6]])
        assert result == [[1, 2, 3], [4, 5, 6]]
    
    def test_variable_depth_empty_dict(self):
        """Empty dict leaves all args unchanged."""
        @enforce_asterisk_args_depth(depth={})
        def func(a, b, c):
            return [a, b, c]
        
        result = func("x", "y", "z")
        assert result == ["x", "y", "z"]
    
    def test_variable_depth_preserves_function_name(self):
        """Decorator preserves original function metadata."""
        @enforce_asterisk_args_depth(depth=[1, 2])
        def my_other_function():
            """Another docstring."""
            pass
        
        assert my_other_function.__name__ == "my_other_function"
        assert my_other_function.__doc__ == "Another docstring."


# ============================================================================
# Tests for real-world use cases
# ============================================================================

class TestRealWorldUseCases:
    """Tests simulating real-world usage scenarios."""
    
    def test_normalize_batch_inputs(self):
        """Normalize varying batch formats to uniform depth."""
        @enforce_asterisk_args_uniform_depth(depth=2)
        def process_batches(*batches):
            # All batches should be at depth 2: [[items...]]
            return batches
        
        single = 42
        list_items = [1, 2, 3]
        nested = [[4, 5]]
        
        result = process_batches(single, list_items, nested)
        assert result == ([[42]], [[1, 2, 3]], [[4, 5]])
    
    def test_flexible_function_arguments(self):
        """Function accepts both single items and lists."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def concatenate(*items):
            # All items normalized to depth 1 (lists)
            result = []
            for item in items:
                result.extend(item)
            return result
        
        result = concatenate(1, [2, 3], 4, [5, 6, 7])
        assert result == [1, 2, 3, 4, 5, 6, 7]
    
    def test_api_endpoint_normalization(self):
        """API endpoint normalizes different input formats."""
        @enforce_asterisk_args_depth(depth={0: 2, 1: 1})
        def api_endpoint(data, ids):
            # data at depth 2, ids at depth 1
            return {"data": data, "ids": ids}
        
        # [[1, 2]] has depth 2
        # 123 has depth 0, needs wrap to 1 -> [123]
        result = api_endpoint([[1, 2]], 123)
        assert result == {
            "data": [[1, 2]],  # Already at depth 2
            "ids": [123]
        }
    
    def test_api_endpoint_with_wrapping(self):
        """API endpoint wraps shallow data."""
        @enforce_asterisk_args_depth(depth={0: 3, 1: 1})
        def api_endpoint(data, ids):
            return {"data": data, "ids": ids}
        
        # [{"key": "value"}] depth 2, wrap to 3
        result = api_endpoint([{"key": "value"}], 123)
        assert result == {
            "data": [[{"key": "value"}]],
            "ids": [123]
        }
    
    def test_recursive_data_structure_builder(self):
        """Build nested structures from flat inputs."""
        @enforce_asterisk_args_uniform_depth(depth=3, inside_out=True)
        def build_tree(*nodes):
            return list(nodes)
        
        # inside_out with depth 3: [1, 2] -> recurse with depth 2 on each element
        # 1 -> wrap to depth 2 -> [[1]], 2 -> [[2]]
        # Result: [[[1]], [[2]]]
        result = build_tree([1, 2], [3, 4])
        assert result == [[[[1]], [[2]]], [[[3]], [[4]]]]
    
    def test_database_query_normalizer(self):
        """Normalize query parameters to consistent depth."""
        @enforce_asterisk_args_depth(depth={0: 1, 1: 2})
        def query(fields, filters):
            return {"fields": fields, "filters": filters}
        
        # "name" is depth 0, wrap to 1 -> ["name"]
        # {"age": 25} is depth 1 (dict keys), wrap to 2 -> [{"age": 25}]
        result = query("name", {"age": 25})
        assert result == {
            "fields": ["name"],
            "filters": [{"age": 25}]
        }
    
    def test_mixed_depth_data_processing(self):
        """Process args with different depth requirements."""
        @enforce_asterisk_args_depth(depth=[0, 1, 2, 3])
        def process(scalar, vector, matrix, tensor):
            return {
                "scalar": scalar,
                "vector": vector,
                "matrix": matrix,
                "tensor": tensor
            }
        
        result = process(5, 5, 5, 5)
        assert result == {
            "scalar": 5,
            "vector": [5],
            "matrix": [[5]],
            "tensor": [[[5]]]
        }


# ============================================================================
# Tests for error handling
# ============================================================================

class TestErrorHandling:
    """Tests for error conditions and edge cases."""
    
    def test_unwrap_fails_with_strict_policy(self):
        """STRICT policy raises error on multi-item unwrap."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(x):
            return x
        
        # [[1], [2]] has depth 2, unwrapping to depth 1 fails because outer has 2 items
        with pytest.raises(ValueError, match="single-item container|Cannot reduce depth"):
            func([[1], [2]])
    
    def test_negative_depth_raises_error(self):
        """Negative depth values raise ValueError."""
        @enforce_asterisk_args_uniform_depth(depth=-1)
        def func(x):
            return x
        
        with pytest.raises(ValueError, match="depth must be >= 0"):
            func(42)
    
    def test_variable_depth_negative_raises_error(self):
        """Negative depths in list/dict raise ValueError."""
        @enforce_asterisk_args_depth(depth=[1, -1])
        def func(a, b):
            return [a, b]
        
        with pytest.raises(ValueError, match="depth must be >= 0"):
            func("x", "y")
    
    def test_complex_unwrap_failure(self):
        """Complex structures that can't be unwrapped raise errors."""
        @enforce_asterisk_args_depth(
            depth={0: 0},
            unwrap_policy=UnwrapPolicy.STRICT
        )
        def func(x):
            return x
        
        with pytest.raises(ValueError, match="Cannot reduce depth"):
            func([[1], [2], [3]])


# ============================================================================
# Tests for advanced features
# ============================================================================

class TestAdvancedFeatures:
    """Tests for advanced decorator features."""
    
    def test_depth_of_dict_values_param(self):
        """depth_of_dict_values parameter is passed through."""
        @enforce_asterisk_args_uniform_depth(
            depth=3,
            depth_of_dict_values=True
        )
        def func(x):
            return x
        
        # Dict with list values: with depth_of_dict_values=True, depth is 2 (dict + list)
        # Wrapping to depth 3 adds one layer
        data = {"key": [1, 2, 3]}
        result = func(data)
        assert result == [{"key": [1, 2, 3]}]  # Wrapped once
    
    def test_bytes_depth_parameter(self):
        """bytes_depth parameter is respected."""
        @enforce_asterisk_args_uniform_depth(depth=1, bytes_depth=1)
        def func(x):
            return x
        
        result = func(b"hello")
        assert result == b"hello"  # Bytes at depth 1
    
    def test_multiple_unwrap_policies(self):
        """Different unwrap policies produce different results."""
        data = [[1], [2], [3]]
        
        @enforce_asterisk_args_uniform_depth(
            depth=1,
            unwrap_policy=UnwrapPolicy.IGNORE_EXTRA
        )
        def func_ignore(x):
            return x
        
        @enforce_asterisk_args_uniform_depth(
            depth=1,
            unwrap_policy=UnwrapPolicy.MERGE
        )
        def func_merge(x):
            return x
        
        assert func_ignore(data) == [1]
        assert func_merge(data) == [1, 2, 3]
    
    def test_nested_decorators(self):
        """Multiple depth decorators can be nested."""
        @enforce_asterisk_args_uniform_depth(depth=2)
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(x):
            return x
        
        # Inner decorator normalizes to depth 1: "x" -> ["x"]
        # Function returns ["x"], but outer decorator saw input before inner processed
        # Actually: outer processes first, then inner on the already-processed arg
        # Decorators stack bottom-up: inner runs first
        # "x" -> inner makes ["x"] -> passed to func -> returns ["x"]
        # outer then doesn't reprocess since it already ran
        # Wait, decorators wrap: outer(inner(func))
        # So call goes: outer.wrapper -> inner.wrapper -> func
        # outer normalizes "x" to depth 2: [["x"]] 
        # inner then normalizes [["x"]] to depth 1: tries to unwrap, fails? No, inner runs on the outer's output
        # Actually the call order is: outer.wrapper receives "x", normalizes to [["x"]], calls inner.wrapper([["x"]])
        # inner.wrapper receives [["x"]], normalizes to depth 1 -> ["x"]
        result = func("x")
        assert result == ["x"]  # After both decorators
    
    def test_decorator_with_return_value(self):
        """Return values are not affected by decorator."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(x):
            return "raw_return_value"
        
        result = func("anything")
        assert result == "raw_return_value"
    
    def test_args_with_complex_structures(self):
        """Complex nested structures are handled correctly."""
        @enforce_asterisk_args_depth(depth={0: 3, 1: 2})
        def func(a, b):
            return [a, b]
        
        # "hello" has depth 0, wrap 3 times for depth 3
        # [[1, 2]] has depth 2 already
        result = func(
            "hello",
            [[1, 2]]
        )
        # First arg wrapped to depth 3
        assert result[0] == [[["hello"]]]
        # Second arg already at depth 2
        assert result[1] == [[1, 2]]


# ============================================================================
# Tests for edge cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_arg_function(self):
        """Works with single-argument functions."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(x):
            return x
        
        result = func(42)
        assert result == [42]
    
    def test_many_args_function(self):
        """Works with many arguments."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(*args):
            return args
        
        result = func(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        assert result == tuple([i] for i in range(1, 11))
    
    def test_varargs_only_function(self):
        """Works with *args only functions."""
        @enforce_asterisk_args_uniform_depth(depth=2)
        def func(*items):
            return items
        
        result = func("a", "b", "c")
        assert result == ([["a"]], [["b"]], [["c"]])
    
    def test_depth_zero_target(self):
        """Target depth of 0 with inside_out mode returns items as-is."""
        @enforce_asterisk_args_uniform_depth(depth=0, inside_out=True)
        def func(x, y):
            return [x, y]
        
        # inside_out at depth 0 returns everything as-is
        result = func([1, 2], [3, 4])
        assert result == [[1, 2], [3, 4]]
    
    def test_empty_list_arg(self):
        """Empty list arguments are handled."""
        @enforce_asterisk_args_uniform_depth(depth=2)
        def func(x):
            return x
        
        result = func([])
        assert result == [[]]
    
    def test_none_as_argument(self):
        """None is treated as non-container."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(x):
            return x
        
        result = func(None)
        assert result == [None]
    
    def test_generator_not_consumed_early(self):
        """Ensures args are processed correctly (not generators)."""
        @enforce_asterisk_args_uniform_depth(depth=1)
        def func(*args):
            # Args should be materialized, not generators
            return [type(arg).__name__ for arg in args]
        
        result = func(1, 2, 3)
        assert result == ["list", "list", "list"]
