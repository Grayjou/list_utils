from .depth import ensure_uniform_depth, UnwrapPolicy
from functools import wraps
from .iterating import enumerate_container, Numerable
from typing import Literal, Dict


def _normalize_depth_spec(depth_spec: Numerable, num_args: int) -> Dict[int, int]:
    """
    Convert depth specification to a dictionary mapping arg indices to depths.
    
    Args:
        depth_spec: Can be:
            - int: Same depth for all args
            - list: Depths for each arg (shorter list = fewer args processed)
            - dict: Explicit mapping of arg index to depth
        num_args: Total number of arguments
    
    Returns:
        Dictionary mapping arg indices to their target depths.
        Only includes indices that should be processed.
    
    Examples:
        >>> _normalize_depth_spec(2, 3)
        {0: 2, 1: 2, 2: 2}
        
        >>> _normalize_depth_spec([1, 2], 5)
        {0: 1, 1: 2}
        
        >>> _normalize_depth_spec({0: 1, 2: 3}, 5)
        {0: 1, 2: 3}
    """
    if isinstance(depth_spec, int):
        # All args get the same depth
        return {i: depth_spec for i in range(num_args)}
    
    elif isinstance(depth_spec, list):
        # Each arg gets corresponding depth from list
        # If list is shorter, only process those indices
        return {i: d for i, d in enumerate(depth_spec)}
    
    elif isinstance(depth_spec, dict):
        # Use as-is, only process specified indices
        return depth_spec
    
    else:
        raise TypeError(f"depth must be int, list, or dict, got {type(depth_spec).__name__}")



def enforce_asterisk_args_depth(
    depth: int | Numerable,
    inside_out: bool = False,
    depth_of_dict_values: bool = False,
    unwrap_policy: UnwrapPolicy = UnwrapPolicy.STRICT,
    str_depth: Literal[0, 1] = 0,
    bytes_depth: Literal[0, 1] = 0,
):
    
    """
    Decorator factory to enforce uniform depth on all `*args` of a function.
    
    This is the @ syntax version of enforce_asterisk_args_depth.
    
    Args:
        depth: The target depth for each argument, or a Numerable for per-arg depths.
        inside_out: Whether to enforce depth inside-out or outside-in.
        depth_of_dict_values: Whether to consider dict values when calculating depth.
        unwrap_policy: Policy for unwrapping when reducing depth.
        str_depth: Depth to assign to non-empty strings (0 = atomic, 1 = sequence).
        bytes_depth: Depth to assign to non-empty bytes (0 = atomic, 1 = sequence).
    
    Returns:
        A decorator that enforces argument depths.
    
    Example:
        >>> @enforce_asterisk_args_depth(depth=1)
        ... def my_func(x, y):
        ...     return x + y
        >>> my_func("a", "b")  # Both args normalized to depth 1
    """
    if isinstance(depth, int):
        return enforce_asterisk_args_uniform_depth(
            depth,
            inside_out=inside_out,
            depth_of_dict_values=depth_of_dict_values,
            unwrap_policy=unwrap_policy,
            str_depth=str_depth,
            bytes_depth=bytes_depth,
        )
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            depth_map = _normalize_depth_spec(depth, len(args))
            new_args = []
            for i, arg in enumerate_container(args):
                if i in depth_map:
                    target_depth = depth_map[i]
                    new_arg = ensure_uniform_depth(
                        arg,
                        target_depth,
                        inside_out=inside_out,
                        depth_of_dict_values=depth_of_dict_values,
                        unwrap_policy=unwrap_policy,
                        str_depth=str_depth,
                        bytes_depth=bytes_depth,
                    )
                    new_args.append(new_arg)
                else:
                    new_args.append(arg)
            return func(*new_args, **kwargs)
        return wrapper
    return decorator



def enforce_asterisk_args_uniform_depth(
    depth: int, # Uniform version is faster and specific
    inside_out: bool = False,
    depth_of_dict_values: bool = False,
    unwrap_policy: UnwrapPolicy = UnwrapPolicy.STRICT,
    str_depth: Literal[0, 1] = 0,
    bytes_depth: Literal[0, 1] = 0,
):

    """
    Decorator factory to enforce uniform depth on all `*args` of a function.
    
    This is the @ syntax version of enforce_asterisk_args_uniform_depth.
    
    Args:
        depth: The target depth for each argument.
        inside_out: Whether to enforce depth inside-out or outside-in.
        depth_of_dict_values: Whether to consider dict values when calculating depth.
        unwrap_policy: Policy for unwrapping when reducing depth.
        str_depth: Depth to assign to non-empty strings (0 = atomic, 1 = sequence).
        bytes_depth: Depth to assign to non-empty bytes (0 = atomic, 1 = sequence).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_args = [
                ensure_uniform_depth(
                    arg,
                    depth,
                    inside_out=inside_out,
                    depth_of_dict_values=depth_of_dict_values,
                    unwrap_policy=unwrap_policy,
                    str_depth=str_depth,
                    bytes_depth=bytes_depth,
                )
                for arg in args
            ]
            return func(*new_args, **kwargs)
        return wrapper
    return decorator