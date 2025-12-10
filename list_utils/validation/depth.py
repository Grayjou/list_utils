

from typing import Any
from .type_checks import is_strict_container

def get_max_depth(x:Any, str_depth = 0, bytes_depth = 0, *, depth_of_dict_values: bool = False) -> int:
    """
    Recursively determine the maximum depth of nested containers in x.
    Strings and bytes are not considered containers for this purpose.
    If you might consider strings or bytes as containers, set str_depth or bytes_depth = 1.
    Args:
        x: The input to check.
        str_depth: Depth to return when encountering a string (0 = not a container, 1 = treat as container).
        bytes_depth: Depth to return when encountering bytes (0 = not a container, 1 = treat as container).
        depth_of_dict_values: If True, iterate over dict values instead of keys when calculating depth.
    Returns:
        The maximum depth of nested containers.
    """
    if isinstance(x, str):
        return str_depth
    if isinstance(x, bytes):
        return bytes_depth
    if not is_strict_container(x):
        return 0
    
    # For dicts, optionally iterate over values instead of keys
    if isinstance(x, dict) and depth_of_dict_values:
        items = x.values()
    else:
        items = x
    
    return 1 + max((get_max_depth(item, str_depth, bytes_depth, depth_of_dict_values=depth_of_dict_values) for item in items), default=0)

def is_depth_at_least(x:Any, depth:int, str_depth = 0, bytes_depth = 0, *, depth_of_dict_values: bool = False) -> bool:
    """
    Check if the depth of nested containers in x is at least the specified depth.
    Strings and bytes are not considered containers for this purpose.
    If you might consider strings or bytes as containers, set str_depth or bytes_depth = 1.
    Args:
        x: The input to check.
        depth: The minimum depth to check against.
        str_depth: Depth to return when encountering a string (0 = not a container, 1 = treat as container).
        bytes_depth: Depth to return when encountering bytes (0 = not a container, 1 = treat as container).
        depth_of_dict_values: If True, iterate over dict values instead of keys when calculating depth.
    Returns:
        True if the depth of nested containers is at least the specified depth, False otherwise.
    """
    if depth <= 0:
        return True

    if isinstance(x, str):
        return str_depth >= depth
    if isinstance(x, bytes):
        return bytes_depth >= depth
    if not is_strict_container(x):
        return False

    # For dicts, optionally iterate over values instead of keys
    if isinstance(x, dict) and depth_of_dict_values:
        items = x.values()
    else:
        items = x

    for item in items:
        if is_depth_at_least(item, depth - 1, str_depth, bytes_depth, depth_of_dict_values=depth_of_dict_values):
            return True

    return False

