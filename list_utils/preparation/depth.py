from ..validation.depth import get_max_depth
from ..validation.type_checks import is_strict_container
from typing import Any, List, Literal, Dict, Union
from enum import Enum, auto


class UnwrapPolicy(Enum):
    """Policy for unwrapping containers during depth normalization."""
    STRICT = auto()         # Require exactly one item → error otherwise
    IGNORE_EXTRA = auto()   # Ignore all but first item
    MERGE = auto()          # Flatten the container into a single merged layer
    ERROR_ON_EXTRA = auto() # Error only when >1 items (alias of STRICT)


def shed_layer(obj, ignore_extra: bool = True) -> Any:
    """
    Remove one layer from a container, returning the first or only item.
    
    Args:
        obj: The container to unwrap
        ignore_extra: If True, ignore extra items; if False, error on multiple items
    
    Returns:
        The first or only item from the container
    
    Raises:
        ValueError: If container is empty, non-container, or has multiple items (when ignore_extra=False)
    """
    if not is_strict_container(obj):
        raise ValueError("Cannot unpack from a non-container object")
    
    try:
        size = len(obj)
    except TypeError:
        # Some iterables don't have len(), just try to iterate
        try:
            return next(iter(obj))
        except StopIteration:
            raise ValueError("Cannot unpack from an empty container")
    
    if size == 0:
        raise ValueError("Cannot unpack from an empty container")
    
    if size > 1 and not ignore_extra:
        raise ValueError(f"Cannot unpack: container has {size} items")
    
    return next(iter(obj))


def add_list_layer(obj: Any) -> List[Any]:
    """Wrap an object in a list."""
    return [obj]


def ensure_uniform_depth(
    x: Any,
    depth: int,
    *,
    inside_out: bool = False,
    depth_of_dict_values: bool = False,
    unwrap_policy: UnwrapPolicy = UnwrapPolicy.STRICT,
    str_depth: Literal[0, 1] = 0,
    bytes_depth: Literal[0, 1] = 0,
) -> Any:
    """
    Ensure that `x` has exactly `depth` layers of containers.

    Args:
        x: The input to adjust.
        depth: The target depth of nested containers.
        inside_out: If True, adjust depth bottom-up (no get_max_depth needed).
                   If False, adjust depth top-down (requires get_max_depth).
        depth_of_dict_values: If True, consider dict values when calculating depth.
        unwrap_policy: Policy for unwrapping containers when depth > target depth.
                      Only applies when inside_out=False.
        str_depth: Depth to assign to non-empty strings (0 = atomic, 1 = sequence).
        bytes_depth: Depth to assign to non-empty bytes (0 = atomic, 1 = sequence).
    
    Returns:
        The input x adjusted to the specified depth.
    
    Examples:
        >>> ensure_uniform_depth("aaa", 2, inside_out=False)
        [["aaa"]]
        
        >>> ensure_uniform_depth([[1], [2]], 3, inside_out=True)
        [[[1]], [[2]]]
        
        >>> ensure_uniform_depth([[1]], 1, inside_out=False)
        [1]
    
    Raises:
        ValueError: If depth is negative or adjustment is impossible according to policy.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")

    if inside_out:
        return _ensure_depth_inside_out(x, depth, str_depth, bytes_depth)

    return _ensure_depth_outside_in(
        x,
        depth,
        depth_of_dict_values=depth_of_dict_values,
        unwrap_policy=unwrap_policy,
        str_depth=str_depth,
        bytes_depth=bytes_depth,
    )


# --------------------------------------------------------
# Inside-Out Implementation
# --------------------------------------------------------

def _ensure_depth_inside_out(
    x: Any, 
    depth: int,
    str_depth: Literal[0, 1],
    bytes_depth: Literal[0, 1]
) -> Any:
    """
    Enforce depth from the inside outward. No global depth check.
    
    This version always succeeds unless the object is not unpackable,
    because each node fixes its depth locally.
    """
    if depth == 0:
        # At depth 0 the object is returned as-is.
        return x

    # Handle strings and bytes according to their configured depth
    if isinstance(x, str):
        if str_depth == 0:
            # String is atomic, wrap it to target depth
            return _wrap_to_depth(x, depth)
        else:
            # String is sequence-like at depth 1, might need additional wrapping
            if depth == 1:
                return x
            else:
                return _wrap_to_depth(x, depth - 1)
    
    if isinstance(x, bytes):
        if bytes_depth == 0:
            # Bytes is atomic, wrap it to target depth
            return _wrap_to_depth(x, depth)
        else:
            # Bytes is sequence-like at depth 1, might need additional wrapping
            if depth == 1:
                return x
            else:
                return _wrap_to_depth(x, depth - 1)

    # Non-container → must wrap until depth is satisfied.
    if not is_strict_container(x):
        return _wrap_to_depth(x, depth)

    # Container → recurse into each element with (depth-1)
    return [
        _ensure_depth_inside_out(item, depth - 1, str_depth, bytes_depth)
        for item in x
    ]


# --------------------------------------------------------
# Outside-In Implementation
# --------------------------------------------------------

def _ensure_depth_outside_in(
    x: Any,
    depth: int,
    *,
    depth_of_dict_values: bool,
    unwrap_policy: UnwrapPolicy,
    str_depth: Literal[0, 1],
    bytes_depth: Literal[0, 1],
) -> Any:
    """
    Adjust depth from the outside: uniform structure required.
    Calls get_max_depth only once and then applies wrapping/unwrapping.
    """
    current_depth = get_max_depth(
        x,
        str_depth=str_depth,
        bytes_depth=bytes_depth,
        depth_of_dict_values=depth_of_dict_values
    )

    # Too shallow → wrap until deep enough
    if current_depth < depth:
        return _wrap_to_depth(x, depth - current_depth)

    # Too deep → unwrap
    if current_depth > depth:
        try:
            return _unwrap_to_depth(x, current_depth - depth, unwrap_policy)
        except ValueError as e:
            raise ValueError(
                f"Cannot reduce depth from {current_depth} to {depth}: {e}"
            ) from e

    # Exact depth → recursively ensure each layer is consistent
    return _fix_exact_depth(x, depth, str_depth, bytes_depth)


# --------------------------------------------------------
# Helper: Wrap Logic
# --------------------------------------------------------

def _wrap_to_depth(x: Any, layers: int) -> Any:
    """
    Wrap x in `layers` list-structures.
    """
    for _ in range(layers):
        x = add_list_layer(x)
    return x


# --------------------------------------------------------
# Helper: Unwrap Logic
# --------------------------------------------------------

def _unwrap_to_depth(x: Any, layers: int, policy: UnwrapPolicy) -> Any:
    """
    Unwrap `layers` levels by removing outermost container layers.
    Behavior depends on the unwrap policy.
    """
    for _ in range(layers):
        x = _unwrap_one_layer(x, policy)
    return x


def _unwrap_one_layer(x: Any, policy: UnwrapPolicy) -> Any:
    """
    Remove one container layer according to the specified policy.
    """
    if not is_strict_container(x):
        raise ValueError(f"Cannot unwrap non-container object of type {type(x).__name__}")

    try:
        size = len(x)
    except TypeError:
        # Fallback for iterables without len()
        size = sum(1 for _ in x)

    # STRICT and ERROR_ON_EXTRA handling
    if policy in (UnwrapPolicy.STRICT, UnwrapPolicy.ERROR_ON_EXTRA):
        if size != 1:
            raise ValueError(
                f"Expected single-item container, found {size} items "
                f"(unwrap policy: {policy.name})"
            )
        return next(iter(x))

    # IGNORE_EXTRA → drop everything but the first element
    if policy is UnwrapPolicy.IGNORE_EXTRA:
        try:
            return next(iter(x))
        except StopIteration:
            raise ValueError("Cannot unwrap empty container")

    # MERGE → flatten the container as a single layer
    if policy is UnwrapPolicy.MERGE:
        merged = []
        for item in x:
            # If item is a container, extend; else append
            if is_strict_container(item):
                merged.extend(item)
            else:
                merged.append(item)
        return merged

    raise RuntimeError(f"Unknown unwrap policy: {policy}")


# --------------------------------------------------------
# Helper: Fix exact depth
# --------------------------------------------------------

def _fix_exact_depth(
    x: Any, 
    depth: int,
    str_depth: Literal[0, 1],
    bytes_depth: Literal[0, 1]
) -> Any:
    """
    Even if depth is correct, children may have wrong depths.
    Recursively ensure structure is consistent.
    """
    if depth == 0:
        return x
    
    # Handle strings and bytes at their configured depth
    if isinstance(x, str) and str_depth == 1 and depth == 1:
        return x
    if isinstance(x, bytes) and bytes_depth == 1 and depth == 1:
        return x

    if not is_strict_container(x):
        raise ValueError(
            f"Invalid structure: expected container at depth {depth}, "
            f"got {type(x).__name__}"
        )

    return [
        _fix_exact_depth(item, depth - 1, str_depth, bytes_depth)
        for item in x
    ]


