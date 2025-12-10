
from typing import Any, List
from collections.abc import Container


from enum import Enum, auto

class UnwrapPolicy(Enum):
    STRICT = auto()        # require exactly one item → error otherwise
    IGNORE_EXTRA = auto()  # ignore all but first item
    MERGE = auto()         # flatten the container into a single merged layer
    ERROR_ON_EXTRA = auto()# error only when >1 items (same as STRICT but semantic)


def unwrap_to_first_layer(x: Any, convert_to_list: bool = False) -> List[Any]:
    # If x is not a list, return it as a single-element list
    if convert_to_list:
        if isinstance(x, Container):
            return list(x)

    if not isinstance(x, list):
        return [x]
    
    # If x is a list with more than one element, return it
    if len(x) > 1:
        return x
    
    # If x is an empty list, return it as is (no deeper layer)
    if len(x) == 0:
        return x
    
    # If x has exactly one element, recursively unwrap
    return unwrap_to_first_layer(x[0], convert_to_list=convert_to_list)


def is_strict_container(x):
    return isinstance(x, Container) and not isinstance(x, (str, bytes))
