
from collections.abc import Generator, Iterable, Sized
from typing import Type, Union

def is_strict_container(x):
    return (
        isinstance(x, Iterable)
        and isinstance(x, Sized)
        and not isinstance(x, (str, bytes, Generator, range))
    )

def enlist_type(items: Union[Type, Iterable[Type]]) -> list:
    if isinstance(items, type):
        return [items]
    elif isinstance(items, Iterable):
        return list(items)
    else:
        raise TypeError("Input must be a type or an iterable of types.")

def validate_monolist(*items, monotype: Union[Iterable[Type], Type, None] = None) -> None:
    """
    Validates that all items are of the same type, within the allowed `monotype` set (if provided).
    
    If `monotype` is given, it should be a single type or an iterable of acceptable types.
    Raises TypeError if any item differs from the determined monotype.
    """
    if len(items) <= 1:
        return

    monotype = enlist_type(monotype) if monotype is not None else [type(items[0])]
    if not all(isinstance(t, type) for t in monotype):
        raise TypeError("All elements in 'monotype' must be types.")

    first_item = items[0]
    try:
        type_to_verify = next(_type for _type in monotype if isinstance(first_item, _type))
    except StopIteration:
        raise TypeError(f"First item {first_item} is not an instance of any allowed monotypes: {monotype}")

    for i, item in enumerate(items[1:], start=1):
        if not isinstance(item, type_to_verify):
            raise TypeError(f"Item {i} is of type {type(item).__name__}, expected {type_to_verify.__name__}.")