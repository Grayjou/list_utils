
from collections.abc import Generator, Iterable, Sized

def is_strict_container(x):
    return (
        isinstance(x, Iterable)
        and isinstance(x, Sized)
        and not isinstance(x, (str, bytes, Generator, range))
    )

