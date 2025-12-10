from typing import Any, List, Tuple, Union, Generator, Dict
Numerable = Dict[int, Any] | List[Any] | Tuple[Any, ...] | Generator[Any, None, None]

def enumerate_container(
    container:Numerable
    ) -> Generator[Tuple[int, Any], None, None]:
    """Generate index-item pairs from a numerable container."""
    #Sort dictionary keys if container is a dict
    if isinstance(container, dict):
        for key in sorted(container.keys()):
            yield key, container[key]
    else:
        yield from enumerate(container)