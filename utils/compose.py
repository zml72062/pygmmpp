"""
compose.py - Compose multiple transformations on graph data.
"""

from typing import Callable, Iterable, Any


def compose(func_list: Iterable[Callable]):
    def func(x: Any) -> Any:
        for f in func_list:
            x = f(x)
        return x
    return func
