"""
Pagination types for list endpoints.
"""
from __future__ import annotations

from typing import Generic, TypeVar, List, Optional, Iterator

T = TypeVar("T")


class SyncPage(Generic[T]):
    """A single page of results from a list endpoint."""

    data: List[T]
    total: Optional[int]

    def __init__(self, data: List[T], total: Optional[int] = None) -> None:
        self.data = data
        self.total = total

    @property
    def has_more(self) -> bool:
        if self.total is None:
            return len(self.data) > 0
        return False  # Determined by caller

    def __iter__(self) -> Iterator[T]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"SyncPage(count={len(self.data)}, total={self.total})"
