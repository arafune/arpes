"""A lazy keep-alive `multiprocessing.Pool`.

Keep a pool alive after one is requested at the cost of memory overhead
because otherwise pools are too slow due to heavy analysis imports (scipy, etc.).
"""

from __future__ import annotations

from multiprocessing import Pool, pool

__all__ = ["hot_pool"]


class HotPool:
    _pool: pool.Pool | None = None

    @property
    def pool(self) -> pool.Pool:
        """Returns a pool object, creating it if necessary.

        This method lazily initializes a pool object and returns it. If the pool has
        already been created, it simply returns the existing one.

        Returns:
            pool.Pool: A pool object.
        """
        if self._pool is not None:
            return self._pool

        self._pool = Pool()
        return self._pool

    def __del__(self) -> None:
        """Cleans up resources when the object is deleted.

        This method ensures that the pool, if it exists, is closed before the object
        is destroyed to release any allocated resources.
        """
        if self._pool is not None:
            self._pool.close()
            self._pool = None


hot_pool = HotPool()
