"""A lazy keep-alive `multiprocesssing.Pool`.

We keep a pool alive after one is requested at the cost of memory overhead
because otherwise pools are too slow due to heavy analysis imports (scipy, etc.).
"""
from __future__ import annotations

from multiprocessing import Pool, pool

__all__ = ["hot_pool"]


class HotPool:
    _pool: pool.Pool | None = None

    @property
    def pool(self) -> pool.Pool:
        """[TODO:summary].

        [TODO:description]

        Args:
            self ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """
        if self._pool is not None:
            return self._pool

        self._pool = Pool()
        return self._pool

    def __del__(self) -> None:
        """[TODO:summary].

        [TODO:description]

        Returns:
            [TODO:description]
        """
        if self._pool is not None:
            self._pool.close()
            self._pool = None


hot_pool = HotPool()
