"""Context manager to temporarily switch workspace using config_manager.

This allows backward-compatible use of `with WorkspaceManager(...)` syntax
while relying internally on the centralized config_manager system.

Note:
    This class is maintained for backward compatibility. For new code,
    prefer using `config_manager.enter_workspace(...)` and `exit_workspace()`.
"""

import warnings
from contextlib import ContextDecorator
from types import TracebackType

from arpes.configuration.manager import config_manager

__all__ = ["WorkspaceManager"]


class WorkspaceManager(ContextDecorator):
    """[DEPRECATED] Context manager for temporary workspace switching, using config_manager.

    This is provided for backward compatibility. Prefer using:

        >>> config_manager.enter_workspace("project")
        >>> # do stuff
        >>> config_manager.exit_workspace()

    Example:
        >>> with WorkspaceManager("another_project"):
        ...     file = load_data(5)

    You can also use it as a decorator:

        >>> @WorkspaceManager("project")
        ... def do_stuff():
        ...     ...
    """

    def __init__(self, workspace_name: str = "") -> None:
        """Initialize the WorkspaceManager.

        This method sets up the initial state for managing the workspace,
        including loading configuration, preparing internal data structures,
        and performing any necessary setup required for workspace operations.

        Args:
            workspace_name (str): The path to the workspace directory.
        """
        self.workspace_name = workspace_name
        self._active = bool(workspace_name)

    def __enter__(self) -> "WorkspaceManager":
        """Enter the runtime context for the WorkspaceManager.

        Returns:
            WorkspaceManager: The WorkspaceManager instance itself.
        """
        if self._active:
            warnings.warn(
                "WorkspaceManager is deprecated. Use config_manager.enter_workspace(...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            config_manager.enter_workspace(self.workspace_name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context for the WorkspaceManager.

        Handles cleanup operations and resource deallocation when
        exiting the context manager block.
        """
        if self._active:
            config_manager.exit_workspace()
