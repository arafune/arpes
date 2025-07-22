from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arpes.provenance import Provenance


def unlayer(
    prov: Provenance | None | str,
) -> tuple[list[Provenance] | list[str], Provenance | str | None]:
    """Extracts the current layer of provenance and identifies the next parent.

    This helper function takes a single provenance record (or string/None)
    and separates it into the current layer's information and its
    immediate parent's provenance. It handles various input types and
    logs warnings for unexpected formats. This is particularly useful
    for traversing a chain of provenance records, allowing for
    step-by-step processing of historical data.

    Parameters:
        prov (Provenance | str | None): The current provenance record to unlayer.
            This can be a dictionary (representing a 'Provenance' object),
            a string (which is considered a legacy format), or None.

    Returns:
        tuple[list[Provenance], Provenance | str | None]: A tuple containing:

        - **current_layer** (list[Provenance]): A list containing the
            current layer's `Provenance` dictionary. If `prov` was a
            string, this list will contain that string. If `prov` was
            `None`, this list will be empty. This represents the
            information directly associated with the current step
            in the provenance chain.
        - **next_parent** (Provenance | str | None): The `parents_provenance`
            of the current layer. This represents the next record to
            process in the history chain, effectively pointing to the
            previous step in the lineage. It can be a `Provenance`
            dictionary, a string (legacy format for the parent), or `None`
            if there are no further parents.

    Warns:
        UserWarning:
            - If `prov` is of type string. This indicates that the
              provenance record is in an older, less structured format,
              and it's advised to convert it to a dictionary-based
              `Provenance` object for better compatibility and data integrity.
            - If `parents_provenance` within the current `prov` dictionary
              is a list, indicating that multiple parents were recorded.
              In such cases, only the *first* parent in the list is
              retained for processing, and a warning is issued to highlight
              this potential loss of information regarding other parents.
              This function is designed to follow a single lineage.
    """
    if prov is None:
        return [], None  # tuple[list[Incomplete] | None]
    if isinstance(prov, str):
        warnings.warn("provenance should be dict type object.", stacklevel=2)
        return [prov], None
    first_layer: Provenance = copy.copy(prov)

    rest = first_layer.pop("parents_provenance", None)
    if isinstance(rest, list):
        warnings.warn(
            "Encountered multiple parents in history extraction, throwing away all but the first.",
            stacklevel=2,
        )
        rest = rest[0] if rest else None

    return [first_layer], rest


def unwrap_provenance(prov: Provenance | None) -> list[Provenance | str]:
    """Recursively unwraps nested provenance records into a flat list.

    This internal helper function uses `unlayer` to iteratively extract
    provenance steps, building a linear history list. It's the core
    recursive logic for flattening the provenance tree.

    Parameters:
        prov (Provenance | None): The current provenance record to process.

    Returns:
        list[Provenance]: A list of Provenance dictionaries representing
        the history from the current `prov` down to the oldest parent.
    """
    if prov is None:
        return []

    first, rest = unlayer(
        prov,
    )

    return first + unwrap_provenance(rest)
