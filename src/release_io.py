"""Official output writers shared by release inference entry points."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def write_csv(
    frame: pd.DataFrame, path: Path, *, header: bool = True
) -> None:
    """Atomically write a deterministic UTF-8 CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(
                stream,
                index=False,
                header=header,
                lineterminator="\n",
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_submission(
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    order: np.ndarray,
    path: Path,
) -> None:
    """Write the official headerless query plus top-10 gallery format."""

    columns = {"query_id": query_ids}
    for rank in range(order.shape[1]):
        columns[f"gallery_id_{rank + 1}"] = gallery_ids[order[:, rank]]
    write_csv(pd.DataFrame(columns), path, header=False)


__all__ = ["write_csv", "write_submission"]
