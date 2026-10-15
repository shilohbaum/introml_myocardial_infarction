"""Simple access to the supplied demographics and point clouds."""

from pathlib import Path

import numpy as np
import pandas as pd


class CardiacDataLoader:
    """Read one existing data folder and load its point clouds on demand.

    ``demographics`` is the original CSV as a DataFrame indexed by sample ID.
    Use ``samples_dir`` when the point clouds live outside the data folder.
    """

    def __init__(
        self, data_dir: str | Path, *, samples_dir: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        training = (self.data_dir / "mixed_demographics.csv").is_file()
        csv_name = "mixed_demographics.csv" if training else "test.csv"
        self.demographics = pd.read_csv(self.data_dir / csv_name, index_col=0)
        self.samples_dir = (
            Path(samples_dir).expanduser() if samples_dir is not None
            else self.data_dir / ("mixed_samples" if training else "test_samples")
        )
    def __len__(self) -> int:
        return len(self.demographics)

    def load_point_cloud(
        self, sample_id: int | str, *, mmap_mode: str | None = None,
    ) -> np.ndarray:
        """Load the original array; use ``mmap_mode='r'`` to read frames on demand."""
        row = self.demographics.loc[sample_id]
        path = self.samples_dir / f"{sample_id}.npy"
        if not path.is_file():
            if self.samples_dir.name == "mixed_samples":
                path = self.samples_dir / f"{sample_id}{row['MI']}.npy"
            else:
                # Test clouds use row positions (0–99), while CSV IDs are 900–999.
                position = self.demographics.index.get_loc(sample_id)
                path = self.samples_dir / f"{position}.npy"
        return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)

    def load_all(self, *, mmap_mode: str | None = "r") -> pd.DataFrame:
        """Return demographics and original arrays in a ``point_cloud`` column.

        One row per sample, indexed by sample ID in demographic row order.
        Each array includes every frame, XYZ coordinates, and surface codes.
        Arrays are memory mapped read-only by default; use ``mmap_mode=None``
        to load them fully into RAM. No frame selection or flattening is applied.
        """
        data = self.demographics.copy()
        data["point_cloud"] = pd.Series(
            [self.load_point_cloud(sample_id, mmap_mode=mmap_mode) for sample_id in data.index],
            index=data.index, dtype=object,
        )
        return data

    def __getitem__(self, index: int) -> dict:
        """Return demographics and a point cloud by row position, e.g. loader[0]."""
        row = self.demographics.iloc[index]
        return {
            "sample_id": row.name,
            "demographics": row,
            "point_cloud": self.load_point_cloud(row.name),
        }

