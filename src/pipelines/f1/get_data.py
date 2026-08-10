from ml_platform.data.sources import KaggleSource
from pipelines.f1 import KAGGLE_DATASET, RAW_DATA_DIR


def main() -> None:
    """Download the F1 dataset."""

    source = KaggleSource(KAGGLE_DATASET)

    output_path = source.fetch(RAW_DATA_DIR)

    print(f"Downloaded {KAGGLE_DATASET} to {output_path}")


if __name__ == "__main__":
    main()
