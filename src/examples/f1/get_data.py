from examples.f1 import KAGGLE_DATASET, RAW_DATA_DIR
from ml_platform.data.sources import KaggleSource


def main() -> None:
    """Download the F1 dataset."""

    source = KaggleSource(KAGGLE_DATASET)

    output_path = source.fetch(RAW_DATA_DIR)

    print(f"Downloaded {KAGGLE_DATASET} to {output_path}")


if __name__ == "__main__":
    main()
