from pathlib import Path

from ml_platform.data.sources import KaggleSource


DATASET = "camnugent/california-housing-prices"

OUTPUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "ca_house_prediction"
    / "data"
)


def main() -> None:
    """Download the California Housing dataset."""

    source = KaggleSource(DATASET)

    output_path = source.fetch(OUTPUT_DIR)

    print(f"Dataset downloaded to: {output_path}")


if __name__ == "__main__":
    main()