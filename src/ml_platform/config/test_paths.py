from ml_platform.config import (
    DATA_DIR,
    MODELS_DIR,
    ROOT_DIR,
)


def test_root_directory_exists():
    assert ROOT_DIR.exists()


def test_data_directory_path():
    assert DATA_DIR.name == "data"


def test_models_directory_path():
    assert MODELS_DIR.name == "models"
