from typing import Any

from ml_platform.serving.contracts import Predictor


class FakePredictor:
    def predict(self, inputs: list[dict[str, Any]]) -> list[Any]:
        return [42.0 for _ in inputs]


def test_predictor_contract() -> None:
    predictor: Predictor = FakePredictor()

    result = predictor.predict(
        [
            {"feature": 1.0},
            {"feature": 2.0},
        ]
    )

    assert result == [42.0, 42.0]