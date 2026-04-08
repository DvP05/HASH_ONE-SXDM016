import pandas as pd

from tools.ml_tools import train_model


def test_train_model_multiclass_string_target():
    X = pd.DataFrame({
        "f1": list(range(1, 16)),
        "f2": [3, 4, 5, 6, 7] * 3,
    })
    y = pd.Series(["l", "n", "h"] * 5)

    result = train_model(X=X, y=y, model_name="random_forest", task_type="classification")

    assert result["cv_mean"] == result["cv_mean"]  # not NaN
    assert result["target_encoder_classes"] == ["h", "l", "n"]
