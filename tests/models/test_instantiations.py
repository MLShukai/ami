from typing import Any, Callable, get_type_hints

from ami.models.instantiations import InstantiationReturnType, image_vae
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper


def assert_return_type_annotation(func: Callable) -> None:
    hints = get_type_hints(func)
    assert "return" in hints, "Function is missing return type annotation"

    expected_type = InstantiationReturnType
    actual_type = hints["return"]

    assert (
        actual_type == expected_type
    ), f"Return type annotation mismatch. Expected: {expected_type}, Actual: {actual_type}"


def assert_return_type(return_value: InstantiationReturnType | Any) -> None:

    assert isinstance(return_value, dict)
    for key, value in return_value.items():
        assert isinstance(key, ModelNames), "Key is not `ModelNames` item!"
        assert isinstance(value, ModelWrapper | ModelNames), "Value must be ModelWrapper or ModelNames"


def test_image_vae():
    assert_return_type_annotation(image_vae)
    assert_return_type(image_vae())
