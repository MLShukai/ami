from enum import Enum

from torch import Tensor
from typing_extensions import override

from .base_sensor import BaseSensor


class Modality(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"


class DictMultimodalSensor(BaseSensor[dict[Modality, Tensor]]):
    """Integrates multiple sensors in dictionary format.

    This class provides a way to manage multiple sensor inputs by
    organizing them in a dictionary where each key represents a modality
    type.
    """

    def __init__(self, sensors: dict[Modality, BaseSensor[Tensor]]) -> None:
        """Initialize the DictMultimodalSensor instance.

        Args:
            sensors (dict[Modality, BaseSensor[Tensor]]): A dictionary mapping modality
                types to their corresponding sensor instances.
        """
        super().__init__()

        for key, value in sensors.items():
            sensors[Modality(key)] = value  # Ensure modal key is of type `Modality`

        self.sensors = sensors

    @override
    def read(self) -> dict[Modality, Tensor]:
        """Read data from all sensors.

        Returns:
            dict[Modality, Tensor]: A dictionary containing sensor readings for each modality.
        """
        return {key: value.read() for key, value in self.sensors.items()}

    @override
    def setup(self) -> None:
        """Set up all sensors."""
        super().setup()
        for sensor in self.sensors.values():
            sensor.setup()

    @override
    def teardown(self) -> None:
        """Clean up and release resources for all sensors."""
        super().teardown()
        for sensor in self.sensors.values():
            sensor.teardown()
