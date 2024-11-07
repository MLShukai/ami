from pathlib import Path
from typing import Generic

from ..._types import ObsType, WrapperObsType
from ...io_wrappers.base_io_wrapper import BaseIOWrapper
from .base_sensor import BaseSensor, BaseSensorWrapper


class SensorWrapperFromIOWrapper(BaseSensorWrapper[WrapperObsType, ObsType], Generic[WrapperObsType, ObsType]):
    """Creates the SensorWrapper from IOWrapper object."""

    def __init__(self, sensor: BaseSensor[ObsType], io_wrapper: BaseIOWrapper[ObsType, WrapperObsType]) -> None:
        super().__init__(sensor)
        self._io_wrapper = io_wrapper

    def wrap_observation(self, observation: ObsType) -> WrapperObsType:
        return self._io_wrapper.wrap(observation)

    def setup(self) -> None:
        super().setup()
        self._io_wrapper.setup()

    def teardown(self) -> None:
        super().teardown()
        self._io_wrapper.teardown()

    def on_paused(self) -> None:
        super().on_paused()
        self._io_wrapper.on_paused()

    def on_resumed(self) -> None:
        super().on_resumed()
        self._io_wrapper.on_resumed()

    def save_state(self, path: Path) -> None:
        super().save_state(path)
        self._io_wrapper.save_state(path)

    def load_state(self, path: Path) -> None:
        super().load_state(path)
        self._io_wrapper.load_state(path)
