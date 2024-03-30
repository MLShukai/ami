import pytest
import torch

from ami.interactions.environments.actuators.vrchat_osc_discrete_actuator import (
    VRChatOSCDiscreteActuator,
)


class TestVRChatOSCDiscreteActuator:
    @pytest.fixture
    def actuator(self) -> VRChatOSCDiscreteActuator:
        return VRChatOSCDiscreteActuator()

    def test_operate(self, actuator: VRChatOSCDiscreteActuator) -> None:
        actuator.setup()
        actuator.operate(torch.tensor([1, 1, 1, 1, 1], dtype=torch.long))
        actuator.teardown()
