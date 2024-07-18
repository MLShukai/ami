from typing import Any

import torch
from mlagents_envs.environment import UnityEnvironment as RawUnityEnv
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import override

from .base_environment import BaseEnvironment


class UnityEnvironment(BaseEnvironment[Tensor, Tensor]):
    """A class for connecting to and interacting with a Unity ML Agents
    environment.

    This class provides an interface to interact with Unity environments
    using the ML-Agents toolkit. It handles the setup, teardown, and
    communication with the Unity environment, allowing for observations
    to be received and actions to be sent.
    """

    def __init__(
        self,
        file_path: str,
        worker_id: int = 0,
        base_port: int | None = None,
        seed: int = 0,
        time_scale: float = 1.0,
        target_frame_rate: int = 60,
    ) -> None:
        """Initialize the UnityEnvironment.

        Args:
            file_path (str): Path to the Unity environment executable.
            worker_id (int, optional): Identifier for the worker. Defaults to 0.
            base_port (int | None, optional): Port to use for communication with the environment.
                If None, the default port will be used. Defaults to None.
            seed (int, optional): Random seed for the environment. Defaults to 0.
            time_scale (float, optional): Time scale for the Unity environment. Defaults to 1.0.
            target_frame_rate (int, optional): Target frame rate for the Unity environment. Defaults to 60.

        The UnityEnvironment is set up with the specified parameters, establishing a connection
        to the Unity executable and configuring the environment settings such as time scale and
        frame rate.
        """
        super().__init__()

        self.engine_configuration_channel = EngineConfigurationChannel()
        self._env = UnityToGymWrapper(
            RawUnityEnv(
                file_path,
                worker_id,
                base_port,
                seed=seed,
                side_channels=[self.engine_configuration_channel],
            ),
            allow_multiple_obs=False,
        )
        self.engine_configuration_channel.set_configuration_parameters(
            time_scale=time_scale, target_frame_rate=target_frame_rate
        )

    @property
    def gym_unity_environment(self) -> UnityToGymWrapper:
        return self._env

    @property
    def raw_unity_environment(self) -> RawUnityEnv:
        return self._env._env

    observation: NDArray[Any]

    @override
    def setup(self) -> None:
        super().setup()
        self.observation = self._env.reset()

    @override
    def teardown(self) -> None:
        super().teardown()
        self._env.close()

    @override
    def affect(self, action: Tensor) -> None:
        self.observation, _, _, _ = self._env.step(action.detach().cpu().numpy())

    @override
    def observe(self) -> Tensor:
        return torch.from_numpy(self.observation)
