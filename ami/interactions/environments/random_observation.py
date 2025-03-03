from typing import TypedDict

import numpy as np
import torch
import torch.types
from numpy.typing import NDArray
from typing_extensions import override

from ami.tensorboard_loggers import LoggableTypes, TensorBoardLogger

from .base_environment import BaseEnvironment


class RandomObservationGenerator:
    """A generator for random observations with controllable information rate.

    This class generates random observations from a discrete uniform
    distribution with configurable parameters that allow precise control
    over the information rate (bits per second). The information rate
    can be calculated using the formula: I = (observation_length *
    length_ratio) * (sample_probability / time_interval) *
    (max_level_order * level_ratio)

    The generator produces observations as vectors of floating point
    values in the range [-1, 1].
    """

    class ParameterDict(TypedDict):
        max_level_order: int
        observation_length: int
        time_interval: float
        level_ratio: float
        length_ratio: float
        sample_probability: float
        num_levels: int
        raw_length: int
        average_time_interval: float
        entropy: float
        information_rate: float

    def __init__(
        self,
        max_level_order: int,
        observation_length: int,
        time_interval: float,
        level_ratio: float = 0.5,
        length_ratio: float = 0.5,
        sample_probability: float = 0.5,
    ):
        """Initialize the random observation generator.

        A method for generating random observations with calculable and controllable
        information rate per unit time.

        Args:
            max_level_order: Maximum order of quantization levels
            observation_length: Length of observation vector
            time_interval: Time interval between timesteps (seconds)
            level_ratio: Ratio of level order [0,1]
            length_ratio: Ratio of observation length [0,1]
            sample_probability: Probability of sampling a new observation [0,1]
        """
        # Validate basic parameters
        if not isinstance(max_level_order, int) or max_level_order <= 0:
            raise ValueError("max_level_order must be a positive integer")
        if not isinstance(observation_length, int) or observation_length <= 0:
            raise ValueError("observation_length must be a positive integer")
        if not isinstance(time_interval, (int, float)) or time_interval <= 0:
            raise ValueError("time_interval must be a positive number")

        self.max_level_order = max_level_order
        self.observation_length = observation_length
        self.time_interval = time_interval

        # Control parameters
        self.set_control_params(level_ratio, length_ratio, sample_probability)

        # Initialize previous observation
        self.prev_observation: NDArray[np.float_] = np.zeros(self.observation_length)

        # Time counter
        self.time_step = 0

    def set_control_params(self, level_ratio: float, length_ratio: float, sample_probability: float) -> None:
        """Set the control parameters.

        Args:
            level_ratio: Ratio of level order [0,1]
            length_ratio: Ratio of observation length [0,1]
            sample_probability: Probability of sampling a new observation [0,1]
        """
        # Validate parameter ranges
        if not (0 <= level_ratio <= 1):
            raise ValueError("level_ratio must be in range [0,1]")
        if not (0 <= length_ratio <= 1):
            raise ValueError("length_ratio must be in range [0,1]")
        if not (0 <= sample_probability <= 1):
            raise ValueError("sample_probability must be in range [0,1]")

        self.level_ratio = level_ratio
        self.length_ratio = length_ratio
        self.sample_probability = sample_probability

        # Calculate number of levels
        self.num_levels = max(round(2 ** (self.max_level_order * self.level_ratio)), 1)

        # Calculate actual sampling length
        self.raw_length = max(round(self.observation_length * self.length_ratio), 1)

    @property
    def average_time_interval(self) -> float:
        """Calculate average interval of sample."""
        inv_interval = self.sample_probability / self.time_interval
        if inv_interval > 0:
            return inv_interval
        return 1 / inv_interval

    def sample_observation(self) -> NDArray[np.float_]:
        """Sample a new observation.

        Returns:
            Sampled observation vector
        """
        # Advance time
        self.time_step += 1

        # Sample new observation with probability sample_probability
        if np.random.random() < self.sample_probability:
            # Generate observation of length raw_length
            raw_observation = self._sample_from_distribution(self.raw_length)

            # Interpolate to length observation_length
            observation = self._interpolate(raw_observation, self.observation_length)

            # Update previous observation
            self.prev_observation = observation.copy()
        else:
            # Use previous observation
            observation = self.prev_observation.copy()

        return observation

    def _sample_from_distribution(self, length: int) -> NDArray[np.float_]:
        """Sample an observation of specified length from probability
        distribution.

        Args:
            length: Length of observation to sample

        Returns:
            Sampled observation vector
        """
        # Sample from discrete uniform distribution
        if self.num_levels <= 1:
            # If number of levels is 1 or less, return zeros
            raw_values = np.zeros(length)
        else:
            # Sample from discrete uniform distribution
            discrete_values = np.random.randint(0, self.num_levels, size=length)

            # Normalize to range [-1, 1]
            raw_values = 2 * (discrete_values / (self.num_levels - 1)) - 1

        return raw_values

    def _interpolate(self, values: NDArray[np.float_], target_length: int) -> NDArray[np.float_]:
        """Linearly interpolate observation values.

        Args:
            values: Original observation values
            target_length: Target length

        Returns:
            Interpolated observation values
        """
        # No interpolation needed if lengths match
        if len(values) == target_length:
            return values

        # Linear interpolation
        indices = np.linspace(0, len(values) - 1, target_length)
        return np.interp(indices, np.arange(len(values)), values)

    def calculate_information_rate(self) -> float:
        """Calculate the information rate per unit time.

        Returns:
            Information rate (bit/s)
        """
        # Calculate entropy: H(p) = max_level_order * level_ratio
        entropy = self.max_level_order * self.level_ratio

        # Calculate information rate: I = (observation_length * length_ratio) * (sample_probability / time_interval) * (max_level_order * level_ratio)
        information_rate = (
            (self.observation_length * self.length_ratio) * (self.sample_probability / self.time_interval) * entropy
        )

        return information_rate

    def get_params(self) -> ParameterDict:
        """Get current parameter settings.

        Returns:
            Dictionary of current parameters
        """
        return self.ParameterDict(
            max_level_order=self.max_level_order,
            observation_length=self.observation_length,
            time_interval=self.time_interval,
            level_ratio=self.level_ratio,
            length_ratio=self.length_ratio,
            sample_probability=self.sample_probability,
            num_levels=self.num_levels,
            raw_length=self.raw_length,
            average_time_interval=self.average_time_interval,
            entropy=self.max_level_order * self.level_ratio,
            information_rate=self.calculate_information_rate(),
        )

    def get_max_information_rate(self) -> float:
        """Get maximum possible information rate (all control parameters set to
        1).

        Returns:
            Maximum information rate (bit/s)
        """
        return (self.observation_length * 1.0) * (1.0 / self.time_interval) * (self.max_level_order * 1.0)

    def get_min_information_rate(self) -> float:
        """Get minimum possible information rate (any control parameter set to
        0).

        Returns:
            Minimum information rate (bit/s)
        """
        return 0.0

    def reset(self) -> None:
        """Reset the generator state."""
        self.prev_observation = np.zeros(self.observation_length)
        self.time_step = 0


class RandomObservationEnvironment(BaseEnvironment[torch.Tensor, torch.Tensor]):
    """An environment that generates random observations with controllable
    information rate.

    This environment wraps the RandomObservationGenerator to provide an interface compatible
    with reinforcement learning or other AI systems expecting an environment with observe/affect
    methods. The environment generates random observations from a discrete uniform distribution
    with configurable parameters that allow precise control over the information rate.

    The environment accepts actions that control three parameters of the observation generator:
    level_ratio, length_ratio, and sample_probability. These parameters directly affect the
    information rate of the generated observations.

    Attributes:
        _observation_generator: The underlying generator for random observations
        _dtype: The torch data type to use for observations
        _logger: Optional TensorboardLogger for logging metrics
    """

    def __init__(
        self,
        max_level_order: int = 2**8,
        observation_length: int = 1024,
        time_interval: float = 0.1,
        level_ratio: float = 0.5,
        length_ratio: float = 0.5,
        sample_probability: float = 0.5,
        dtype: torch.dtype = torch.float,
        logger: TensorBoardLogger | None = None,
    ) -> None:
        """Initialize the random observation environment.

        Args:
            max_level_order: Maximum order of quantization levels (2^max_level_order possible values)
            observation_length: Length of observation vector
            time_interval: Time interval between timesteps (seconds)
            level_ratio: Initial ratio of level order [0,1]
            length_ratio: Initial ratio of observation length [0,1]
            sample_probability: Initial probability of sampling a new observation [0,1]
            dtype: Torch data type for observations
            logger: Optional TensorboardLogger for logging metrics
        """
        super().__init__()

        self._observation_generator = RandomObservationGenerator(
            max_level_order, observation_length, time_interval, level_ratio, length_ratio, sample_probability
        )
        self._dtype = dtype
        self._logger = logger

    @override
    def observe(self) -> torch.Tensor:
        """Generate and return a new observation.

        Returns:
            A torch.Tensor observation with shape [observation_length]
        """
        return torch.tensor(self._observation_generator.sample_observation()).to(self._dtype)

    @override
    def affect(self, action: torch.Tensor) -> None:
        """Apply an action to modify the environment parameters.

        The action should be a 1D tensor with 3 elements representing:
        [level_ratio, length_ratio, sample_probability]

        Args:
            action: A torch.Tensor with shape [3] containing control parameters
                   in the range [0,1] for each parameter

        Raises:
            ValueError: If action has incorrect dimensions or type
        """
        if action.ndim != 1:
            raise ValueError("Action must be a 1D tensor")
        if action.numel() != 3:
            raise ValueError("Action must have exactly 3 elements")
        if not torch.is_floating_point(action):
            raise ValueError("Action must be a floating-point tensor")

        self._observation_generator.set_control_params(*action.cpu().detach().tolist())

        # Logging
        if self._logger is not None:
            prefix = "random-observation/"
            params = self._observation_generator.get_params()
            for name, value in params.items():
                if isinstance(value, LoggableTypes):
                    self._logger.log(prefix + name, value)
            self._logger.log(prefix + "average_sample_fps", 1 / params["average_time_interval"])
