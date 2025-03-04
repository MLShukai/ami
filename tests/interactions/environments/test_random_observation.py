from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from ami.interactions.environments.random_observation import (
    RandomObservationEnvironment,
    RandomObservationEnvironmentDiscreteAction,
    RandomObservationGenerator,
)
from ami.tensorboard_loggers import TensorBoardLogger


class TestRandomObservationGenerator:
    @pytest.fixture
    def default_generator(self):
        """Create a default RandomObservationGenerator for testing."""
        return RandomObservationGenerator(max_level_order=8, observation_length=10, time_interval=0.1)

    def test_initialization(self, default_generator):
        """Test initialization with different parameters."""
        # Test default initialization
        params = default_generator.get_params()

        assert params["max_level_order"] == 8
        assert params["observation_length"] == 10
        assert params["time_interval"] == 0.1
        assert params["level_ratio"] == 0.5
        assert params["length_ratio"] == 0.5
        assert params["sample_probability"] == 0.5

        # Test custom initialization
        custom_generator = RandomObservationGenerator(
            max_level_order=4,
            observation_length=20,
            time_interval=0.2,
            level_ratio=0.7,
            length_ratio=0.8,
            sample_probability=0.9,
        )

        custom_params = custom_generator.get_params()
        assert custom_params["max_level_order"] == 4
        assert custom_params["observation_length"] == 20
        assert custom_params["time_interval"] == 0.2
        assert custom_params["level_ratio"] == 0.7
        assert custom_params["length_ratio"] == 0.8
        assert custom_params["sample_probability"] == 0.9

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid max_level_order
        with pytest.raises(ValueError):
            RandomObservationGenerator(max_level_order=0, observation_length=10, time_interval=0.1)

        # Test invalid observation_length
        with pytest.raises(ValueError):
            RandomObservationGenerator(max_level_order=8, observation_length=-5, time_interval=0.1)

        # Test invalid time_interval
        with pytest.raises(ValueError):
            RandomObservationGenerator(max_level_order=8, observation_length=10, time_interval=-0.1)

    def test_set_control_params(self, default_generator):
        """Test setting control parameters."""
        # Test setting valid parameters
        default_generator.set_control_params(level_ratio=0.8, length_ratio=0.9, sample_probability=0.7)

        params = default_generator.get_params()
        assert params["level_ratio"] == 0.8
        assert params["length_ratio"] == 0.9
        assert params["sample_probability"] == 0.7

        # Test boundary values
        default_generator.set_control_params(level_ratio=0.0, length_ratio=0.0, sample_probability=0.0)

        # Test setting invalid level_ratio
        with pytest.raises(ValueError):
            default_generator.set_control_params(level_ratio=1.5, length_ratio=0.5, sample_probability=0.5)

    def test_sample_observation(self, default_generator):
        """Test sampling observations."""
        # Test observation shape and type
        observation = default_generator.sample_observation()
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (10,)
        assert np.issubdtype(observation.dtype, np.floating)

        # Test observation values range
        assert np.all(observation >= -1)
        assert np.all(observation <= 1)

        # Test time step increment
        assert default_generator.time_step == 1

    def test_information_rate_calculation(self, default_generator):
        """Test information rate calculation."""
        # Calculate expected information rate
        level_ratio = 0.5
        length_ratio = 0.5
        sample_probability = 0.5
        expected_rate = (10 * length_ratio) * (sample_probability / 0.1) * (8 * level_ratio)

        # Test calculated rate
        actual_rate = default_generator.calculate_information_rate()
        assert isinstance(actual_rate, float)
        assert actual_rate == pytest.approx(expected_rate, abs=1e-6)

        # Test with minimal parameters (should result in 0 information rate)
        default_generator.set_control_params(level_ratio=0.0, length_ratio=0.5, sample_probability=0.5)
        actual_rate = default_generator.calculate_information_rate()
        assert actual_rate == 0.0

    def test_get_min_max_information_rate(self, default_generator):
        """Test getting minimum and maximum information rates."""
        # Calculate expected maximum rate
        expected_max_rate = (10 * 1.0) * (1.0 / 0.1) * (8 * 1.0)
        actual_max_rate = default_generator.get_max_information_rate()
        assert actual_max_rate == expected_max_rate

        # Test minimum rate
        assert default_generator.get_min_information_rate() == 0.0

    def test_get_params(self, default_generator):
        """Test getting current parameters."""
        params = default_generator.get_params()

        # Check if all expected keys are present
        expected_keys = [
            "max_level_order",
            "observation_length",
            "time_interval",
            "level_ratio",
            "length_ratio",
            "sample_probability",
            "num_levels",
            "raw_length",
            "average_time_interval",
            "entropy",
            "information_rate",
        ]

        for key in expected_keys:
            assert key in params

    def test_reset(self, default_generator):
        """Test resetting the generator."""
        # Sample some observations to change internal state
        for _ in range(5):
            default_generator.sample_observation()

        # Check time step
        assert default_generator.time_step == 5

        # Reset generator
        default_generator.reset()

        # Check if time step is reset
        assert default_generator.time_step == 0

        # Check if previous observation is reset
        assert np.all(default_generator.prev_observation == 0)

    def test_sample_probability_effect(self):
        """Test the effect of sample_probability on observations."""
        # Create generator with 0 sample probability
        generator = RandomObservationGenerator(
            max_level_order=8, observation_length=10, time_interval=0.1, sample_probability=0.0
        )

        # First observation with reset state (all zeros)
        first_observation = generator.sample_observation()
        assert np.all(first_observation == 0)

        # Next observations should be the same as first
        for _ in range(5):
            next_observation = generator.sample_observation()
            np.testing.assert_array_equal(next_observation, first_observation)

    def test_interpolation(self):
        """Test interpolation of observations."""
        # Test with different length ratios
        for length_ratio in [0.1, 0.5, 0.9]:
            generator = RandomObservationGenerator(
                max_level_order=8, observation_length=10, time_interval=0.1, length_ratio=length_ratio
            )

            # Check that raw_length is calculated correctly
            expected_raw_length = max(round(10 * length_ratio), 1)
            assert generator.raw_length == expected_raw_length

            # Sample an observation
            observation = generator.sample_observation()

            # Check observation length
            assert len(observation) == 10

    def test_level_quantization(self):
        """Test quantization levels based on level_ratio."""
        # Test with level_ratio = 0 (should result in num_levels = 1)
        generator = RandomObservationGenerator(
            max_level_order=8, observation_length=10, time_interval=0.1, level_ratio=0.0
        )

        assert generator.num_levels == 1

        # Test with level_ratio = 1 (should result in num_levels = 2^max_level_order)
        generator = RandomObservationGenerator(
            max_level_order=8, observation_length=10, time_interval=0.1, level_ratio=1.0
        )

        assert generator.num_levels == 2**8

    def test_internal_helper_methods(self, default_generator):
        """Test internal helper methods."""
        # Test _sample_from_distribution
        values = default_generator._sample_from_distribution(5)
        assert len(values) == 5
        assert np.all(values >= -1)
        assert np.all(values <= 1)

        # Test _interpolate
        original = np.array([1.0, 3.0])
        result = default_generator._interpolate(original, 3)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sampling_with_mocked_random(self, mocker):
        """Test sampling with mocked random functions."""
        # Mock numpy.random.random to always return 0.3 (below default sample_probability of 0.5)
        mocker.patch("numpy.random.random", return_value=0.3)

        # Mock numpy.random.randint to return predictable values
        mock_randint = mocker.patch("numpy.random.randint")
        mock_randint.return_value = np.array([2, 2, 2, 2, 2])

        generator = RandomObservationGenerator(max_level_order=8, observation_length=5, time_interval=0.1)

        # With mocked random, should always get new samples
        observation = generator.sample_observation()

        # Verify random.randint was called with correct parameters
        num_levels = generator.num_levels
        mock_randint.assert_called_once_with(0, num_levels, size=generator.raw_length)


class TestRandomObservationEnvironment:
    @pytest.fixture
    def mock_generator(self, mocker: MockerFixture):
        """Create a mocked RandomObservationGenerator."""
        generator = mocker.patch(
            "ami.interactions.environments.random_observation.RandomObservationGenerator", autospec=True
        )
        generator_instance = generator.return_value
        generator_instance.sample_observation.return_value = np.zeros(10)
        generator_instance.get_params.return_value = {
            "max_level_order": 8,
            "observation_length": 10,
            "time_interval": 0.1,
            "level_ratio": 0.5,
            "length_ratio": 0.5,
            "sample_probability": 0.5,
            "num_levels": 16,
            "raw_length": 5,
            "average_time_interval": 0.2,
            "entropy": 4.0,
            "information_rate": 100.0,
        }
        return generator_instance

    @pytest.fixture
    def mock_logger(self):
        """Create a mocked TensorBoardLogger."""
        return MagicMock(spec=TensorBoardLogger)

    def test_initialization(self):
        """Test initialization with default parameters."""
        env = RandomObservationEnvironment()

        assert env._dtype == torch.float
        assert env._logger is None

        # Test with custom parameters
        custom_env = RandomObservationEnvironment(
            max_level_order=4,
            observation_length=20,
            time_interval=0.2,
            level_ratio=0.7,
            length_ratio=0.8,
            sample_probability=0.9,
            dtype=torch.float32,
        )

        assert custom_env._dtype == torch.float32

    def test_observe(self, mocker: MockerFixture, mock_generator):
        """Test the observe method."""
        env = RandomObservationEnvironment(observation_length=10)

        # Get an observation
        observation = env.observe()

        # Check that the generator's sample_observation method was called
        mock_generator.sample_observation.assert_called_once()

        # Check that the returned observation is a tensor of the right type and shape
        assert isinstance(observation, torch.Tensor)
        assert observation.dtype == torch.float
        assert observation.shape == (10,)

    def test_affect_valid_action(self, mocker: MockerFixture, mock_generator):
        """Test the affect method with valid actions."""

        env = RandomObservationEnvironment()

        # Create a valid action tensor
        action = torch.tensor([0.3, 0.6, 0.7], dtype=torch.float32)

        # Apply the action
        env.affect(action)

    def test_affect_invalid_actions(self, mocker: MockerFixture, mock_generator):
        """Test the affect method with invalid actions."""

        env = RandomObservationEnvironment()

        # Test with wrong dimensionality
        with pytest.raises(ValueError, match="Action must be a 1D tensor"):
            env.affect(torch.tensor([[0.3, 0.6, 0.7]], dtype=torch.float32))

        # Test with wrong number of elements
        with pytest.raises(ValueError, match="Action must have exactly 3 elements"):
            env.affect(torch.tensor([0.3, 0.6, 0.7, 0.8], dtype=torch.float32))

        # Test with non-floating-point tensor
        with pytest.raises(ValueError, match="Action must be a floating-point tensor"):
            env.affect(torch.tensor([3, 6, 7], dtype=torch.int32))

    def test_logging(self, mocker: MockerFixture, mock_generator, mock_logger):
        """Test logging functionality."""

        # Create environment with logger
        env = RandomObservationEnvironment(logger=mock_logger)

        # Apply an action to trigger logging
        action = torch.tensor([0.3, 0.6, 0.7], dtype=torch.float32)
        env.affect(action)

        # Check that log was called for each parameter
        params = mock_generator.get_params.return_value
        for name, value in params.items():
            mock_logger.log.assert_any_call(f"random-observation/{name}", value)

        # Check that average_sample_fps was logged
        mock_logger.log.assert_any_call("random-observation/average_sample_fps", 1 / params["average_time_interval"])

    def test_full_integration(self):
        """Test full integration with actual instances (not mocks)."""
        # Create environment with default parameters
        env = RandomObservationEnvironment(max_level_order=4, observation_length=10, time_interval=0.1)

        # Get an observation
        observation = env.observe()
        assert isinstance(observation, torch.Tensor)
        assert observation.shape == (10,)
        assert observation.dtype == torch.float

        # Apply an action
        action = torch.tensor([0.8, 0.7, 0.6], dtype=torch.float32)
        env.affect(action)

        # Get another observation after changing parameters
        observation2 = env.observe()
        assert isinstance(observation2, torch.Tensor)
        assert observation2.shape == (10,)

        # Check that the parameters were updated correctly
        params = env._observation_generator.get_params()
        assert params["level_ratio"] == pytest.approx(0.8)
        assert params["length_ratio"] == pytest.approx(0.7)
        assert params["sample_probability"] == pytest.approx(0.6)

    def test_save_and_load_state(self, mock_logger, tmp_path):
        mock_logger.state_dict.return_value = 0
        env = RandomObservationEnvironment(logger=mock_logger)
        test_path = tmp_path / "env"
        env.save_state(test_path)
        assert (test_path / "random_observation.pkl").is_file()
        env.load_state(test_path)
        mock_logger.load_state_dict.assert_called_once_with(0)


class TestRandomObservationEnvironmentDiscreteAction:
    def test_initialization_valid_params(self):
        """Test initialization with valid parameters."""
        # Test with default parameters plus action_quantization_levels
        env = RandomObservationEnvironmentDiscreteAction(action_quantization_levels=[3, 4, 5])

        assert env._dtype == torch.float
        assert env._logger is None
        assert len(env.action_levels) == 3
        assert len(env.action_levels[0]) == 3
        assert len(env.action_levels[1]) == 4
        assert len(env.action_levels[2]) == 5

        # Test action_levels are correctly created
        assert env.action_levels[0][0] == 0.0
        assert env.action_levels[0][-1] == 1.0
        assert env.action_levels[1][0] == 0.0
        assert env.action_levels[1][-1] == 1.0
        assert env.action_levels[2][0] == 0.0
        assert env.action_levels[2][-1] == 1.0

        # Test with custom parameters
        custom_env = RandomObservationEnvironmentDiscreteAction(
            max_level_order=4,
            observation_length=20,
            time_interval=0.2,
            level_ratio=0.7,
            length_ratio=0.8,
            sample_probability=0.9,
            dtype=torch.float32,
            action_quantization_levels=[2, 2, 2],
        )

        assert custom_env._dtype == torch.float32
        assert len(custom_env.action_levels) == 3
        assert all(len(level) == 2 for level in custom_env.action_levels)

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        # Test with wrong number of quantization levels
        with pytest.raises(ValueError):
            RandomObservationEnvironmentDiscreteAction(action_quantization_levels=[3, 4])  # Should be 3 elements

        # Test with non-integer quantization levels
        with pytest.raises(ValueError):
            RandomObservationEnvironmentDiscreteAction(action_quantization_levels=[3, 4.5, 5])  # Should be all integers

        # Test with non-positive quantization levels
        with pytest.raises(ValueError):
            RandomObservationEnvironmentDiscreteAction(action_quantization_levels=[3, 0, 5])  # Should be all positive

        with pytest.raises(ValueError):
            RandomObservationEnvironmentDiscreteAction(action_quantization_levels=[3, -1, 5])  # Should be all positive

    def test_affect_valid_action(self, mocker: MockerFixture):
        """Test the affect method with valid discrete actions."""
        # Mock the parent class affect method
        parent_affect = mocker.patch(
            "ami.interactions.environments.random_observation.RandomObservationEnvironment.affect"
        )

        env = RandomObservationEnvironmentDiscreteAction(action_quantization_levels=[3, 3, 3])

        # Create a valid discrete action tensor
        action = torch.tensor([0, 1, 2], dtype=torch.long)

        # Apply the action
        env.affect(action)

        # Check that parent affect was called with the correct continuous action
        expected_continuous_action = torch.tensor([0.0, 0.5, 1.0])
        parent_affect.assert_called_once()
        continuous_action = parent_affect.call_args[0][0]
        assert torch.allclose(continuous_action, expected_continuous_action)

    def test_affect_invalid_actions(self):
        """Test the affect method with invalid actions."""
        env = RandomObservationEnvironmentDiscreteAction(action_quantization_levels=[3, 3, 3])

        # Test with wrong dimensionality
        with pytest.raises(ValueError, match="Action must be a 1D tensor"):
            env.affect(torch.tensor([[0, 1, 2]], dtype=torch.long))

        # Test with wrong number of elements
        with pytest.raises(ValueError, match="Action must have exactly 3 elements"):
            env.affect(torch.tensor([0, 1, 2, 0], dtype=torch.long))
