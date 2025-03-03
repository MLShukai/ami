import numpy as np
import pytest

from ami.interactions.environments.random_observation import RandomObservationGenerator


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
