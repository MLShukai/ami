import math

import pytest

from ami.interactions.environments.actuators.first_order_delay_system import (
    FirstOrderDelaySystemStepResponce,
)


class TestFirstOrderDelaySystemStepResponce:
    @pytest.fixture
    def system(self):
        """Create a default system for testing with standard parameters."""
        return FirstOrderDelaySystemStepResponce(delta_time=0.1, time_constant=0.5, initial_value=0.0)

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(AssertionError):
            FirstOrderDelaySystemStepResponce(delta_time=-0.1, time_constant=0.5)

        with pytest.raises(AssertionError):
            FirstOrderDelaySystemStepResponce(delta_time=0.1, time_constant=-0.5)

    def test_initial_state(self, system: FirstOrderDelaySystemStepResponce):
        """Test the initial state of the system."""
        assert system._target_value == 0.0
        assert system._current_elapsed_time == 0.0
        assert system._start_value == 0.0
        assert system.get_current_value() == 0.0

    def test_set_target_value(self, system: FirstOrderDelaySystemStepResponce):
        """Test setting target values and verifying state changes."""
        # Step a few times to get a non-zero current value
        system.set_target_value(1.0)
        for _ in range(3):
            system.step()
        current_value = system.get_current_value()

        # Set new target
        system.set_target_value(2.0)
        assert system._target_value == 2.0
        assert system._current_elapsed_time == 0.0
        assert system._start_value == current_value

        # Set same target (shouldn't reset elapsed time)
        system._current_elapsed_time = 0.5
        system.set_target_value(2.0)
        assert system._current_elapsed_time == 0.5

    @pytest.mark.parametrize(
        "time_point,expected_percentage",
        [
            # t = τ, 2τ, 3τ
            (0.5, 0.632),
            (1.0, 0.865),
            (1.5, 0.950),
        ],
    )
    def test_analytical_response(self, system: FirstOrderDelaySystemStepResponce, time_point, expected_percentage):
        """Test that the system follows the analytical solution."""
        target_value = 1.0
        system.set_target_value(target_value)

        # Step until we reach the desired time
        steps_needed = int(time_point / system.delta_time)
        for _ in range(steps_needed):
            value = system.step()
        print(value)

        assert value == pytest.approx(target_value * expected_percentage, rel=1e-3)

    def test_response_direction(self, system: FirstOrderDelaySystemStepResponce):
        """Test system response in both positive and negative directions."""
        # Test increasing response
        system.set_target_value(1.0)
        values_up = [system.step() for _ in range(5)]
        assert all(values_up[i] < values_up[i + 1] for i in range(len(values_up) - 1))

        # Test decreasing response
        system.set_target_value(-1.0)
        values_down = [system.step() for _ in range(5)]
        assert all(values_down[i] > values_down[i + 1] for i in range(len(values_down) - 1))

    def test_zero_response(self, system: FirstOrderDelaySystemStepResponce):
        """Test system response when target equals initial value."""
        system.set_target_value(0.0)  # Same as initial value
        for _ in range(10):
            value = system.step()
            assert value == 0.0  # Should be exactly zero, no approximation needed

    def test_step_timing(self, system: FirstOrderDelaySystemStepResponce):
        """Test that elapsed time is correctly updated."""
        system.set_target_value(1.0)

        # Step multiple times and check elapsed time
        steps = 5
        for i in range(steps):
            system.step()
            assert system._current_elapsed_time == pytest.approx((i + 1) * system.delta_time)
