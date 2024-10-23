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

    def test_set_target_value(self, system: FirstOrderDelaySystemStepResponce):
        """Test setting target values and verifying state changes."""
        # Initial state
        assert system._target_value == 0.0
        assert system._current_elapsed_time == 0.0

        # Set new target
        system.set_target_value(1.0)
        assert system._target_value == 1.0
        assert system._current_elapsed_time == 0.0
        assert system._start_value == system._current_value

        # Set same target (shouldn't reset elapsed time)
        system._current_elapsed_time = 0.5
        system.set_target_value(1.0)
        assert system._current_elapsed_time == 0.5

        # Set different target (should reset elapsed time)
        system.set_target_value(2.0)
        assert system._current_elapsed_time == 0.0

    def test_step_response(self, system: FirstOrderDelaySystemStepResponce):
        """Test the step response behavior of the system."""
        # Set target to 1.0
        system.set_target_value(1.0)

        # Step through simulation and verify response
        values = []
        for _ in range(10):  # Simulate for 1 second (10 * 0.1s)
            value = system.step()
            values.append(value)

        # Verify that values are monotonically increasing
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))

        # Verify final value is approaching target
        assert values[-1] == pytest.approx(1.0, rel=0.3)  # Should be close to 1.0 but not quite there

    def test_multiple_target_changes(self, system: FirstOrderDelaySystemStepResponce):
        """Test system response to multiple target value changes."""
        # First target
        system.set_target_value(1.0)
        values1 = [system.step() for _ in range(5)]

        # Change target midway
        system.set_target_value(-1.0)
        values2 = [system.step() for _ in range(5)]

        # Verify direction changes
        assert all(values1[i] <= values1[i + 1] for i in range(len(values1) - 1))  # Increasing
        assert all(values2[i] >= values2[i + 1] for i in range(len(values2) - 1))  # Decreasing

    def test_zero_response(self, system: FirstOrderDelaySystemStepResponce):
        """Test system response when target equals initial value."""
        system.set_target_value(0.0)  # Same as initial value
        for _ in range(10):
            value = system.step()
            assert value == pytest.approx(0.0, abs=1e-10)
