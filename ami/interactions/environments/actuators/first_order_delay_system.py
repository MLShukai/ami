import math


class FirstOrderDelaySystemStepResponce:
    """A first-order delay system step response simulator.

    This class implements a first-order delay system that gradually approaches
    a target value according to the following differential equation:
        τ * (dy/dt) + y = Ku
    where:
        τ: time constant
        y: output value
        K: system gain (assumed to be 1.0)
        u: input value (target value)

    The system response is computed using the analytical solution
    for the step response of a first-order system:
        y(t) = y_0 + (u - y_0)(1 - e^(-t/τ))
    where:
        y_0: initial value at the start of the current transition
        u: target value
        t: elapsed time since the last target value change
        τ: time constant

    The time constant τ determines how quickly the system approaches
    the target value. The output will reach approximately:
        - 63.2% of the target value after t = τ
        - 86.5% of the target value after t = 2τ
        - 95.0% of the target value after t = 3τ

    Usage:
        # Create a system with 0.1s time step and 0.5s time constant
        system = FirstOrderDelaySystemStepResponce(
            delta_time=0.1,
            time_constant=0.5,
            initial_value=0.0
        )

        # Set target to 1.0 and simulate for 2 seconds
        system.set_target_value(1.0)
        for _ in range(20):  # 20 steps * 0.1s = 2s
            current_value = system.step()
            print(f"Current value: {current_value}")
    """

    def __init__(self, delta_time: float, time_constant: float, initial_value: float = 0.0) -> None:
        """Initialize the first-order delay system.

        Args:
            delta_time: Time step for state updates in seconds.
                Must be positive. Determines how often the system state is updated.
            time_constant: Time constant (τ) of the system in seconds.
                Must be positive. Larger values result in slower response.
            initial_value: Initial output value of the system.
                Defaults to 0.0.

        Raises:
            AssertionError: If delta_time or time_constant is not positive.
        """
        assert delta_time > 0
        assert time_constant > 0

        self.delta_time = delta_time
        self.time_constant = time_constant

        self._start_value = initial_value
        self._target_value = initial_value
        self._current_elapsed_time = 0.0

    def set_target_value(self, value: float) -> None:
        """Set a new target value for the system.

        When the target value changes, the system response will start
        from the current value and exponentially approach the new target.
        The elapsed time counter is reset to maintain the correct
        exponential response from the current state.

        Args:
            value: New target value for the system to approach.
        """
        if self._target_value != value:
            self._start_value = self.get_current_value()
            self._current_elapsed_time = 0.0

        self._target_value = value

    def get_current_value(self) -> float:
        """Get the current output value of the system.

        Calculates the current value using the analytical solution
        of the first-order differential equation:
            y(t) = y_0 + (u - y_0)(1 - e^(-t/τ))

        Returns:
            float: Current output value of the system.
        """
        return self._start_value + (self._target_value - self._start_value) * (
            1 - math.exp(-self._current_elapsed_time / self.time_constant)
        )

    def step(self) -> float:
        """Step the simulation forward by one time step.

        Updates the elapsed time and calculates the new output value
        using the analytical solution. This provides exact results
        without numerical integration errors.

        Returns:
            float: Current output value of the system after the time step.
        """
        self._current_elapsed_time += self.delta_time

        return self.get_current_value()
