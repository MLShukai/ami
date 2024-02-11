import pytest

from ami.interactions.environments.dummy_environment import (
    ActionTypeChecker,
    DummyEnvironment,
    SameObservationGenerator,
)


class TestDummyEnvironment:
    @pytest.fixture
    def dummy_env(self):
        return DummyEnvironment(
            SameObservationGenerator("observation"),
            ActionTypeChecker(int),
        )

    def test_observe(self, dummy_env: DummyEnvironment[str, int]):
        assert dummy_env.observe() == "observation"

    def test_affect(self, dummy_env):

        dummy_env.affect(0)

        with pytest.raises(ValueError):
            dummy_env.affect("str")
