from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from ami.interactions.environments.unity_environment import UnityEnvironment


class TestUnityEnvironment:
    @pytest.fixture
    def mock_unity_env(self):
        with patch("ami.interactions.environments.unity_environment.RawUnityEnv") as mock_raw_env, patch(
            "ami.interactions.environments.unity_environment.UnityToGymWrapper"
        ) as mock_gym_wrapper:
            mock_raw_env.return_value = Mock()
            mock_gym_wrapper.return_value = Mock()
            yield mock_gym_wrapper.return_value

    @pytest.fixture
    def unity_env(self, mock_unity_env):
        return UnityEnvironment("dummy_path", worker_id=0, base_port=5005, seed=42)

    def test_initialization(self, unity_env, mock_unity_env):
        assert isinstance(unity_env, UnityEnvironment)
        assert unity_env._env == mock_unity_env

    def test_setup(self, unity_env, mock_unity_env):
        mock_unity_env.reset.return_value = torch.zeros(10)
        unity_env.setup()
        assert torch.equal(unity_env.observation, torch.zeros(10))

    def test_teardown(self, unity_env, mock_unity_env):
        unity_env.teardown()
        mock_unity_env.close.assert_called_once()

    def test_affect(self, unity_env, mock_unity_env):
        action = torch.tensor([1.0, 2.0, 3.0])
        mock_unity_env.step.return_value = (torch.ones(10), 0, False, {})
        unity_env.affect(action)
        mock_unity_env.step.assert_called_once()
        assert torch.equal(unity_env.observation, torch.ones(10))

    def test_observe(self, unity_env):
        unity_env.observation = np.ones(10)
        observation = unity_env.observe()
        assert torch.equal(observation, torch.ones(10))

    def test_gym_unity_environment(self, unity_env, mock_unity_env):
        assert unity_env.gym_unity_environment == mock_unity_env

    def test_raw_unity_environment(self, unity_env, mock_unity_env):
        assert unity_env.raw_unity_environment == mock_unity_env._env
