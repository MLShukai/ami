import time

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from ami.interactions.environments.unity_environment import UnityEnvironment


class TestUnityEnvironment:
    @pytest.fixture
    def mock_unity_env(self, mocker: MockerFixture):
        """内部で使用されているUnity環境との連携クラスをモックで置換する。"""
        mock_raw_env = mocker.patch("ami.interactions.environments.unity_environment.RawUnityEnv")
        mock_gym_wrapper = mocker.patch("ami.interactions.environments.unity_environment.UnityToGymWrapper")
        return mock_gym_wrapper()  # インスタンスモックを返す。

    @pytest.fixture
    def unity_env(self, mock_unity_env):
        mock_unity_env.reset.return_value = np.zeros(10)
        mock_unity_env.step.return_value = (np.ones(10), 0, False, {})
        env = UnityEnvironment("dummy_path", worker_id=0, base_port=5005, seed=42)
        env.setup()
        yield env
        env.teardown()

    def test_setup(self, unity_env, mock_unity_env):
        np.testing.assert_equal(unity_env.observation, np.zeros(10))

    def test_teardown(self, unity_env, mock_unity_env):
        unity_env.teardown()
        mock_unity_env.close.assert_called_once()

    def test_affect(self, unity_env, mock_unity_env):
        action = torch.tensor([1.0, 2.0, 3.0])
        unity_env.affect(action)
        time.sleep(0.01)
        mock_unity_env.step.assert_called_once()
        np.testing.assert_equal(unity_env.observe(), np.ones(10))

    def test_observe(self, unity_env):
        unity_env.observation = np.zeros(10)
        action = torch.tensor([1.0, 2.0, 3.0])
        unity_env.affect(action)
        time.sleep(0.01)
        observation = unity_env.observe()
        assert torch.equal(observation, torch.ones(10))

    def test_gym_unity_environment(self, unity_env, mock_unity_env):
        assert unity_env.gym_unity_environment == mock_unity_env

    def test_raw_unity_environment(self, unity_env, mock_unity_env):
        assert unity_env.raw_unity_environment == mock_unity_env._env
