import pytest
from pytest_mock import MockerFixture

from ami.interactions.interaction import (
    BaseActionWrapper,
    BaseAgent,
    BaseEnvironment,
    BaseObservationWrapper,
    Interaction,
)


class TestInteraction:
    @pytest.fixture
    def mock_obs_wrapper(self, mocker: MockerFixture):
        wrapper = mocker.Mock(spec=BaseObservationWrapper)
        wrapper.wrap.side_effect = lambda x: f"wrapped_obs({x})"
        return wrapper

    @pytest.fixture
    def mock_act_wrapper(self, mocker: MockerFixture):
        wrapper = mocker.Mock(spec=BaseActionWrapper)
        wrapper.wrap.side_effect = lambda x: f"wrapped_act({x})"
        return wrapper

    @pytest.fixture
    def interaction(self, mock_env, mock_agent, mock_obs_wrapper, mock_act_wrapper):
        return Interaction(
            mock_env, mock_agent, observation_wrappers=[mock_obs_wrapper], action_wrappers=[mock_act_wrapper]
        )

    def test_setup(self, interaction, mock_env, mock_agent, mock_obs_wrapper, mock_act_wrapper):
        interaction.setup()

        mock_obs_wrapper.setup.assert_called_once()
        mock_act_wrapper.setup.assert_called_once()
        mock_env.setup.assert_called_once()
        mock_agent.setup.assert_called_once()

    def test_step(self, interaction, mock_env, mock_agent):
        interaction.step()

        mock_agent.step.assert_called_once_with("wrapped_obs(observation)")
        mock_env.affect.assert_called_once_with("wrapped_act(action)")

    def test_teardown(self, interaction, mock_env, mock_agent, mock_obs_wrapper, mock_act_wrapper):
        interaction.teardown()

        mock_agent.teardown.assert_called_once()
        mock_env.teardown.assert_called_once()
        mock_obs_wrapper.teardown.assert_called_once()
        mock_act_wrapper.teardown.assert_called_once()

    def test_save_and_load_state(self, interaction, mock_env, mock_agent, mock_obs_wrapper, mock_act_wrapper, tmp_path):
        interaction_path = tmp_path / "interaction"

        interaction.save_state(interaction_path)
        assert interaction_path.exists()
        mock_env.save_state.assert_called_once_with(interaction_path / "environment")
        mock_agent.save_state.assert_called_once_with(interaction_path / "agent")
        mock_obs_wrapper.save_state.assert_called_once_with(interaction_path / "observation_wrapper.0")
        mock_act_wrapper.save_state.assert_called_once_with(interaction_path / "action_wrapper.0")

        interaction.load_state(interaction_path)
        mock_env.load_state.assert_called_once_with(interaction_path / "environment")
        mock_agent.load_state.assert_called_once_with(interaction_path / "agent")
        mock_obs_wrapper.load_state.assert_called_once_with(interaction_path / "observation_wrapper.0")
        mock_act_wrapper.load_state.assert_called_once_with(interaction_path / "action_wrapper.0")

    def test_pause_resume_event_callbacks(self, interaction, mock_env, mock_agent, mock_obs_wrapper, mock_act_wrapper):
        interaction.on_paused()
        mock_env.on_paused.assert_called_once()
        mock_agent.on_paused.assert_called_once()
        mock_obs_wrapper.on_paused.assert_called_once()
        mock_act_wrapper.on_paused.assert_called_once()

        interaction.on_resumed()
        mock_env.on_resumed.assert_called_once()
        mock_agent.on_resumed.assert_called_once()
        mock_obs_wrapper.on_resumed.assert_called_once()
        mock_act_wrapper.on_resumed.assert_called_once()

    def test_multiple_wrappers(self, mocker: MockerFixture):
        mock_env = mocker.Mock(spec=BaseEnvironment)
        mock_agent = mocker.Mock(spec=BaseAgent)
        mock_obs_wrapper1 = mocker.Mock(spec=BaseObservationWrapper)
        mock_obs_wrapper2 = mocker.Mock(spec=BaseObservationWrapper)
        mock_act_wrapper1 = mocker.Mock(spec=BaseActionWrapper)
        mock_act_wrapper2 = mocker.Mock(spec=BaseActionWrapper)

        mock_obs_wrapper1.wrap.side_effect = lambda x: f"obs1({x})"
        mock_obs_wrapper2.wrap.side_effect = lambda x: f"obs2({x})"
        mock_act_wrapper1.wrap.side_effect = lambda x: f"act1({x})"
        mock_act_wrapper2.wrap.side_effect = lambda x: f"act2({x})"

        mock_env.observe.return_value = "raw_obs"
        mock_agent.step.return_value = "raw_act"

        interaction = Interaction(
            mock_env,
            mock_agent,
            observation_wrappers=[mock_obs_wrapper1, mock_obs_wrapper2],
            action_wrappers=[mock_act_wrapper1, mock_act_wrapper2],
        )

        interaction.step()

        mock_agent.step.assert_called_once_with("obs2(obs1(raw_obs))")
        mock_env.affect.assert_called_once_with("act2(act1(raw_act))")
