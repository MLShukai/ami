from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pytest_mock import MockerFixture

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.multimodal_temporal_data_buffer import (
    MultimodalTemporalDataBuffer,
)
from ami.data.utils import DataCollectorsDict
from ami.interactions.agents.multimodal_temporal_encoding_agent import (
    MultimodalTemporalEncodingAgent,
)
from ami.interactions.agents.unimodal_encoding_agent import UnimodalEncodingAgent
from ami.models.components.sioconvps import SioConvPS
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.temporal_encoder import MultimodalTemporalEncoder
from ami.models.utils import ModelWrappersDict
from ami.utils import Modality
from tests.helpers import DataBufferImpl


class TestMultimodalTemporalEncodingAgent:
    # Define test dimensions
    IMAGE_DIM = 4
    AUDIO_DIM = 6
    HIDDEN_DIM = 8
    DEPTH = 4

    @pytest.fixture
    def temporal_encoder(self) -> MultimodalTemporalEncoder:
        """Create a mock temporal encoder for testing.

        Returns:
            MultimodalTemporalEncoder: A simple temporal encoder for testing.
        """
        observation_flattens = {Modality.IMAGE: nn.Identity(), Modality.AUDIO: nn.Identity()}
        # Project concatenated features (IMAGE_DIM + AUDIO_DIM) to PROJECTION_DIM
        flattened_obses_projection = nn.Linear(self.IMAGE_DIM + self.AUDIO_DIM, self.HIDDEN_DIM)
        core_model = SioConvPS(self.DEPTH, self.HIDDEN_DIM, self.HIDDEN_DIM * 2, False)
        obs_hat_dist_heads = {
            Modality.IMAGE: nn.Linear(self.HIDDEN_DIM, self.IMAGE_DIM),
            Modality.AUDIO: nn.Linear(self.HIDDEN_DIM, self.AUDIO_DIM),
        }
        return MultimodalTemporalEncoder(
            observation_flattens=observation_flattens,
            flattened_obses_projection=flattened_obses_projection,
            core_model=core_model,
            obs_hat_dist_heads=obs_hat_dist_heads,
        )

    @pytest.fixture
    def models(self, temporal_encoder) -> ModelWrappersDict:
        """Create mock models dictionary for testing.

        Returns:
            ModelWrappersDict: Dictionary containing all required model wrappers.
        """
        mwd = ModelWrappersDict(
            {
                ModelNames.IMAGE_ENCODER: ModelWrapper(nn.Identity(), "cpu"),
                ModelNames.AUDIO_ENCODER: ModelWrapper(nn.Identity(), "cpu"),
                ModelNames.MULTIMODAL_TEMPORAL_ENCODER: ModelWrapper(temporal_encoder, "cpu"),
            }
        )
        return mwd.inference_wrappers_dict

    @pytest.fixture
    def data_collectors(self) -> DataCollectorsDict:
        """Create mock data collectors for testing.

        Returns:
            DataCollectorsDict: Dictionary containing all required data collectors.
        """
        temporal_buf = MultimodalTemporalDataBuffer(max_len=10)
        return DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.MULTIMODAL_TEMPORAL: temporal_buf,
                BufferNames.IMAGE: DataBufferImpl(),
                BufferNames.AUDIO: DataBufferImpl(),
            }
        )

    @pytest.fixture
    def agent(self, models, data_collectors, mocker) -> MultimodalTemporalEncodingAgent:
        """Create a MultimodalTemporalEncodingAgent instance for testing.

        Returns:
            MultimodalTemporalEncodingAgent: The agent instance for testing.
        """
        unimodal_agents = {
            Modality.IMAGE: UnimodalEncodingAgent(Modality.IMAGE),
            Modality.AUDIO: UnimodalEncodingAgent(Modality.AUDIO),
        }

        initial_hidden = torch.zeros(self.DEPTH, self.HIDDEN_DIM)
        agent = MultimodalTemporalEncodingAgent(unimodal_agents, initial_hidden)
        agent.attach_inference_models(models)
        agent.attach_data_collectors(data_collectors)
        return agent

    def test_step_dimensions(self, agent):
        """Test the dimensions of inputs and outputs in the step function."""
        # Create simple observations
        observation = {Modality.IMAGE: torch.randn(self.IMAGE_DIM), Modality.AUDIO: torch.randn(self.AUDIO_DIM)}

        # Test step function
        output = agent.step(observation)

        # Verify dimensions
        assert output.shape == (self.HIDDEN_DIM,)
        assert agent.encoder_hidden_state.shape == (self.DEPTH, self.HIDDEN_DIM)

    def test_temporal_encoding(self, agent):
        """Test if temporal information is being encoded properly."""
        # Create two different observations
        obs1 = {Modality.IMAGE: torch.ones(self.IMAGE_DIM), Modality.AUDIO: torch.ones(self.AUDIO_DIM)}

        obs2 = {Modality.IMAGE: torch.zeros(self.IMAGE_DIM), Modality.AUDIO: torch.zeros(self.AUDIO_DIM)}

        # Process sequences and verify hidden states change
        hidden_state_1 = agent.encoder_hidden_state.clone()
        _ = agent.step(obs1)
        hidden_state_2 = agent.encoder_hidden_state.clone()
        _ = agent.step(obs2)
        hidden_state_3 = agent.encoder_hidden_state.clone()

        # Hidden states should be different after processing different observations
        assert not torch.allclose(hidden_state_1, hidden_state_2)
        assert not torch.allclose(hidden_state_2, hidden_state_3)

    def test_buffer_collection(self, agent: MultimodalTemporalEncodingAgent):
        """Test if observations and hidden states are being collected
        properly."""
        observation = {Modality.IMAGE: torch.randn(self.IMAGE_DIM), Modality.AUDIO: torch.randn(self.AUDIO_DIM)}

        initial_buffer_size = len(agent.collector._buffer)
        _ = agent.step(observation)

        # Verify buffer size increased
        buffer = agent.collector._buffer
        assert len(buffer) == initial_buffer_size + 1

        # Get collected data and verify its structure
        dataset = buffer.make_dataset()
        assert len(dataset) == len(buffer)

    def test_save_load_state(self, agent: MultimodalTemporalEncodingAgent, tmp_path: Path, mocker: MockerFixture):
        """Test the state saving and loading functionality."""
        agent_path = tmp_path / "agent"

        save_methods = []
        save_method_expected_arg = []
        load_methods = []
        load_method_expected_arg = []

        for m, a in agent.unimodal_agents.items():
            save_methods.append(mocker.spy(a, "save_state"))
            save_method_expected_arg.append(agent_path / m)
            load_methods.append(mocker.spy(a, "load_state"))
            load_method_expected_arg.append(agent_path / m)

        agent.save_state(agent_path)
        assert (agent_path / "encoder_hidden_state.pt").exists()
        for mock, expected in zip(save_methods, save_method_expected_arg):
            mock.assert_called_once_with(expected)

        hidden = agent.encoder_hidden_state.clone()
        agent.encoder_hidden_state.random_()
        assert not torch.equal(agent.encoder_hidden_state, hidden)

        agent.load_state(agent_path)
        assert torch.equal(agent.encoder_hidden_state, hidden)
        for mock, expected in zip(load_methods, load_method_expected_arg):
            mock.assert_called_once_with(expected)
