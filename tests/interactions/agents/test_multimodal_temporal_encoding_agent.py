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
from ami.models.temporal_encoder import MultimodalTemporalEncoder, inference_forward
from ami.models.utils import ModelWrappersDict
from ami.utils import Modality


class TestMultimodalTemporalEncodingAgent:
    # Define test dimensions
    RAW_IMAGE_DIM = 8  # Raw image input dimension
    RAW_AUDIO_DIM = 10  # Raw audio input dimension
    IMAGE_DIM = 4  # Encoded image dimension
    AUDIO_DIM = 6  # Encoded audio dimension
    HIDDEN_DIM = 8
    DEPTH = 4

    @pytest.fixture
    def unimodal_agents(self, mocker: MockerFixture) -> dict[Modality, UnimodalEncodingAgent]:
        """Create mock unimodal encoding agents.

        These agents should encode raw input dimensions to encoded dimensions:
        - Image: RAW_IMAGE_DIM -> IMAGE_DIM
        - Audio: RAW_AUDIO_DIM -> AUDIO_DIM
        """
        image_agent = mocker.create_autospec(UnimodalEncodingAgent, instance=True)
        image_agent.step.return_value = torch.randn(self.IMAGE_DIM)
        audio_agent = mocker.create_autospec(UnimodalEncodingAgent, instance=True)
        audio_agent.step.return_value = torch.randn(self.AUDIO_DIM)
        return {Modality.IMAGE: image_agent, Modality.AUDIO: audio_agent}

    @pytest.fixture
    def models(self) -> ModelWrappersDict:
        """Create mock models dictionary for testing.

        The MultimodalTemporalEncoder expects encoded inputs from
        unimodal agents. Input dimensions should match the encoded
        dimensions (IMAGE_DIM, AUDIO_DIM).
        """
        observation_flattens = {Modality.IMAGE: nn.Identity(), Modality.AUDIO: nn.Identity()}
        # Project concatenated encoded features
        flattened_obses_projection = nn.Linear(self.IMAGE_DIM + self.AUDIO_DIM, self.HIDDEN_DIM)
        core_model = SioConvPS(self.DEPTH, self.HIDDEN_DIM, self.HIDDEN_DIM * 2, False)
        obs_hat_dist_heads = {
            Modality.IMAGE: nn.Linear(self.HIDDEN_DIM, self.IMAGE_DIM),
            Modality.AUDIO: nn.Linear(self.HIDDEN_DIM, self.AUDIO_DIM),
        }
        temporal_encoder = MultimodalTemporalEncoder(
            observation_flattens=observation_flattens,
            flattened_obses_projection=flattened_obses_projection,
            core_model=core_model,
            obs_hat_dist_heads=obs_hat_dist_heads,
        )

        mwd = ModelWrappersDict(
            {
                ModelNames.MULTIMODAL_TEMPORAL_ENCODER: ModelWrapper(
                    temporal_encoder, "cpu", inference_forward=inference_forward
                ),
            }
        )
        return mwd.inference_wrappers_dict

    @pytest.fixture
    def data_collectors(self) -> DataCollectorsDict:
        """Create mock data collectors for testing."""
        temporal_buf = MultimodalTemporalDataBuffer(max_len=10)
        return DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.MULTIMODAL_TEMPORAL: temporal_buf,
            }
        )

    @pytest.fixture
    def agent(self, models, data_collectors, unimodal_agents) -> MultimodalTemporalEncodingAgent:
        """Create a MultimodalTemporalEncodingAgent instance for testing."""
        initial_hidden = torch.zeros(self.DEPTH, self.HIDDEN_DIM)
        agent = MultimodalTemporalEncodingAgent(initial_hidden, unimodal_agents)
        agent.attach_inference_models(models)
        agent.attach_data_collectors(data_collectors)
        return agent

    def test_step_dimensions(self, agent, unimodal_agents):
        """Test the dimensions of inputs and outputs in the step function."""
        # Create raw observations
        observation = {Modality.IMAGE: torch.randn(self.RAW_IMAGE_DIM), Modality.AUDIO: torch.randn(self.RAW_AUDIO_DIM)}

        # Store original observation for input verification
        orig_observation = {k: v.clone() for k, v in observation.items()}

        # Test step function
        output = agent.step(observation)

        # Verify unimodal agents were called with correct inputs
        for modality, unimodal_agent in unimodal_agents.items():
            unimodal_agent.step.assert_called_once()
            assert torch.equal(unimodal_agent.step.call_args[0][0], orig_observation[modality])

        # Verify dimensions
        assert output.shape == (self.HIDDEN_DIM,)
        assert agent.encoder_hidden_state.shape == (self.DEPTH, self.HIDDEN_DIM)

    def test_temporal_encoding(self, agent, unimodal_agents):
        """Test if temporal information is being encoded properly."""
        # Create two different raw observations
        obs1 = {Modality.IMAGE: torch.ones(self.RAW_IMAGE_DIM), Modality.AUDIO: torch.ones(self.RAW_AUDIO_DIM)}
        obs2 = {Modality.IMAGE: torch.zeros(self.RAW_IMAGE_DIM), Modality.AUDIO: torch.zeros(self.RAW_AUDIO_DIM)}

        # Process sequences and verify hidden states change
        hidden_state_1 = agent.encoder_hidden_state.clone()
        _ = agent.step(obs1)
        hidden_state_2 = agent.encoder_hidden_state.clone()
        _ = agent.step(obs2)
        hidden_state_3 = agent.encoder_hidden_state.clone()

        # Hidden states should be different after processing different observations
        assert not torch.allclose(hidden_state_1, hidden_state_2)
        assert not torch.allclose(hidden_state_2, hidden_state_3)

        # Verify unimodal agents were called with correct inputs
        for modality, unimodal_agent in unimodal_agents.items():
            assert unimodal_agent.step.call_count == 2
            assert torch.equal(unimodal_agent.step.call_args_list[0][0][0], obs1[modality])
            assert torch.equal(unimodal_agent.step.call_args_list[1][0][0], obs2[modality])

    def test_buffer_collection(self, agent: MultimodalTemporalEncodingAgent, unimodal_agents):
        """Test if observations and hidden states are being collected
        properly."""
        observation = {Modality.IMAGE: torch.randn(self.RAW_IMAGE_DIM), Modality.AUDIO: torch.randn(self.RAW_AUDIO_DIM)}

        # Store original observation and current hidden state
        orig_observation = {k: v.clone() for k, v in observation.items()}
        orig_hidden = agent.encoder_hidden_state.clone()

        initial_buffer_size = len(agent.collector._buffer)
        _ = agent.step(observation)

        # Verify buffer size increased
        buffer = agent.collector._buffer
        assert len(buffer) == initial_buffer_size + 1

        # Get collected data and verify its structure
        dataset = buffer.make_dataset()
        assert len(dataset) == len(buffer)

        # Verify the collected data
        last_data = dataset[-1]
        observation, hidden = last_data
        assert torch.equal(hidden, orig_hidden)

        # Verify encoded observations were stored
        for modality, unimodal_agent in unimodal_agents.items():
            assert torch.equal(unimodal_agent.step.call_args[0][0], orig_observation[modality])
            assert torch.equal(observation[modality], unimodal_agent.step.return_value)

    def test_save_load_state(self, agent: MultimodalTemporalEncodingAgent, tmp_path: Path, unimodal_agents):
        """Test the state saving and loading functionality."""
        agent_path = tmp_path / "agent"

        # Generate some initial states
        orig_hidden = agent.encoder_hidden_state.clone()

        # Test save state
        agent.save_state(agent_path)
        assert (agent_path / "encoder_hidden_state.pt").exists()

        # Verify save was called for each unimodal agent with correct paths
        for modality, unimodal_agent in unimodal_agents.items():
            expected_path = agent_path / modality
            unimodal_agent.save_state.assert_called_once_with(expected_path)

        # Modify states
        agent.encoder_hidden_state.random_()
        assert not torch.equal(agent.encoder_hidden_state, orig_hidden)

        # Test load state
        agent.load_state(agent_path)

        # Verify states were restored
        assert torch.equal(agent.encoder_hidden_state, orig_hidden)

        # Verify load was called for each unimodal agent with correct paths
        for modality, unimodal_agent in unimodal_agents.items():
            expected_path = agent_path / modality
            unimodal_agent.load_state.assert_called_once_with(expected_path)
