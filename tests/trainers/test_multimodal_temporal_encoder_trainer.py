from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.optim import Adam
from torch.utils.data import DataLoader

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.multimodal_temporal_data_buffer import (
    MultimodalTemporalDataBuffer,
)
from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict
from ami.models.components.fully_connected_fixed_std_normal import (
    FullyConnectedFixedStdNormal,
)
from ami.models.components.sioconvps import SioConvPS
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.temporal_encoder import MultimodalTemporalEncoder
from ami.models.utils import ModelWrappersDict
from ami.tensorboard_loggers import StepIntervalLogger
from ami.trainers.components.random_time_series_sampler import RandomTimeSeriesSampler
from ami.trainers.components.transpose_and_stack_collator import (
    transpose_and_stack_collator,
)
from ami.trainers.multimodal_temporal_encoder_trainer import (
    MultimodalTemporalEncoderTrainer,
)
from ami.utils import Modality


class TestMultimodalTemporalEncoderTrainer:
    # Define test dimensions
    IMAGE_DIM = 32
    AUDIO_DIM = 16
    HIDDEN_DIM = 64
    DEPTH = 4
    BATCH_SIZE = 2

    @pytest.fixture
    def observation_flattens(self):
        return {Modality.IMAGE: nn.Identity(), Modality.AUDIO: nn.Identity()}

    @pytest.fixture
    def flattened_obses_projection(self):
        return nn.Linear(self.IMAGE_DIM + self.AUDIO_DIM, self.HIDDEN_DIM)

    @pytest.fixture
    def core_model(self):
        return SioConvPS(self.DEPTH, self.HIDDEN_DIM, self.HIDDEN_DIM * 2, False)

    @pytest.fixture
    def obs_hat_dist_heads(self):
        return {
            Modality.IMAGE: FullyConnectedFixedStdNormal(self.HIDDEN_DIM, self.IMAGE_DIM),
            Modality.AUDIO: FullyConnectedFixedStdNormal(self.HIDDEN_DIM, self.AUDIO_DIM),
        }

    @pytest.fixture
    def temporal_encoder(self, observation_flattens, flattened_obses_projection, core_model, obs_hat_dist_heads):
        return MultimodalTemporalEncoder(
            observation_flattens=observation_flattens,
            flattened_obses_projection=flattened_obses_projection,
            core_model=core_model,
            obs_hat_dist_heads=obs_hat_dist_heads,
        )

    @pytest.fixture
    def step_data(self) -> StepData:
        d = StepData()
        obs_dict = {Modality.IMAGE: torch.randn(self.IMAGE_DIM), Modality.AUDIO: torch.randn(self.AUDIO_DIM)}
        d[DataKeys.OBSERVATION] = TensorDict(obs_dict, batch_size=())
        d[DataKeys.HIDDEN] = torch.randn(self.DEPTH, self.HIDDEN_DIM)
        return d

    @pytest.fixture
    def buffer_dict(self, step_data: StepData) -> DataCollectorsDict:
        d = DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.MULTIMODAL_TEMPORAL: MultimodalTemporalDataBuffer.reconstructable_init(
                    max_len=32,
                )
            }
        )

        # Add multiple data points
        for _ in range(4):
            d.collect(step_data)
        return d

    @pytest.fixture
    def partial_dataloader(self):
        return partial(DataLoader, batch_size=self.BATCH_SIZE, collate_fn=transpose_and_stack_collator)

    @pytest.fixture
    def partial_optimizer(self):
        return partial(Adam, lr=0.001)

    @pytest.fixture
    def model_wrappers_dict(self, temporal_encoder, device):
        d = ModelWrappersDict(
            {
                ModelNames.MULTIMODAL_TEMPORAL_ENCODER: ModelWrapper(temporal_encoder, device, True),
            }
        )
        d.send_to_default_device()
        return d

    @pytest.fixture
    def logger(self, tmp_path):
        return StepIntervalLogger(f"{tmp_path}/tensorboard", 1)

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_optimizer,
        device,
        model_wrappers_dict,
        buffer_dict,
        logger,
    ):
        trainer = MultimodalTemporalEncoderTrainer(
            partial_dataloader=partial_dataloader,
            partial_sampler=partial(RandomTimeSeriesSampler, sequence_length=2),
            partial_optimizer=partial_optimizer,
            device=device,
            logger=logger,
            minimum_new_data_count=2,
        )
        trainer.attach_model_wrappers_dict(model_wrappers_dict)
        trainer.attach_data_users_dict(buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: MultimodalTemporalEncoderTrainer) -> None:
        """Test if training can run without errors."""
        trainer.run()

    def test_is_trainable(self, trainer: MultimodalTemporalEncoderTrainer) -> None:
        """Test the trainable condition checking."""
        assert trainer.is_trainable() is True
        trainer.temporal_data_user.clear()
        assert trainer.is_trainable() is False

    def test_is_new_data_available(self, trainer: MultimodalTemporalEncoderTrainer):
        """Test if new data availability is correctly tracked."""
        trainer.temporal_data_user.update()
        assert trainer._is_new_data_available() is True
        trainer.run()
        assert trainer._is_new_data_available() is False

    def test_save_and_load_state(self, trainer: MultimodalTemporalEncoderTrainer, tmp_path: Path, mocker) -> None:
        """Test state saving and loading functionality."""
        trainer_path = tmp_path / "multimodal_temporal_encoder"
        trainer.save_state(trainer_path)

        # Check if files exist
        assert trainer_path.exists()
        assert (trainer_path / "optimizer.pt").exists()
        assert (trainer_path / "logger.pt").exists()
        assert (trainer_path / "dataset_previous_get_time.pt").exists()

        # Store states for comparison
        logger_state = trainer.logger.state_dict()
        dataset_previous_get_time = trainer.dataset_previous_get_time

        # Spy on logger's load_state_dict
        mocked_logger_load_state_dict = mocker.spy(trainer.logger, "load_state_dict")

        # Clear states
        trainer.optimizer_state.clear()
        trainer.dataset_previous_get_time = float("-inf")
        assert trainer.optimizer_state == {}

        # Load and verify states
        trainer.load_state(trainer_path)
        assert trainer.optimizer_state != {}
        mocked_logger_load_state_dict.assert_called_once_with(logger_state)
        assert trainer.dataset_previous_get_time == dataset_previous_get_time
