from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict, DataUsersDict
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import ModelWrappersDict
from ami.models.vae import (
    Conv2dDecoder,
    Conv2dEncoder,
    Decoder,
    Encoder,
    EncoderWrapper,
)
from ami.tensorboard_loggers import StepIntervalLogger
from ami.trainers.image_vae_trainer import ImageVAETrainer


class TestImageVAETrainer:
    @pytest.fixture
    def partial_dataloader(self):
        partial_dataloader = partial(DataLoader, batch_size=2, shuffle=True)
        return partial_dataloader

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(Adam, lr=0.001)
        return partial_optimizer

    @pytest.fixture
    def image_encoder(self):
        return Conv2dEncoder(256, 256, 3, 256)

    @pytest.fixture
    def image_decoder(self):
        return Conv2dDecoder(256, 256, 3, 256)

    @pytest.fixture
    def image_step_data(self) -> StepData:
        d = StepData()
        d[DataKeys.OBSERVATION] = torch.randn(3, 256, 256)
        return d

    @pytest.fixture
    def image_buffer_dict(self, image_step_data: StepData) -> DataCollectorsDict:
        d = DataCollectorsDict.from_data_buffers(
            **{BufferNames.IMAGE: RandomDataBuffer.reconstructable_init(32, [DataKeys.OBSERVATION])}
        )

        d.collect(image_step_data)
        d.collect(image_step_data)
        d.collect(image_step_data)
        d.collect(image_step_data)
        return d

    @pytest.fixture
    def encoder_decoder_wrappers_dict(
        self, device: torch.device, image_encoder: Encoder, image_decoder: Decoder
    ) -> ModelWrappersDict:
        d = ModelWrappersDict(
            {
                ModelNames.IMAGE_ENCODER: EncoderWrapper(image_encoder, device, True),
                ModelNames.IMAGE_DECODER: ModelWrapper(image_decoder, device, False),
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
        encoder_decoder_wrappers_dict: DataCollectorsDict,
        image_buffer_dict: DataCollectorsDict,
        device: torch.device,
        logger: StepIntervalLogger,
    ) -> ImageVAETrainer:
        trainer = ImageVAETrainer(partial_dataloader, partial_optimizer, device, logger)
        trainer.attach_model_wrappers_dict(encoder_decoder_wrappers_dict)
        trainer.attach_data_users_dict(image_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: ImageVAETrainer) -> None:
        trainer.run()

    def test_is_trainable(self, trainer: ImageVAETrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.image_data_user.clear()
        assert trainer.is_trainable() is False

    def test_save_and_load_state(self, trainer: ImageVAETrainer, tmp_path) -> None:
        trainer_path = tmp_path / "image_vae"
        trainer.save_state(trainer_path)
        assert trainer_path.exists()
        assert (trainer_path / "optimizer.pt").exists()
        assert (trainer_path / "logger.pt").exists()

        trainer.optimizer_state.clear()
        assert trainer.optimizer_state == {}
        trainer.load_state(trainer_path)
        assert trainer.optimizer_state != {}
