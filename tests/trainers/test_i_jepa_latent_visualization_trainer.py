import copy
from functools import partial

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import ModelWrappersDict
from ami.tensorboard_loggers import StepIntervalLogger
from ami.trainers.i_jepa_latent_visualization_trainer import (
    BoolMaskIJEPAEncoder,
    IJEPALatentVisualizationDecoder,
    IJEPALatentVisualizationDecoderTrainer,
)

# input data params
IMAGE_SIZE = 144
PATCH_SIZE = 12
assert IMAGE_SIZE % PATCH_SIZE == 0
N_PATCHES = IMAGE_SIZE // PATCH_SIZE

# model params
ENCODER_EMBEDDING_DIM = 64
ENCODER_NUM_HEADS = 4
assert ENCODER_EMBEDDING_DIM % ENCODER_NUM_HEADS == 0
ENCODER_OUT_DIM = 32


@pytest.mark.parametrize(
    "decoder_name",
    [ModelNames.I_JEPA_CONTEXT_VISUALIZATION_DECODER, ModelNames.I_JEPA_TARGET_VISUALIZATION_DECODER],
)
class TestIJEPALatentVisualizationTrainer:
    @pytest.fixture
    def partial_dataloader(self):
        partial_dataloader = partial(DataLoader, batch_size=2, shuffle=True)
        return partial_dataloader

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(AdamW, lr=1e-4, weight_decay=0.01)
        return partial_optimizer

    @pytest.fixture
    def image_buffer_dict(self) -> DataCollectorsDict:
        image_step_data = StepData()
        image_step_data[DataKeys.OBSERVATION] = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
        d = DataCollectorsDict.from_data_buffers(
            **{BufferNames.IMAGE: RandomDataBuffer.reconstructable_init(32, [DataKeys.OBSERVATION])}
        )

        for _ in range(4):
            d.collect(image_step_data)
        return d

    @pytest.fixture
    def model_wrappers_dict(
        self,
        device: torch.device,
    ) -> ModelWrappersDict:
        encoder = BoolMaskIJEPAEncoder(
            img_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            embed_dim=ENCODER_EMBEDDING_DIM,
            out_dim=ENCODER_OUT_DIM,
            depth=3,
            num_heads=ENCODER_NUM_HEADS,
            mlp_ratio=4.0,
        )
        decoder = IJEPALatentVisualizationDecoder(
            input_n_patches=N_PATCHES,
            input_latents_dim=ENCODER_OUT_DIM,
            decoder_blocks_in_and_out_channels=[(32, 32)] * 3,
            n_res_blocks=3,
            num_heads=4,
        )
        d = ModelWrappersDict(
            {
                ModelNames.I_JEPA_CONTEXT_ENCODER: ModelWrapper(encoder, device, True),
                ModelNames.I_JEPA_TARGET_ENCODER: ModelWrapper(copy.deepcopy(encoder), device, False),
                ModelNames.I_JEPA_CONTEXT_VISUALIZATION_DECODER: ModelWrapper(decoder, device, False),
                ModelNames.I_JEPA_TARGET_VISUALIZATION_DECODER: ModelWrapper(decoder, device, False),
            }
        )
        d.send_to_default_device()
        return d

    @pytest.fixture
    def logger(self, tmp_path):
        return StepIntervalLogger(f"{tmp_path}/tensorboard", 1)

    @pytest.fixture
    def validation_dataloader(self):
        return DataLoader(TensorDataset(torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE)), 8)

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_optimizer,
        model_wrappers_dict: ModelWrappersDict,
        image_buffer_dict: DataCollectorsDict,
        device,
        logger,
        decoder_name,
        validation_dataloader,
    ):
        trainer = IJEPALatentVisualizationDecoderTrainer(
            partial_dataloader=partial_dataloader,
            partial_optimizer=partial_optimizer,
            device=device,
            logger=logger,
            decoder_name=decoder_name,
            minimum_new_data_count=1,
            validation_dataloader=validation_dataloader,
            num_visualize_images=8,
        )
        trainer.attach_model_wrappers_dict(model_wrappers_dict)
        trainer.attach_data_users_dict(image_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: IJEPALatentVisualizationDecoderTrainer) -> None:
        trainer.run()

    def test_is_trainable(self, trainer: IJEPALatentVisualizationDecoderTrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.image_data_user.clear()
        assert trainer.is_trainable() is False

    def test_is_new_data_available(self, trainer: IJEPALatentVisualizationDecoderTrainer):
        trainer.image_data_user.update()
        assert trainer._is_new_data_available() is True
        trainer.run()
        assert trainer._is_new_data_available() is False

    def test_save_and_load_state(self, trainer: IJEPALatentVisualizationDecoderTrainer, tmp_path, mocker) -> None:
        trainer_path = tmp_path / "bool_mask_i_jepa"
        trainer.save_state(trainer_path)
        assert trainer_path.exists()
        assert (trainer_path / "optimizer.pt").exists()
        assert (trainer_path / "logger.pt").exists()
        assert (trainer_path / "dataset_previous_get_time.pt").exists()
        logger_state = trainer.logger.state_dict()
        dataset_previous_get_time = trainer.dataset_previous_get_time

        mocked_logger_load_state_dict = mocker.spy(trainer.logger, "load_state_dict")
        trainer.optimizer_state.clear()
        trainer.dataset_previous_get_time = None
        assert trainer.optimizer_state == {}
        trainer.load_state(trainer_path)
        assert trainer.optimizer_state != {}
        mocked_logger_load_state_dict.assert_called_once_with(logger_state)
        assert trainer.dataset_previous_get_time == dataset_previous_get_time
