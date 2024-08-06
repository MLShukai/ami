import copy
from functools import partial

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict
from ami.models.i_jepa import IJEPAEncoder, IJEPAPredictor
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import ModelWrappersDict
from ami.tensorboard_loggers import StepIntervalLogger
from ami.trainers.components.i_jepa_mask_collator import IJEPAMultiBlockMaskCollator
from ami.trainers.i_jepa_trainer import IJEPATrainer

# input data params
IMAGE_SIZE = 256
PATCH_SIZE = 16
assert IMAGE_SIZE % PATCH_SIZE == 0
N_PATCHES = IMAGE_SIZE // PATCH_SIZE

# model params
ENCODER_EMBEDDING_DIM = 48
ENCODER_NUM_HEADS = 6
assert ENCODER_EMBEDDING_DIM % ENCODER_NUM_HEADS == 0
PREDICTOR_EMBEDDING_DIM = 24
PREDICTOR_NUM_HEADS = 3
assert PREDICTOR_EMBEDDING_DIM % PREDICTOR_NUM_HEADS == 0


class TestIJEPATrainer:
    @pytest.fixture
    def i_jepa_mask_collator(self) -> IJEPAMultiBlockMaskCollator:
        # based on the original paper settings
        i_jepa_mask_collator = IJEPAMultiBlockMaskCollator(
            input_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            encoder_mask_scale=(0.85, 1.0),
            predictor_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5),
            n_masks_for_context_encoder=1,
            n_masks_for_predictor=4,
            min_keep=10,
        )
        return i_jepa_mask_collator

    @pytest.fixture
    def partial_dataloader(self, i_jepa_mask_collator: IJEPAMultiBlockMaskCollator):
        partial_dataloader = partial(DataLoader, batch_size=2, shuffle=True, collate_fn=i_jepa_mask_collator)
        return partial_dataloader

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(AdamW, lr=1e-4, weight_decay=0.04)  # based on the original paper's initial params
        return partial_optimizer

    @pytest.fixture
    def i_jepa_encoder(self):
        return IJEPAEncoder(
            img_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            embed_dim=ENCODER_EMBEDDING_DIM,
            depth=3,
            num_heads=ENCODER_NUM_HEADS,
            mlp_ratio=4.0,
        )

    @pytest.fixture
    def i_jepa_predictor(self):
        return IJEPAPredictor(
            n_patches=N_PATCHES,
            context_encoder_embed_dim=ENCODER_EMBEDDING_DIM,
            predictor_embed_dim=PREDICTOR_EMBEDDING_DIM,
            depth=3,
            num_heads=PREDICTOR_NUM_HEADS,
        )

    @pytest.fixture
    def image_step_data(self) -> StepData:
        d = StepData()
        d[DataKeys.OBSERVATION] = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
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
    def model_wrappers_dict(
        self, device: torch.device, i_jepa_encoder: IJEPAEncoder, i_jepa_predictor: IJEPAPredictor
    ) -> ModelWrappersDict:
        d = ModelWrappersDict(
            {
                ModelNames.I_JEPA_CONTEXT_ENCODER: ModelWrapper(i_jepa_encoder, device, has_inference=False),
                ModelNames.I_JEPA_PREDICTOR: ModelWrapper(i_jepa_predictor, device, has_inference=False),
                ModelNames.I_JEPA_TARGET_ENCODER: ModelWrapper(
                    copy.deepcopy(i_jepa_encoder), device, has_inference=True
                ),
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
        model_wrappers_dict: DataCollectorsDict,
        image_buffer_dict: DataCollectorsDict,
        device: torch.device,
        logger: StepIntervalLogger,
    ) -> IJEPATrainer:
        trainer = IJEPATrainer(partial_dataloader, partial_optimizer, device, logger, minimum_new_data_count=1)
        trainer.attach_model_wrappers_dict(model_wrappers_dict)
        trainer.attach_data_users_dict(image_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: IJEPATrainer) -> None:
        trainer.run()

    def test_is_trainable(self, trainer: IJEPATrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.image_data_user.clear()
        assert trainer.is_trainable() is False

    def test_is_new_data_available(self, trainer: IJEPATrainer):
        trainer.image_data_user.update()
        assert trainer._is_new_data_available() is True
        trainer.run()
        assert trainer._is_new_data_available() is False

    def test_save_and_load_state(self, trainer: IJEPATrainer, tmp_path, mocker) -> None:
        trainer_path = tmp_path / "image_vae"
        trainer.save_state(trainer_path)
        assert trainer_path.exists()
        assert (trainer_path / "optimizer.pt").exists()
        assert (trainer_path / "logger.pt").exists()
        logger_state = trainer.logger.state_dict()

        mocked_logger_load_state_dict = mocker.spy(trainer.logger, "load_state_dict")
        trainer.optimizer_state.clear()
        assert trainer.optimizer_state == {}
        trainer.load_state(trainer_path)
        assert trainer.optimizer_state != {}
        mocked_logger_load_state_dict.assert_called_once_with(logger_state)
