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
from ami.models.bool_mask_i_jepa import BoolMaskIJEPAEncoder, BoolTargetIJEPAPredictor
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import ModelWrappersDict
from ami.tensorboard_loggers import StepIntervalLogger
from ami.trainers.bool_mask_i_jepa_trainer import BoolMaskIJEPATrainer
from ami.trainers.components.bool_i_jepa_mask_collator import (
    BoolIJEPAMultiBlockMaskCollator,
)

# input data params
IMAGE_SIZE = 256
PATCH_SIZE = 16
assert IMAGE_SIZE % PATCH_SIZE == 0
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# model params
ENCODER_EMBEDDING_DIM = 48
ENCODER_NUM_HEADS = 6
assert ENCODER_EMBEDDING_DIM % ENCODER_NUM_HEADS == 0
PREDICTOR_HIDDEN_DIM = 24
PREDICTOR_NUM_HEADS = 3
assert PREDICTOR_HIDDEN_DIM % PREDICTOR_NUM_HEADS == 0


class TestBoolMaskIJEPATrainer:
    @pytest.fixture
    def bool_i_jepa_mask_collator(self) -> BoolIJEPAMultiBlockMaskCollator:
        bool_i_jepa_mask_collator = BoolIJEPAMultiBlockMaskCollator(
            input_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            n_masks=4,
            aspect_ratio=(0.75, 1.5),
            min_keep=10,
        )
        return bool_i_jepa_mask_collator

    @pytest.fixture
    def partial_dataloader(self, bool_i_jepa_mask_collator: BoolIJEPAMultiBlockMaskCollator):
        partial_dataloader = partial(DataLoader, batch_size=2, shuffle=True, collate_fn=bool_i_jepa_mask_collator)
        return partial_dataloader

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(AdamW, lr=1e-4, weight_decay=0.04)
        return partial_optimizer

    @pytest.fixture
    def bool_mask_i_jepa_encoder(self):
        return BoolMaskIJEPAEncoder(
            img_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            embed_dim=ENCODER_EMBEDDING_DIM,
            out_dim=ENCODER_EMBEDDING_DIM,
            depth=3,
            num_heads=ENCODER_NUM_HEADS,
            mlp_ratio=4.0,
        )

    @pytest.fixture
    def bool_target_i_jepa_predictor(self):
        return BoolTargetIJEPAPredictor(
            n_patches=(IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE),
            context_encoder_out_dim=ENCODER_EMBEDDING_DIM,
            hidden_dim=PREDICTOR_HIDDEN_DIM,
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

        for _ in range(4):
            d.collect(image_step_data)
        return d

    @pytest.fixture
    def model_wrappers_dict(
        self,
        device: torch.device,
        bool_mask_i_jepa_encoder: BoolMaskIJEPAEncoder,
        bool_target_i_jepa_predictor: BoolTargetIJEPAPredictor,
    ) -> ModelWrappersDict:
        d = ModelWrappersDict(
            {
                ModelNames.I_JEPA_CONTEXT_ENCODER: ModelWrapper(bool_mask_i_jepa_encoder, device, has_inference=False),
                ModelNames.I_JEPA_PREDICTOR: ModelWrapper(bool_target_i_jepa_predictor, device, has_inference=False),
                ModelNames.I_JEPA_TARGET_ENCODER: ModelWrapper(
                    copy.deepcopy(bool_mask_i_jepa_encoder), device, has_inference=True
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
        model_wrappers_dict: ModelWrappersDict,
        image_buffer_dict: DataCollectorsDict,
        device: torch.device,
        logger: StepIntervalLogger,
    ) -> BoolMaskIJEPATrainer:
        trainer = BoolMaskIJEPATrainer(partial_dataloader, partial_optimizer, device, logger, minimum_new_data_count=1)
        trainer.attach_model_wrappers_dict(model_wrappers_dict)
        trainer.attach_data_users_dict(image_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: BoolMaskIJEPATrainer) -> None:
        trainer.run()

    def test_is_trainable(self, trainer: BoolMaskIJEPATrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.image_data_user.clear()
        assert trainer.is_trainable() is False

    def test_is_new_data_available(self, trainer: BoolMaskIJEPATrainer):
        trainer.image_data_user.update()
        assert trainer._is_new_data_available() is True
        trainer.run()
        assert trainer._is_new_data_available() is False

    def test_save_and_load_state(self, trainer: BoolMaskIJEPATrainer, tmp_path, mocker) -> None:
        trainer_path = tmp_path / "bool_mask_i_jepa"
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
