import copy
from functools import partial

import pytest
import torch
import torchaudio
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
from ami.trainers.hifigan_trainer import (
    BoolMaskAudioJEPAEncoder,
    HifiGANGenerator,
    HifiGANTrainer,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)

# input data params
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_SIZE = 1040
PATCH_SAMPLE_SIZE = 400
STRIDE = 320
assert PATCH_SAMPLE_SIZE <= AUDIO_SAMPLE_SIZE
assert STRIDE <= PATCH_SAMPLE_SIZE
assert (AUDIO_SAMPLE_SIZE - (PATCH_SAMPLE_SIZE - STRIDE)) % STRIDE == 0
N_PATCHES = (AUDIO_SAMPLE_SIZE - (PATCH_SAMPLE_SIZE - STRIDE)) // STRIDE

# model params
ENCODER_EMBEDDING_DIM = 64
ENCODER_NUM_HEADS = 4
assert ENCODER_EMBEDDING_DIM % ENCODER_NUM_HEADS == 0
ENCODER_OUT_DIM = 32


@pytest.mark.parametrize(
    "generator_name",
    [ModelNames.HIFIGAN_CONTEXT_AURALIZATION_GENERATOR, ModelNames.HIFIGAN_TARGET_AURALIZATION_GENERATOR],
)
class TestHifiGANTrainer:
    @pytest.fixture
    def partial_dataloader(self):
        partial_dataloader = partial(DataLoader, batch_size=2, shuffle=True)
        return partial_dataloader

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(AdamW, lr=1e-4, weight_decay=0.01)
        return partial_optimizer

    @pytest.fixture
    def audio_buffer_dict(self) -> DataCollectorsDict:
        audio_step_data = StepData()
        audio_step_data[DataKeys.OBSERVATION] = torch.randn(AUDIO_CHANNELS, AUDIO_SAMPLE_SIZE)
        d = DataCollectorsDict.from_data_buffers(
            **{BufferNames.AUDIO: RandomDataBuffer.reconstructable_init(32, [DataKeys.OBSERVATION])}
        )

        for _ in range(4):
            d.collect(audio_step_data)
        return d

    @pytest.fixture
    def model_wrappers_dict(
        self,
        device: torch.device,
    ) -> ModelWrappersDict:
        encoder = BoolMaskAudioJEPAEncoder(
            input_sample_size=AUDIO_SAMPLE_SIZE,
            patch_sample_size=PATCH_SAMPLE_SIZE,
            stride=STRIDE,
            in_channels=AUDIO_CHANNELS,
            embed_dim=ENCODER_EMBEDDING_DIM,
            out_dim=ENCODER_EMBEDDING_DIM,
            depth=2,
            num_heads=ENCODER_NUM_HEADS,
            mlp_ratio=4.0,
        )
        # Setting params to match shapes between waveform reconstructed and original.
        generator = HifiGANGenerator(
            in_channels=ENCODER_EMBEDDING_DIM,
            out_channels=AUDIO_CHANNELS,
            upsample_rates=[10, 8, 2, 2],
            upsample_kernel_sizes=[20, 16, 4, 4],
            upsample_paddings=[4, 2, 1, 1],
            upsample_initial_channel=16,
        )
        multi_period_discriminator = MultiPeriodDiscriminator(in_channels=AUDIO_CHANNELS)
        multi_scale_discriminator = MultiScaleDiscriminator(in_channels=AUDIO_CHANNELS)
        d = ModelWrappersDict(
            {
                ModelNames.AUDIO_JEPA_CONTEXT_ENCODER: ModelWrapper(encoder, device, True),
                ModelNames.AUDIO_JEPA_TARGET_ENCODER: ModelWrapper(copy.deepcopy(encoder), device, False),
                ModelNames.HIFIGAN_CONTEXT_AURALIZATION_GENERATOR: ModelWrapper(generator, device, False),
                ModelNames.HIFIGAN_CONTEXT_AURALIZATION_MULTI_PERIOD_DISCRIMINATOR: ModelWrapper(
                    multi_period_discriminator, device, False
                ),
                ModelNames.HIFIGAN_CONTEXT_AURALIZATION_MULTI_SCALE_DISCRIMINATOR: ModelWrapper(
                    multi_scale_discriminator, device, False
                ),
                ModelNames.HIFIGAN_TARGET_AURALIZATION_GENERATOR: ModelWrapper(generator, device, False),
                ModelNames.HIFIGAN_TARGET_AURALIZATION_MULTI_PERIOD_DISCRIMINATOR: ModelWrapper(
                    multi_period_discriminator, device, False
                ),
                ModelNames.HIFIGAN_TARGET_AURALIZATION_MULTI_SCALE_DISCRIMINATOR: ModelWrapper(
                    multi_scale_discriminator, device, False
                ),
            }
        )
        d.send_to_default_device()
        return d

    @pytest.fixture
    def logger(self, tmp_path):
        return StepIntervalLogger(f"{tmp_path}/tensorboard", 1)

    @pytest.fixture
    def mel_spectrogram(self) -> torchaudio.transforms.MelSpectrogram:
        # ref: https://github.com/jik876/hifi-gan/blob/master/config_v1.json
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
            window_fn=torch.hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
        )

    @pytest.fixture
    def validation_dataloader(self):
        return DataLoader(TensorDataset(torch.randn(16, AUDIO_CHANNELS, AUDIO_SAMPLE_SIZE)), 8)

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_optimizer,
        model_wrappers_dict: ModelWrappersDict,
        audio_buffer_dict: DataCollectorsDict,
        device,
        logger,
        generator_name,
        mel_spectrogram: torchaudio.transforms.MelSpectrogram,
        validation_dataloader,
    ):
        trainer = HifiGANTrainer(
            partial_dataloader=partial_dataloader,
            partial_optimizer=partial_optimizer,
            device=device,
            logger=logger,
            generator_name=generator_name,
            mel_spectrogram=mel_spectrogram,
            rec_coef=45.0,
            minimum_new_data_count=1,
            validation_dataloader=validation_dataloader,
            num_auralize_audios=4,
        )
        trainer.attach_model_wrappers_dict(model_wrappers_dict)
        trainer.attach_data_users_dict(audio_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: HifiGANTrainer) -> None:
        trainer.run()

    def test_is_trainable(self, trainer: HifiGANTrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.audio_data_user.clear()
        assert trainer.is_trainable() is False

    def test_is_new_data_available(self, trainer: HifiGANTrainer):
        trainer.audio_data_user.update()
        assert trainer._is_new_data_available() is True
        trainer.run()
        assert trainer._is_new_data_available() is False

    def test_save_and_load_state(self, trainer: HifiGANTrainer, tmp_path, mocker) -> None:
        trainer_path = tmp_path / "bool_mask_audio_jepa"
        trainer.save_state(trainer_path)
        assert trainer_path.exists()
        assert (trainer_path / "optimizer_g.pt").exists()
        assert (trainer_path / "optimizer_mpd.pt").exists()
        assert (trainer_path / "optimizer_msd.pt").exists()
        assert (trainer_path / "logger.pt").exists()
        assert (trainer_path / "dataset_previous_get_time.pt").exists()
        logger_state = trainer.logger.state_dict()
        dataset_previous_get_time = trainer.dataset_previous_get_time

        mocked_logger_load_state_dict = mocker.spy(trainer.logger, "load_state_dict")
        trainer.optimizer_state_g.clear()
        trainer.optimizer_state_mpd.clear()
        trainer.optimizer_state_msd.clear()
        trainer.dataset_previous_get_time = None
        assert trainer.optimizer_state_g == {}
        assert trainer.optimizer_state_mpd == {}
        assert trainer.optimizer_state_msd == {}
        trainer.load_state(trainer_path)
        assert trainer.optimizer_state_g != {}
        assert trainer.optimizer_state_mpd != {}
        assert trainer.optimizer_state_msd != {}
        mocked_logger_load_state_dict.assert_called_once_with(logger_state)
        assert trainer.dataset_previous_get_time == dataset_previous_get_time
