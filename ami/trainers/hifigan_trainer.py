import time
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms.v2.functional
import torchvision.utils
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.bool_mask_audio_jepa import BoolMaskAudioJEPAEncoder
from ami.models.hifigan.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from ami.models.hifigan.generator import HifiGANGenerator
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.tensorboard_loggers import StepIntervalLogger
from ami.utils import min_max_normalize

from .base_trainer import BaseTrainer


class HifiGANTrainer(BaseTrainer):
    """Trainer for HifiGAN to auralize AudioJEPA latents."""

    @override
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        generator_name: Literal[
            ModelNames.HIFIGAN_CONTEXT_AURALIZATION_GENERATOR, ModelNames.HIFIGAN_TARGET_AURALIZATION_GENERATOR
        ],
        mel_spectrogram: torchaudio.transforms.MelSpectrogram,
        rec_coef: float = 45.0,
        max_epochs: int = 1,
        minimum_dataset_size: int = 1,
        minimum_new_data_count: int = 0,
        validation_dataloader: DataLoader[tuple[Tensor]] | None = None,
        num_auralize_audios: int = 4,
    ) -> None:
        """Initializes an HifiGANTrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            logger: The logger object for recording training metrics and auralizations.
            generator_name: Name of the generator (context or target) to be trained for auralization.
            mel_spectrogram: Converter waveform into mel-spec.
            rec_coef: Coefficient for reconstruction loss.
            max_epochs: Maximum number of epochs to train the generator and discriminators. Default is 1.
            minimum_dataset_size: Minimum number of samples required in the dataset to start training. Default is 1.
            minimum_new_data_count: Minimum number of new data samples required to run the training. Default is 0.
            validation_dataloader DataLoader instance for validation.
            num_auralize_audios: Number of audios to use for auralization. Default is 4.
        """
        super().__init__()

        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.logger = logger
        self.log_prefix = "audio-jepa-latent-auralization/"

        # Prepare self.log_prefix correspond to generator name.
        match generator_name:
            case ModelNames.HIFIGAN_CONTEXT_AURALIZATION_GENERATOR:
                multi_period_discriminator_name = ModelNames.HIFIGAN_CONTEXT_AURALIZATION_MULTI_PERIOD_DISCRIMINATOR
                multi_scale_discriminator_name = ModelNames.HIFIGAN_CONTEXT_AURALIZATION_MULTI_SCALE_DISCRIMINATOR
                encoder_name = ModelNames.AUDIO_JEPA_CONTEXT_ENCODER
                self.log_prefix += "context/"
            case ModelNames.HIFIGAN_TARGET_AURALIZATION_GENERATOR:
                multi_period_discriminator_name = ModelNames.HIFIGAN_TARGET_AURALIZATION_MULTI_PERIOD_DISCRIMINATOR
                multi_scale_discriminator_name = ModelNames.HIFIGAN_TARGET_AURALIZATION_MULTI_SCALE_DISCRIMINATOR
                encoder_name = ModelNames.AUDIO_JEPA_TARGET_ENCODER
                self.log_prefix += "target/"
            case _:
                raise ValueError(f"Unexpected generator_name: {generator_name}")

        self.encoder_name = encoder_name
        self.generator_name = generator_name
        self.multi_period_discriminator_name = multi_period_discriminator_name
        self.multi_scale_discriminator_name = multi_scale_discriminator_name

        self.mel_spectrogram = mel_spectrogram
        self.rec_coef = rec_coef

        self.max_epochs = max_epochs
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count
        self.validation_dataloader = validation_dataloader
        self.num_auralize_audios = num_auralize_audios

        self.dataset_previous_get_time = float("-inf")

    def on_data_users_dict_attached(self) -> None:
        self.audio_data_user: ThreadSafeDataUser[RandomDataBuffer] = self.get_data_user(BufferNames.AUDIO)

    def on_model_wrappers_dict_attached(self) -> None:
        self.encoder: ModelWrapper[BoolMaskAudioJEPAEncoder] = self.get_frozen_model(self.encoder_name)
        self.generator: ModelWrapper[HifiGANGenerator] = self.get_training_model(self.generator_name)
        self.multi_period_discriminator: ModelWrapper[MultiPeriodDiscriminator] = self.get_training_model(
            self.multi_period_discriminator_name
        )
        self.multi_scale_discriminator: ModelWrapper[MultiScaleDiscriminator] = self.get_training_model(
            self.multi_scale_discriminator_name
        )

        self.optimizer_state_g = self.partial_optimizer(self.generator.parameters()).state_dict()
        self.optimizer_state_mpd = self.partial_optimizer(self.multi_period_discriminator.parameters()).state_dict()
        self.optimizer_state_msd = self.partial_optimizer(self.multi_scale_discriminator.parameters()).state_dict()

    @override
    def is_trainable(self) -> bool:
        self.audio_data_user.update()
        return len(self.audio_data_user.buffer) >= self.minimum_dataset_size and self._is_new_data_available()

    def _is_new_data_available(self) -> bool:
        return (
            self.audio_data_user.buffer.count_data_added_since(self.dataset_previous_get_time)
            >= self.minimum_new_data_count
        )

    def get_dataset(self) -> Dataset[Tensor]:
        dataset = self.audio_data_user.get_dataset()
        self.dataset_previous_get_time = time.time()
        return dataset

    @torch.inference_mode()
    def validation(self, dataloader: DataLoader[tuple[Tensor]]) -> None:
        """Compute the reconstruction loss and log auralization."""

        input_audio_batch_list = []
        reconstruction_audio_batch_list = []
        losses_adv_mpd = []
        losses_adv_msd = []
        losses_rec = []
        losses_fm = []
        losses_adv_g = []

        batch: tuple[Tensor]
        for batch in dataloader:
            (audio_batch,) = batch
            input_audio_batch_list.append(audio_batch)
            audio_batch = audio_batch.to(self.device)

            latents = self.encoder.infer(audio_batch)
            latents = latents.transpose(-1, -2)
            reconstructions: Tensor = self.generator(latents)
            reconstruction_audio_batch_list.append(reconstructions.cpu())

            # multi_period_discriminator
            authenticity_list_real, _ = self.multi_period_discriminator(audio_batch)
            authenticity_list_fake, _ = self.multi_period_discriminator(reconstructions.detach())
            loss_adv_mpd = self._discriminator_adversarial_losses(authenticity_list_real, authenticity_list_fake)
            losses_adv_mpd.append(loss_adv_mpd)

            # multi_scale_discriminator
            authenticity_list_real, _ = self.multi_scale_discriminator(audio_batch)
            authenticity_list_fake, _ = self.multi_scale_discriminator(reconstructions.detach())
            loss_adv_msd = self._discriminator_adversarial_losses(authenticity_list_real, authenticity_list_fake)
            losses_adv_msd.append(loss_adv_msd)

            # generator
            # calc reconstruction loss
            losses_rec.append(
                self._reconstruction_loss(
                    waveform_batch=audio_batch,
                    waveform_reconstructed=reconstructions,
                )
                * self.rec_coef
            )
            # calc feature matching loss
            authenticity_list_mpd_real, fmaps_list_mpd_real = self.multi_period_discriminator(audio_batch)
            authenticity_list_mpd_fake, fmaps_list_mpd_fake = self.multi_period_discriminator(reconstructions)
            authenticity_list_msd_real, fmaps_list_msd_real = self.multi_scale_discriminator(audio_batch)
            authenticity_list_msd_fake, fmaps_list_msd_fake = self.multi_scale_discriminator(reconstructions)
            loss_fm = self._feature_loss(fmaps_list_mpd_real, fmaps_list_mpd_fake) + self._feature_loss(
                fmaps_list_msd_real, fmaps_list_msd_fake
            )
            losses_fm.append(loss_fm)
            # calc adversarial loss
            loss_adv_g = self._generator_adversarial_losses(
                authenticity_list_mpd_fake
            ) + self._generator_adversarial_losses(authenticity_list_msd_fake)
            losses_adv_g.append(loss_adv_g)

        input_audio_batches = torch.cat(input_audio_batch_list)
        reconstruction_audio_batches = torch.cat(reconstruction_audio_batch_list)
        auralize_indices = torch.randperm(input_audio_batches.size(0))[: self.num_auralize_audios].sort().values
        input_audio_selected = input_audio_batches[auralize_indices]
        input_audio_selected = min_max_normalize(input_audio_selected.flatten(1), 0, 1, dim=-1).reshape(
            input_audio_selected.shape
        )
        reconstruction_audio_selected = reconstruction_audio_batches[auralize_indices]
        reconstruction_audio_selected = min_max_normalize(
            reconstruction_audio_selected.flatten(1), 0, 1, dim=-1
        ).reshape(reconstruction_audio_selected.shape)

        self.logger.log(
            self.log_prefix + "losses/mpd/valid-adv", torch.mean(torch.stack(losses_adv_mpd)), force_log=True
        )
        self.logger.log(
            self.log_prefix + "losses/msd/valid-adv", torch.mean(torch.stack(losses_adv_msd)), force_log=True
        )
        self.logger.log(self.log_prefix + "losses/g/valid-rec", torch.mean(torch.stack(losses_rec)), force_log=True)
        self.logger.log(self.log_prefix + "losses/g/valid-fm", torch.mean(torch.stack(losses_fm)), force_log=True)
        self.logger.log(self.log_prefix + "losses/g/valid-adv", torch.mean(torch.stack(losses_adv_g)), force_log=True)

        for i, (in_audio, rec_audio) in enumerate(zip(input_audio_selected, reconstruction_audio_selected)):
            self.logger.tensorboard.add_audio(
                self.log_prefix + f"metrics/input-{i}",
                in_audio,
                self.logger.global_step,
                sample_rate=self.mel_spectrogram.sample_rate,
            )
            self.logger.tensorboard.add_audio(
                self.log_prefix + f"metrics/reconstruction-{i}",
                rec_audio,
                self.logger.global_step,
                sample_rate=self.mel_spectrogram.sample_rate,
            )

    @override
    def train(self) -> None:
        # move to device
        self.encoder.to(self.device)
        self.generator.to(self.device)
        self.multi_period_discriminator.to(self.device)
        self.multi_scale_discriminator.to(self.device)

        optimizer_g = self.partial_optimizer(self.generator.parameters())
        optimizer_mpd = self.partial_optimizer(self.multi_period_discriminator.parameters())
        optimizer_msd = self.partial_optimizer(self.multi_scale_discriminator.parameters())

        optimizer_g.load_state_dict(self.optimizer_state_g)
        optimizer_mpd.load_state_dict(self.optimizer_state_mpd)
        optimizer_msd.load_state_dict(self.optimizer_state_msd)

        # prepare about dataset
        dataloader = self.partial_dataloader(dataset=self.get_dataset())

        for _ in range(self.max_epochs):
            batch: tuple[Tensor]
            for batch in dataloader:
                (audio_batch,) = batch
                audio_batch = audio_batch.to(self.device)

                with torch.no_grad():
                    latents = self.encoder.infer(audio_batch)
                    # latents: [batch_size, n_patches, latents_dim]
                    latents = latents.transpose(-1, -2)

                # reconstruct
                audio_out: Tensor = self.generator(latents)

                # train multi_period_discriminator
                authenticity_list_real, _ = self.multi_period_discriminator(audio_batch)
                authenticity_list_fake, _ = self.multi_period_discriminator(audio_out.detach())
                loss_adv_mpd = self._discriminator_adversarial_losses(authenticity_list_real, authenticity_list_fake)
                optimizer_mpd.zero_grad()
                loss_adv_mpd.backward()
                optimizer_mpd.step()
                self.logger.log(self.log_prefix + "losses/mpd/train_adv", loss_adv_mpd)

                # train multi_scale_discriminator
                authenticity_list_real, _ = self.multi_scale_discriminator(audio_batch)
                authenticity_list_fake, _ = self.multi_scale_discriminator(audio_out.detach())
                loss_adv_msd = self._discriminator_adversarial_losses(authenticity_list_real, authenticity_list_fake)
                optimizer_msd.zero_grad()
                loss_adv_msd.backward()
                optimizer_msd.step()
                self.logger.log(self.log_prefix + "losses/msd/train_adv", loss_adv_msd)

                # train generator
                # calc reconstruction loss
                loss_rec = (
                    self._reconstruction_loss(
                        waveform_batch=audio_batch,
                        waveform_reconstructed=audio_out,
                    )
                    * self.rec_coef
                )
                # calc feature matching loss
                authenticity_list_mpd_real, fmaps_list_mpd_real = self.multi_period_discriminator(audio_batch)
                authenticity_list_mpd_fake, fmaps_list_mpd_fake = self.multi_period_discriminator(audio_out)
                authenticity_list_msd_real, fmaps_list_msd_real = self.multi_scale_discriminator(audio_batch)
                authenticity_list_msd_fake, fmaps_list_msd_fake = self.multi_scale_discriminator(audio_out)
                loss_fm = self._feature_loss(fmaps_list_mpd_real, fmaps_list_mpd_fake) + self._feature_loss(
                    fmaps_list_msd_real, fmaps_list_msd_fake
                )
                # calc adversarial loss
                loss_adv_g = self._generator_adversarial_losses(
                    authenticity_list_mpd_fake
                ) + self._generator_adversarial_losses(authenticity_list_msd_fake)
                # calc sum
                loss_g = loss_rec + loss_fm + loss_adv_g
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()
                self.logger.log(self.log_prefix + "losses/g/train_rec", loss_rec)
                self.logger.log(self.log_prefix + "losses/g/train_fm", loss_fm)
                self.logger.log(self.log_prefix + "losses/g/train_adv", loss_adv_g)

                self.logger.update()

        if self.validation_dataloader is not None:
            self.validation(self.validation_dataloader)

        self.optimizer_state_g = optimizer_g.state_dict()
        self.optimizer_state_mpd = optimizer_mpd.state_dict()
        self.optimizer_state_msd = optimizer_msd.state_dict()
        self.logger_state = self.logger.state_dict()

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.optimizer_state_g, path / "optimizer_g.pt")
        torch.save(self.optimizer_state_mpd, path / "optimizer_mpd.pt")
        torch.save(self.optimizer_state_msd, path / "optimizer_msd.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")
        torch.save(self.dataset_previous_get_time, path / "dataset_previous_get_time.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.optimizer_state_g = torch.load(path / "optimizer_g.pt")
        self.optimizer_state_mpd = torch.load(path / "optimizer_mpd.pt")
        self.optimizer_state_msd = torch.load(path / "optimizer_msd.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))
        self.dataset_previous_get_time = torch.load(path / "dataset_previous_get_time.pt")

    def _discriminator_adversarial_losses(
        self, authenticity_list_real: list[torch.Tensor], authenticity_list_fake: list[torch.Tensor]
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=torch.float32)
        for authenticity_real, authenticity_fake in zip(authenticity_list_real, authenticity_list_fake):
            real_loss = torch.mean((1 - authenticity_real) ** 2)
            fake_loss = torch.mean(authenticity_fake**2)
            loss += real_loss + fake_loss
        return loss

    def _feature_loss(
        self, fmaps_list_real: list[list[torch.Tensor]], fmaps_list_fake: list[list[torch.Tensor]]
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=torch.float32)
        for fmaps_real, fmaps_fake in zip(fmaps_list_real, fmaps_list_fake):
            for fmap_real, fmap_fake in zip(fmaps_real, fmaps_fake):
                loss += torch.mean(torch.abs(fmap_real - fmap_fake))
        return loss * 2

    def _generator_adversarial_losses(self, authenticity_list_fake: list[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=torch.float32)
        for authenticity_fake in authenticity_list_fake:
            loss += torch.mean((1 - authenticity_fake) ** 2)
        return loss

    def _spectral_normalize(self, x: torch.Tensor, clip_val: float = 1e-5) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=clip_val))

    def _reconstruction_loss(self, waveform_batch: torch.Tensor, waveform_reconstructed: torch.Tensor) -> torch.Tensor:
        mel_batch = self._spectral_normalize(self.mel_spectrogram(waveform_batch))
        mel_reconstructed = self._spectral_normalize(self.mel_spectrogram(waveform_reconstructed))
        loss_rec = F.l1_loss(mel_batch, mel_reconstructed)
        return loss_rec
