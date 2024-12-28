import pytest
import torch
import torch.nn as nn

from ami.models.components.fully_connected_fixed_std_normal import (
    FullyConnectedFixedStdNormal,
    Normal,
)
from ami.models.components.sioconvps import SioConvPS
from ami.models.model_wrapper import ModelWrapper
from ami.models.temporal_encoder import MultimodalTemporalEncoder, inference_forward
from ami.utils import Modality

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1

MODALITY_DIMS = {Modality.IMAGE: 128, Modality.AUDIO: 32}


class TestMultimodalTemporalEncoder:
    @pytest.fixture
    def encoder(self):
        return MultimodalTemporalEncoder(
            observation_flattens={
                Modality.IMAGE: nn.Identity(),
                Modality.AUDIO: nn.Identity(),
            },
            flattened_obses_projection=nn.Linear(sum(MODALITY_DIMS.values()), DIM),
            core_model=SioConvPS(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT),
            obs_hat_dist_heads={m: FullyConnectedFixedStdNormal(DIM, d) for m, d in MODALITY_DIMS.items()},
        )

    @pytest.fixture
    def wrapped_encoder(self, encoder, device):
        wrapper = ModelWrapper(encoder, device, has_inference=True, inference_forward=inference_forward)
        wrapper.to_default_device()
        return wrapper

    def test_multimodal_temporal_encoder(self, encoder):
        obs_shapes = {m: (BATCH, LEN, d) for m, d in MODALITY_DIMS.items()}
        obses = {m: torch.randn(*s) for m, s in obs_shapes.items()}
        hidden_shape = (BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        embed_obs, hidden, next_obs_hats = encoder(obses, hidden[:, -1])
        assert embed_obs.shape == (BATCH, LEN, DIM)
        for m, d in next_obs_hats.items():
            assert isinstance(d, Normal)
            assert d.sample().shape == obs_shapes[m]

    def test_inference_forward(self, device, wrapped_encoder: ModelWrapper[MultimodalTemporalEncoder]):
        obs_shapes = {m: (BATCH, LEN, d) for m, d in MODALITY_DIMS.items()}
        obses = {m: torch.randn(*s) for m, s in obs_shapes.items()}
        hidden_shape = (BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        embed_obs, hidden = wrapped_encoder.infer(obses, hidden[:, -1])

        assert embed_obs.shape == (BATCH, LEN, DIM)
        assert hidden.shape == hidden_shape

        assert embed_obs.device == device
        assert hidden.device == device
