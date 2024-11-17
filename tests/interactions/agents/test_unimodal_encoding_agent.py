import pytest
import torch

from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.utils import DataCollectorsDict
from ami.interactions.agents.unimodal_encoding_agent import (
    BufferNames,
    DataKeys,
    Modality,
    UnimodalEncodingAgent,
)
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import ModelWrappersDict


class TestUnimodalEncodingAgent:
    @pytest.fixture
    def models(self) -> ModelWrappersDict:
        image_encoder = torch.nn.Conv2d(3, 16, 32)
        audio_encoder = torch.nn.Conv1d(2, 8, 1600)
        mwd = ModelWrappersDict(
            {
                ModelNames.IMAGE_ENCODER: ModelWrapper(image_encoder, "cpu"),
                ModelNames.AUDIO_ENCODER: ModelWrapper(audio_encoder, "cpu"),
            }
        )
        return mwd.inference_wrappers_dict

    @pytest.fixture
    def data_collectors(self) -> DataCollectorsDict:
        image_buf = RandomDataBuffer.reconstructable_init(max_len=10, key_list=[DataKeys.OBSERVATION])
        audio_buf = RandomDataBuffer.reconstructable_init(max_len=10, key_list=[DataKeys.OBSERVATION])
        return DataCollectorsDict.from_data_buffers(**{BufferNames.IMAGE: image_buf, BufferNames.AUDIO: audio_buf})

    observation_shapes = {Modality.IMAGE: (3, 32, 32), Modality.AUDIO: (2, 1600)}
    encoded_shapes = {Modality.IMAGE: (16, 1, 1), Modality.AUDIO: (8, 1)}

    @pytest.mark.parametrize("modality", [Modality.IMAGE, Modality.AUDIO])
    def test_step(self, modality, models, data_collectors):
        agent = UnimodalEncodingAgent(modality)
        agent.attach_data_collectors(data_collectors)
        agent.attach_inference_models(models)

        obs = torch.randn(self.observation_shapes[modality])
        out = agent.step(obs)
        assert out.shape == self.encoded_shapes[modality]

        buf: RandomDataBuffer = agent.collector.move_data()
        dataset = buf.make_dataset()
        assert dataset[:][0].shape == (1, *self.observation_shapes[modality])
