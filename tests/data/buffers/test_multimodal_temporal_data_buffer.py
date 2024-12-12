import time

import pytest
import torch
from tensordict.tensordict import TensorDict
from torch.utils.data import TensorDataset

from ami.data.buffers.multimodal_temporal_data_buffer import (
    MultimodalTemporalDataBuffer,
)
from ami.data.step_data import DataKeys
from ami.utils import Modality


class TestMultimodalTemporalDataBuffer:
    max_len = 16
    observation_key = DataKeys.EMBED_OBSERVATION

    # テストデータの作成
    @pytest.fixture
    def sample_step_data(self):
        return {
            self.observation_key: {
                Modality.IMAGE: torch.randn(3, 224, 224),  # 画像データ
                Modality.AUDIO: torch.randn(1, 16000),  # 音声データ (16kHz)
            },
            DataKeys.HIDDEN: torch.randn(256),  # 隠れ状態を想定
        }

    def test__init__(self):
        mod = MultimodalTemporalDataBuffer(self.max_len)
        assert len(mod) == 0

    def test_add(self, sample_step_data):
        mod = MultimodalTemporalDataBuffer(self.max_len)
        mod.add(sample_step_data)
        mod.add(sample_step_data)
        mod.add(sample_step_data)
        assert len(mod) == 3

    def test_add_overflow(self, sample_step_data):
        mod = MultimodalTemporalDataBuffer(self.max_len)
        for _ in range(32):
            mod.add(sample_step_data)
        assert len(mod) == self.max_len

    def test_concatenate(self, sample_step_data):
        mod1 = MultimodalTemporalDataBuffer(self.max_len)
        previous_get_time = time.time()
        mod1.add(sample_step_data)
        mod1.add(sample_step_data)

        mod2 = MultimodalTemporalDataBuffer(self.max_len)
        mod2.add(sample_step_data)
        mod2.add(sample_step_data)
        mod2.add(sample_step_data)

        mod1.concatenate(mod2)
        assert len(mod1) == 5
        assert mod1.count_data_added_since(previous_get_time) == 5

    def test_make_dataset(self, sample_step_data):
        mod = MultimodalTemporalDataBuffer(self.max_len)
        mod.add(sample_step_data)
        mod.add(sample_step_data)

        dataset = mod.make_dataset()
        assert isinstance(dataset, TensorDataset)

        # データセットの形状を確認
        observations, hiddens = dataset[0]
        assert isinstance(observations, TensorDict)
        assert Modality.IMAGE in observations.keys()
        assert Modality.AUDIO in observations.keys()
        assert hiddens.shape == torch.Size([256])

    def test_count_data_added_since(self, sample_step_data):
        mod = MultimodalTemporalDataBuffer(self.max_len)
        previous_get_time = time.time()

        assert mod.count_data_added_since(previous_get_time) == 0
        mod.add(sample_step_data)
        mod.add(sample_step_data)
        assert mod.count_data_added_since(previous_get_time) == 2
        assert mod.count_data_added_since(time.time()) == 0

    def test_save_and_load_state(self, sample_step_data, tmp_path):
        mod = MultimodalTemporalDataBuffer(self.max_len)
        previous_get_time = float("-inf")
        mod.add(sample_step_data)

        data_dir = tmp_path / "data"
        mod.save_state(data_dir)

        # 必要なファイルが保存されているか確認
        assert (data_dir / "observations.pkl").exists()
        assert (data_dir / "hiddens.pkl").exists()
        assert (data_dir / "added_times.pkl").exists()

        # データセットの保存状態を取得
        saved_buffer_dataset = mod.make_dataset()

        # 新しいバッファーを作成して状態を読み込み
        mod = MultimodalTemporalDataBuffer(self.max_len)
        assert mod.count_data_added_since(previous_get_time) == 0
        mod.load_state(data_dir)
        assert mod.count_data_added_since(previous_get_time) == 1

        # 読み込んだデータセットを取得
        loaded_buffer_dataset = mod.make_dataset()

        # データセットの内容を比較
        assert len(loaded_buffer_dataset) == len(saved_buffer_dataset) == 1

        # TensorDictとTensorの比較
        loaded_obs, loaded_hidden = loaded_buffer_dataset[0]
        saved_obs, saved_hidden = saved_buffer_dataset[0]

        assert torch.equal(loaded_hidden, saved_hidden)
        assert all(torch.equal(loaded_obs[k], saved_obs[k]) for k in saved_obs.keys())
