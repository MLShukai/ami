import pytest
import torch
from pytest_mock import MockerFixture

from ami.interactions.environments.sensors.dict_multimodal_sensor import (
    BaseSensor,
    DictMultimodalSensor,
    Modality,
)


class TestDictMultimodalSensor:
    @pytest.fixture
    def mock_sensors(self, mocker: MockerFixture):
        mock_image_sensor = mocker.Mock(spec=BaseSensor)
        mock_image_sensor.read.return_value = torch.zeros(3, 480, 640)

        mock_audio_sensor = mocker.Mock(spec=BaseSensor)
        mock_audio_sensor.read.return_value = torch.ones(2, 1600)

        return {Modality.IMAGE: mock_image_sensor, Modality.AUDIO: mock_audio_sensor}

    @pytest.fixture
    def multimodal_sensor(self, mock_sensors):
        return DictMultimodalSensor(mock_sensors)

    def test_init(self, mocker: MockerFixture):

        with pytest.raises(ValueError):
            DictMultimodalSensor({"invalid_modality": mocker.Mock(spec=BaseSensor)})

    def test_read(self, multimodal_sensor):
        out = multimodal_sensor.read()

        assert torch.equal(torch.zeros(3, 480, 640), out[Modality.IMAGE])
        assert torch.equal(torch.ones(2, 1600), out[Modality.AUDIO])

    def test_setup(self, multimodal_sensor, mock_sensors):
        multimodal_sensor.setup()
        for mock in mock_sensors.values():
            mock.setup.assert_called_once()

    def test_teardown(self, multimodal_sensor, mock_sensors):
        multimodal_sensor.teardown()
        for mock in mock_sensors.values():
            mock.teardown.assert_called_once()
