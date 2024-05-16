import torch

from ami.models.components.fc_complex_to_real import FCComplexToReal


class TestFCComplexToReal:
    def test_fc_complex_to_real(self):
        batch_size = 8
        in_features = 16
        out_features = 32

        model = FCComplexToReal(in_features, out_features)
        input_tensor = torch.randn(batch_size, in_features, dtype=torch.cfloat)
        output_tensor = model(input_tensor)

        assert output_tensor.shape == (
            batch_size,
            out_features,
        ), f"Expected shape {(batch_size, out_features)}, but got {output_tensor.shape}"
