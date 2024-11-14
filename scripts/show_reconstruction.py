import rootutils
import torch
import torchvision
import torchvision.transforms.v2

from ami.interactions.io_wrappers.function_wrapper import normalize_tensor
from ami.models.bool_mask_i_jepa import BoolMaskIJEPAEncoder
from ami.models.i_jepa_latent_visualization_decoder import (
    IJEPALatentVisualizationDecoder,
)
from ami.utils import min_max_normalize

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")

encoder_parameter_file = PROJECT_ROOT / "data/2024-09-14_09-42-23,678417.ckpt/i_jepa_target_encoder.pt"
decoder_parameter_file = PROJECT_ROOT / "data/2024-09-14_09-42-23,678417.ckpt/i_jepa_target_visualization_decoder.pt"


image_paths = [PROJECT_ROOT / "data/random_observation_action_log/validation/000000.jpg"]

encoder = BoolMaskIJEPAEncoder(
    img_size=144, patch_size=12, in_channels=3, embed_dim=648, out_dim=32, depth=12, num_heads=9, mlp_ratio=4.0
)

decoder = IJEPALatentVisualizationDecoder(
    input_n_patches=12,
    input_latents_dim=32,
    decoder_blocks_in_and_out_channels=[(512, 512), (512, 256), (256, 128), (128, 64)],
    n_res_blocks=3,
    num_heads=4,
)

encoder.to(device)
decoder.to(device)

encoder.load_state_dict(torch.load(encoder_parameter_file, map_location=device))
decoder.load_state_dict(torch.load(decoder_parameter_file, map_location=device))

reconstructions = []
for image_path in image_paths:
    original = torchvision.io.read_image(str(image_path)).float()
    input = normalize_tensor(original).to(device)

    reconstruction = decoder(encoder(input.unsqueeze(0)))

    reconstruction = min_max_normalize(reconstruction.flatten(), new_min=0, new_max=255).reshape(reconstruction.shape)
    reconstructions.append(reconstruction.squeeze(0).byte().cpu())

reconstruction_grid = torchvision.utils.make_grid(reconstructions, nrow=1)
torchvision.io.write_png(reconstruction_grid, str(PROJECT_ROOT / "data" / "reconstruction.png"))
