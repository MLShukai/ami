# Parameter: https://drive.google.com/file/d/1saWq1fAbyA_BCZryUiWgPixJ6JCssLy0/view?usp=sharing
# In Linux, run `PYTHON=($which python3) && $PYTHON scripts/play_imagination_world.py``

import time

import cv2
import keyboard
import rootutils
import torch
import torchvision
import torchvision.transforms.functional
from torch import Tensor

from ami.interactions.environments.actuators.vrchat_osc_discrete_actuator import (
    ACTION_CHOICES_PER_CATEGORY,
)
from ami.models.bool_mask_i_jepa import BoolMaskIJEPAEncoder
from ami.models.components.discrete_policy_head import DiscretePolicyHead
from ami.models.components.fully_connected_fixed_std_normal import (
    DeterministicNormal,
    FullyConnectedFixedStdNormal,
)
from ami.models.components.multi_embeddings import MultiEmbeddings
from ami.models.components.sioconv import SioConv
from ami.models.components.stacked_features import (
    LerpStackedFeatures,
    ToStackedFeatures,
)
from ami.models.forward_dynamics import ForwardDynamcisWithActionReward
from ami.models.i_jepa_latent_visualization_decoder import (
    IJEPALatentVisualizationDecoder,
)


class KeyboardActionHandler:
    def __init__(self) -> None:
        self.initial_actions = {"MoveVertical": 0, "MoveHorizontal": 0, "LookHorizontal": 0, "Jump": 0, "Run": 0}
        self.actions = self.initial_actions.copy()

        self.key_map = {
            "w": ("MoveVertical", 1),
            "s": ("MoveVertical", 2),
            "d": ("MoveHorizontal", 1),
            "a": ("MoveHorizontal", 2),
            ".": ("LookHorizontal", 1),
            ",": ("LookHorizontal", 2),
            "space": ("Jump", 1),
            "shift": ("Run", 1),
        }

    def update(self) -> None:
        self.actions = self.initial_actions.copy()

        for key, (action, value) in self.key_map.items():
            if keyboard.is_pressed(key):
                self.actions[action] = value

    def get_action(self) -> Tensor:
        return torch.tensor(
            [
                self.actions["MoveVertical"],
                self.actions["MoveHorizontal"],
                self.actions["LookHorizontal"],
                self.actions["Jump"],
                self.actions["Run"],
            ]
        )


@torch.inference_mode()
def main() -> None:
    PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    encoder_parameter_file = PROJECT_ROOT / "data/2024-09-17_04-47-37,195367.ckpt/i_jepa_target_encoder.pt"
    encoder = BoolMaskIJEPAEncoder(
        img_size=(144, 144),
        in_channels=3,
        patch_size=12,
        embed_dim=648,
        out_dim=32,
        depth=12,
        num_heads=9,
        mlp_ratio=4.0,
    )
    encoder.to(device)
    encoder.load_state_dict(torch.load(encoder_parameter_file, map_location=device))

    decoder_parameter_file = (
        PROJECT_ROOT / "data/2024-09-17_04-47-37,195367.ckpt/i_jepa_target_visualization_decoder.pt"
    )
    decoder = IJEPALatentVisualizationDecoder(
        input_n_patches=(12, 12),
        input_latents_dim=32,
        decoder_blocks_in_and_out_channels=[(512, 512), (512, 256), (256, 128), (128, 64)],
        n_res_blocks=3,
        num_heads=4,
    )
    decoder.to(device)
    decoder.load_state_dict(torch.load(decoder_parameter_file, map_location=device))

    forward_dynamics_parameter_file = PROJECT_ROOT / "data/2024-09-17_04-47-37,195367.ckpt/forward_dynamics.pt"
    forward_dynamics = ForwardDynamcisWithActionReward(
        observation_flatten=LerpStackedFeatures(
            dim_in=32,
            dim_out=2048,
            num_stack=144,
        ),
        action_flatten=MultiEmbeddings(
            choices_per_category=ACTION_CHOICES_PER_CATEGORY, embedding_dim=8, do_flatten=True
        ),
        obs_action_projection=torch.nn.Linear(in_features=2088, out_features=2048),
        core_model=SioConv(depth=12, dim=2048, num_head=8, dim_ff_hidden=4096, chunk_size=512, dropout=0.1),
        obs_hat_dist_head=torch.nn.Sequential(
            ToStackedFeatures(dim_in=2048, dim_out=32, num_stack=144),
            FullyConnectedFixedStdNormal(dim_in=32, dim_out=32, normal_cls=DeterministicNormal),
        ),
        action_hat_dist_head=DiscretePolicyHead(dim_in=2048, action_choices_per_category=ACTION_CHOICES_PER_CATEGORY),
        reward_hat_dist_head=FullyConnectedFixedStdNormal(dim_in=2048, dim_out=1, squeeze_feature_dim=True),
    )
    forward_dynamics.to(device)
    forward_dynamics.load_state_dict(torch.load(forward_dynamics_parameter_file, map_location=device))

    handler = KeyboardActionHandler()

    image = torchvision.io.read_image("data/2024-09-17_04-47-37,195367.ckpt/init.png")[:3]  # ignore Alpha channel.
    image = torchvision.transforms.functional.resize(image, 144)
    image = torchvision.transforms.functional.crop(image, 0, image.shape[2] // 2 - 72, 144, 144)
    initial_observation = torch.nn.functional.layer_norm(image.float(), [3, 144, 144]).to(device)
    embedding = encoder(initial_observation).squeeze(0)
    hidden = torch.zeros(12, 2048, device=device)

    while True:
        handler.update()
        action = handler.get_action().to(device)
        # print(action)
        next_embedding_dist, _, _, hidden = forward_dynamics(embedding, hidden, action)
        embedding = next_embedding_dist.rsample()
        reconstruction = decoder(embedding.unsqueeze(0)).squeeze(0)
        reconstruction = (
            (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min() + 1e-8) * 255
        )
        reconstruction = reconstruction.to(torch.uint8)
        reconstruction[torch.tensor([0, 1, 2])] = reconstruction[torch.tensor([2, 1, 0])]
        cv2.imshow("imagination world", reconstruction.permute(1, 2, 0).to("cpu").detach().numpy())
        cv2.waitKey(1)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
