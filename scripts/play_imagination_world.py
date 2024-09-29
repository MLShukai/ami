import cv2
import keyboard
import hydra
import rootutils
import torch
from torch import Tensor
from omegaconf import DictConfig

@hydra.main(config_path="../data/2024-09-17_04-47-37,195367.ckpt", config_name="launch-configuration.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    encoder_parameter_file = PROJECT_ROOT / "data/2024-09-17_04-47-37,195367.ckpt/i_jepa_target_encoder.pt"
    encoder = hydra.utils.instantiate(cfg.models.i_jepa_target_encoder.model)
    encoder.to(device)
    encoder.load_state_dict(torch.load(encoder_parameter_file, map_location=device))

    decoder_parameter_file = PROJECT_ROOT / "data/2024-09-17_04-47-37,195367.ckpt/i_jepa_target_visualization_decoder.pt"
    decoder = hydra.utils.instantiate(cfg.models.i_jepa_target_visualization_decoder.model)
    decoder.to(device)
    decoder.load_state_dict(torch.load(decoder_parameter_file, map_location=device))

    forward_dynamics_parameter_file = PROJECT_ROOT / "data/2024-09-17_04-47-37,195367.ckpt/forward_dynamics.pt"
    forward_dynamics = hydra.utils.instantiate(cfg.models.forward_dynamics.model)
    forward_dynamics.to(device)
    forward_dynamics.load_state_dict(torch.load(forward_dynamics_parameter_file, map_location=device))

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
            return torch.tensor([
                self.actions["MoveVertical"],
                self.actions["MoveHorizontal"],
                self.actions["LookHorizontal"],
                self.actions["Jump"],
                self.actions["Run"],
            ])

    handler = KeyboardActionHandler()

    initial_observation = torch.randn(cfg.models.i_jepa_target_encoder.model.in_channels, *cfg.models.i_jepa_target_encoder.model.img_size, device=device)
    embedding = encoder(initial_observation).squeeze(0)
    hidden = torch.zeros(cfg.models.forward_dynamics.model.core_model.depth, cfg.models.forward_dynamics.model.core_model.dim, device=device)

    while True:
        handler.update()
        action = handler.get_action().to(device)
        print(action)
        next_embedding_dist, _, _, hidden = forward_dynamics(embedding, hidden, action)
        embedding = next_embedding_dist.rsample()
        reconstruction = decoder(embedding.unsqueeze(0))
        cv2.imshow("imagination world", reconstruction.squeeze(0).permute(1,2,0).to("cpu").detach().numpy())
        cv2.waitKey(1)


if __name__ == "__main__":
    main()