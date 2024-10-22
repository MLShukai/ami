import argparse

import torch

from ami.trainers.components.bool_i_jepa_mask_collator import (
    BoolIJEPAMultiBlockMaskCollator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the ratio of the masked regions output from BoolIJEPAMultiBlockMaskCollator by simulation."
    )
    parser.add_argument("--image_size", type=int, nargs=2, default=[144, 144], help="Image size (height, width)")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[12, 12], help="Patch size (height, width)")
    parser.add_argument("--mask_scale", type=float, nargs=2, default=[0.10, 0.25], help="Mask scale range")
    parser.add_argument("--n_masks", type=int, default=4, help="Number of mask candidates")
    parser.add_argument("--aspect_ratio", type=float, nargs=2, default=[0.75, 1.5], help="Aspect ratio range")
    parser.add_argument("--min_keep", type=int, default=10, help="Minimum number of patches to keep unmasked")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples for simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    collator = BoolIJEPAMultiBlockMaskCollator(
        input_size=args.image_size,
        patch_size=args.patch_size,
        mask_scale=args.mask_scale,
        n_masks=args.n_masks,
        aspect_ratio=args.aspect_ratio,
        min_keep=args.min_keep,
    )

    encoder_mask_ratios = []
    predictor_target_ratios = []

    for _ in range(args.num_samples):
        encoder_mask, predictor_target = collator.sample_masks_and_target(g)
        encoder_mask_ratios.append(encoder_mask.float().mean().item())
        predictor_target_ratios.append(predictor_target.float().mean().item())

    avg_encoder_mask_ratio = sum(encoder_mask_ratios) / len(encoder_mask_ratios)
    avg_predictor_target_ratio = sum(predictor_target_ratios) / len(predictor_target_ratios)

    print(f"Average encoder mask region: {avg_encoder_mask_ratio:.4f}")
    print(f"Average predictor target region: {avg_predictor_target_ratio:.4f}")

    # Additional statistics
    encoder_mask_tensor = torch.tensor(encoder_mask_ratios)
    predictor_target_tensor = torch.tensor(predictor_target_ratios)

    print("\nEncoder mask statistics:")
    print(f"Min: {encoder_mask_tensor.min().item():.4f}")
    print(f"Max: {encoder_mask_tensor.max().item():.4f}")
    print(f"Median: {encoder_mask_tensor.median().item():.4f}")
    print(f"Std Dev: {encoder_mask_tensor.std().item():.4f}")

    print("\nPredictor target statistics:")
    print(f"Min: {predictor_target_tensor.min().item():.4f}")
    print(f"Max: {predictor_target_tensor.max().item():.4f}")
    print(f"Median: {predictor_target_tensor.median().item():.4f}")
    print(f"Std Dev: {predictor_target_tensor.std().item():.4f}")


if __name__ == "__main__":
    main()
