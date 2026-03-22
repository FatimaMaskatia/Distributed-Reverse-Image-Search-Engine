from __future__ import annotations

import argparse
from pathlib import Path

from deep_features import ExtractionConfig, extract_deep_features, save_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 1: Deep CNN feature extraction pipeline"
    )
    parser.add_argument("--image-dir", type=Path, required=True, help="Input image folder")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExtractionConfig(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model_name,
        image_size=args.image_size,
    )

    features, image_paths = extract_deep_features(config)
    save_outputs(config.output_dir, features, image_paths)

    print(f"Extracted vectors: {features.shape[0]}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Saved: {config.output_dir / 'features.npy'}")
    print(f"Saved: {config.output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
