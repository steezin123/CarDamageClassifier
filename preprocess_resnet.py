import argparse
from pathlib import Path

from preprocess_car_dd import dump_numpy_bundle


def parse_args():
    parser = argparse.ArgumentParser("CAR-DD ResNet Preprocessor")
    parser.add_argument("--images_dir", required=True, help="Path to images root")
    parser.add_argument("--annotations", required=True, help="Path to COCO-style JSON annotations")
    parser.add_argument("--out_dir", required=True, help="Where to write numpy bundle outputs")
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Square letterbox size (typical ResNet input). Use 0 to keep original size.",
    )
    parser.add_argument(
        "--no_letterbox",
        action="store_true",
        help="Disable resizing/letterboxing; keep original resolution.",
    )
    parser.add_argument(
        "--build_semantic",
        action="store_true",
        help="Also build per-pixel category map (not usually needed for classification).",
    )
    parser.add_argument(
        "--save_flat_vectors",
        action="store_true",
        help="Optionally save flattened pixel vectors alongside images.npy.",
    )
    parser.add_argument(
        "--skip_unlabeled",
        action="store_true",
        help="Drop images that have no annotations (recommended for classification).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    img_size = None if args.no_letterbox or args.img_size == 0 else args.img_size
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    dump_numpy_bundle(
        images_dir=args.images_dir,
        annotations_path=args.annotations,
        out_dir=args.out_dir,
        img_size=img_size,
        build_semantic=args.build_semantic,
        save_flat_vectors=args.save_flat_vectors,
        skip_unlabeled=args.skip_unlabeled,
    )

    print("ResNet preprocessing complete.")
    print(f"- Bundle written to: {args.out_dir}")
    print("  Files: images.npy, instance_masks.npz, targets.json, image_ids.npy")
    if args.build_semantic:
        print("  Also wrote: semantic_masks.npy")
    if args.save_flat_vectors:
        print("  Also wrote: flat_vectors.npy")


if __name__ == "__main__":
    main()
