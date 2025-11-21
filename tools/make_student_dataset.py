#!/usr/bin/env python3
import argparse
import glob
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Build (edge, teacher) pairs from GAN-Compression compressed outputs."
    )
    parser.add_argument(
        "--gan-root",
        type=str,
        default="/home/kshitij/proj/gan-compression",
        help="Path to gan-compression repo",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="results-pretrained/pix2pix/edges2shoes-r/compressed/images",
        help="Relative path under gan-root where pix2pix outputs are stored",
    )
    parser.add_argument(
        "--edge-subdir",
        type=str,
        default="input_A",
        help="Subdirectory name (under images-root) for edge/input images",
    )
    parser.add_argument(
        "--teacher-subdir",
        type=str,
        default="fake_B",
        help="Subdirectory name (under images-root) for teacher (fake_B) images",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/edges2shoes_student",
        help="Output dataset root under this repo",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of pairs to copy",
    )
    args = parser.parse_args()

    # Resolve source dirs
    images_root = os.path.join(args.gan_root, args.images_root)
    edge_dir = os.path.join(images_root, args.edge_subdir)
    teacher_dir = os.path.join(images_root, args.teacher_subdir)

    print(f"Using teacher_dir: {teacher_dir}")
    print(f"Using edge_dir   : {edge_dir}")

    if not os.path.isdir(teacher_dir):
        raise SystemExit(f"Teacher dir not found: {teacher_dir}")
    if not os.path.isdir(edge_dir):
        raise SystemExit(f"Edge dir not found: {edge_dir}")

    # Output dirs
    out_root = args.out_root
    a_dir = os.path.join(out_root, "A")
    b_dir = os.path.join(out_root, "B")
    os.makedirs(a_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)

    teacher_files = sorted(glob.glob(os.path.join(teacher_dir, "*.png")))
    if not teacher_files:
        raise SystemExit(f"No .png files found in {teacher_dir}")

    num_copied = 0
    for t_path in teacher_files:
        t_name = os.path.basename(t_path)

        # First try exact same filename in edge dir
        candidate_edge = os.path.join(edge_dir, t_name)
        if os.path.exists(candidate_edge):
            edge_path = candidate_edge
        else:
            # Try some simple pattern fixes, e.g. 0_AB.png -> 0_A.png or 0_B.png
            stem, ext = os.path.splitext(t_name)
            edge_path = None
            if "_AB" in stem:
                base = stem.split("_AB")[0]
                for suffix in ["_A", "_B", ""]:
                    alt_name = base + suffix + ext
                    alt_path = os.path.join(edge_dir, alt_name)
                    if os.path.exists(alt_path):
                        edge_path = alt_path
                        break

        if edge_path is None or not os.path.exists(edge_path):
            print(f"Skipping {t_name}: no matching edge file in {edge_dir}")
            continue

        # Use a clean id for output name
        out_id = os.path.splitext(t_name)[0]
        out_edge = os.path.join(a_dir, out_id + ".png")
        out_shoe = os.path.join(b_dir, out_id + ".png")

        shutil.copy2(edge_path, out_edge)
        shutil.copy2(t_path, out_shoe)
        num_copied += 1

        if num_copied % 50 == 0:
            print(f"Copied {num_copied} pairs...")

        if num_copied >= args.max_samples:
            break

    print(f"Done. Copied {num_copied} (edge, teacher) pairs to {out_root}")


if __name__ == "__main__":
    main()
