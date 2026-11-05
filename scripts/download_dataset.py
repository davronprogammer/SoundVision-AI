"""Script to download and extract the ESC-50 dataset for SoundVision AI."""

import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile
import pandas as pd
from tqdm import tqdm

from soundvision.config import load_config
from soundvision.data.dataset import SELECTED_CLASSES


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_esc50(raw_dir: Path, url: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ESC-50-master.zip"
    extract_target = raw_dir / "ESC-50-master"

    if extract_target.is_dir() and (extract_target / "meta" / "esc50.csv").is_file():
        print(f"ESC-50 dataset already exists at '{extract_target}'.")
        return extract_target

    if not zip_path.is_file():
        print(f"Downloading ESC-50 archive from {url} ...")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="ESC-50 Master Zip") as t:
            urllib.request.urlretrieve(url, filename=str(zip_path), reporthook=t.update_to)
        print("Download complete.")

    print(f"Extracting archive to {raw_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    print(f"Dataset extracted to {extract_target}")
    return extract_target


def verify_esc50_subset(extract_dir: Path, target_classes: list[str]) -> bool:
    meta_path = extract_dir / "meta" / "esc50.csv"
    audio_dir = extract_dir / "audio"

    if not meta_path.is_file():
        print(f"Error: Metadata file missing at {meta_path}")
        return False

    df = pd.read_csv(meta_path)
    subset_df = df[df["category"].isin(target_classes)]
    print(f"Found {len(subset_df)} samples across target {len(target_classes)} classes.")

    missing_files = []
    for filename in subset_df["filename"]:
        if not (audio_dir / filename).is_file():
            missing_files.append(filename)

    if missing_files:
        print(f"Warning: {len(missing_files)} audio files missing from {audio_dir}!")
        return False

    print("All 400 subset audio files successfully verified!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and verify ESC-50 dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg.dataset.raw_dir)
    url = cfg.dataset.repo_url

    extract_dir = download_esc50(raw_dir, url)
    verify_esc50_subset(extract_dir, cfg.dataset.classes)


if __name__ == "__main__":
    main()
