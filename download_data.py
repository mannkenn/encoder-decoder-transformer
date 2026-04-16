import argparse
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError as exc:
    raise SystemExit(
        "Kaggle is not installed. Run 'pip install -r requirements.txt' first."
    ) from exc


DEFAULT_DATASET = "mohamedlotfy50/wmt-2014-english-german"
DEFAULT_OUTPUT_DIR = Path("data")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the WMT 2014 English-German dataset from Kaggle."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug to download (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the dataset archive will be downloaded and extracted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if the output directory already contains data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        args.dataset,
        path=str(output_dir),
        unzip=True,
        force=args.force,
    )

    print(f"Downloaded {args.dataset} to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
