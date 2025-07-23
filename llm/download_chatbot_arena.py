import pandas as pd
from argparse import ArgumentParser
import requests
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Where to save the data",
        default=root / "data/llm/chatbot_arena",
    )
    args = parser.parse_args()
    download_chatbot_arena(args)


def download_chatbot_arena(args):
    """Download the Chatbot Arena data."""
    file_id = "1ow-HUNz2pdxLk-Twe30-qJJbx1R6epBz"
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "arena_hard.json"

    if response.status_code == 200:
        with open(path, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Failed to download the Arena file.")

    arena_hard = pd.read_json(path, lines=True)
    print(arena_hard.head())
    print(f"Arena Hard: {len(arena_hard) : >7} models")


if __name__ == "__main__":
    main()
