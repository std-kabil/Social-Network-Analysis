import gzip
import shutil
from pathlib import Path

import requests


EDGES_URL = "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"
README_URL = "https://snap.stanford.edu/data/soc-pokec-readme.txt"


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Already exists, skipping download: {dest_path}")
        return

    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = 100.0 * downloaded / total
                    print(f"\r  {pct:5.1f}%", end="", flush=True)

    if total > 0:
        print()
    print(f"Saved: {dest_path}")


def extract_gzip(gz_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Already exists, skipping extract: {out_path}")
        return

    print(f"Extracting {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Extracted: {out_path}")


def main() -> None:
    data_dir = Path("data")

    gz_edges = data_dir / "soc-pokec-relationships.txt.gz"
    txt_edges = data_dir / "soc-pokec-relationships.txt"

    download_file(EDGES_URL, gz_edges)
    extract_gzip(gz_edges, txt_edges)

    readme_path = data_dir / "soc-pokec-readme.txt"
    download_file(README_URL, readme_path)

    print("\nDone.")
    print(f"Edges:  {txt_edges}")
    print(f"Readme: {readme_path}")


if __name__ == "__main__":
    main()
