from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_PAGE = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/222/"
    "bank%2Bmarketing.zip"
)

CSV_MEMBER = "bank-additional/bank-additional-full.csv"
DOCUMENTATION_MEMBER = "bank-additional/bank-additional-names.txt"

EXPECTED_CSV_SHA256 = (
    "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
)
EXPECTED_DOCUMENTATION_SHA256 = (
    "10045462a4c026f858dffa159370c4270510230c1fc63c6c91ca51bba4b3df26"
)


def calculate_sha256(content: bytes) -> str:
    """Return the SHA-256 checksum of byte content."""
    return hashlib.sha256(content).hexdigest()


def verify_content(
    content: bytes,
    expected_checksum: str,
    file_description: str,
) -> str:
    """Verify downloaded content against its expected checksum."""
    actual_checksum = calculate_sha256(content)

    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Checksum failed for {file_description}.\n"
            f"Expected: {expected_checksum}\n"
            f"Actual:   {actual_checksum}"
        )

    return actual_checksum


def verify_existing_file(
    path: Path,
    expected_checksum: str,
) -> None:
    """Verify that an existing raw file has not changed."""
    actual_checksum = calculate_sha256(path.read_bytes())

    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Existing file failed checksum validation: {path}\n"
            f"Expected: {expected_checksum}\n"
            f"Actual:   {actual_checksum}"
        )


def raw_files_are_ready() -> bool:
    """Return True when previously downloaded files are valid."""
    csv_path = RAW_DIR / "bank-additional-full.csv"
    documentation_path = RAW_DIR / "bank-additional-names.txt"
    metadata_path = RAW_DIR / "source_metadata.json"

    if not all(
        path.exists()
        for path in [csv_path, documentation_path, metadata_path]
    ):
        return False

    verify_existing_file(csv_path, EXPECTED_CSV_SHA256)
    verify_existing_file(
        documentation_path,
        EXPECTED_DOCUMENTATION_SHA256,
    )

    print("Existing raw files passed checksum validation.")
    print(f"Dataset: {csv_path}")
    print(f"Metadata: {metadata_path}")
    return True


def download_archive() -> bytes:
    """Download the official UCI archive."""
    print(f"Downloading from: {DOWNLOAD_URL}")

    response = requests.get(
        DOWNLOAD_URL,
        timeout=60,
        headers={"User-Agent": "before-the-call-project/1.0"},
    )
    response.raise_for_status()

    return response.content


def extract_selected_files(outer_archive: bytes) -> tuple[bytes, bytes]:
    """Extract the selected full dataset from UCI's nested archives."""
    with zipfile.ZipFile(io.BytesIO(outer_archive)) as outer_zip:
        nested_archive = outer_zip.read("bank-additional.zip")

    with zipfile.ZipFile(io.BytesIO(nested_archive)) as nested_zip:
        csv_content = nested_zip.read(CSV_MEMBER)
        documentation_content = nested_zip.read(DOCUMENTATION_MEMBER)

    return csv_content, documentation_content


def main() -> None:
    """Download, verify, extract, and document the raw dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if raw_files_are_ready():
        return

    outer_archive = download_archive()
    archive_checksum = calculate_sha256(outer_archive)

    csv_content, documentation_content = extract_selected_files(
        outer_archive
    )

    csv_checksum = verify_content(
        csv_content,
        EXPECTED_CSV_SHA256,
        "bank-additional-full.csv",
    )
    documentation_checksum = verify_content(
        documentation_content,
        EXPECTED_DOCUMENTATION_SHA256,
        "bank-additional-names.txt",
    )

    csv_path = RAW_DIR / "bank-additional-full.csv"
    documentation_path = RAW_DIR / "bank-additional-names.txt"
    archive_path = RAW_DIR / "bank-marketing.zip"
    metadata_path = RAW_DIR / "source_metadata.json"

    csv_path.write_bytes(csv_content)
    documentation_path.write_bytes(documentation_content)
    archive_path.write_bytes(outer_archive)

    metadata = {
        "dataset_name": "Bank Marketing",
        "uci_dataset_id": 222,
        "dataset_page": DATASET_PAGE,
        "download_url": DOWNLOAD_URL,
        "selected_variant": "bank-additional-full.csv",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY 4.0",
        "archive_sha256": archive_checksum,
        "files": {
            csv_path.name: {
                "sha256": csv_checksum,
                "rows_documented_by_source": 41188,
                "input_features_documented_by_source": 20,
            },
            documentation_path.name: {
                "sha256": documentation_checksum,
            },
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Download and checksum validation completed.")
    print(f"Dataset saved to: {csv_path}")
    print(f"Documentation saved to: {documentation_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()