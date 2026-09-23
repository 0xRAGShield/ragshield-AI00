import os
import sys
import time

import requests

QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "ragshield_test"

SNAPSHOT_PATH = (
    "/qdrant/snapshots/ragshield_test/ragshield_test.snapshot"
)
CHECKSUM_PATH = f"{SNAPSHOT_PATH}.checksum"


def wait_for_qdrant() -> None:
    print("Waiting for Qdrant...")

    for _ in range(60):
        try:
            response = requests.get(
                f"{QDRANT_URL}/collections",
                timeout=5,
            )

            if response.status_code == 200:
                print("Qdrant is ready.")
                return

            print(
                f"Qdrant returned status {response.status_code}."
            )

        except requests.RequestException as exc:
            print(f"Qdrant not ready yet: {exc}")

        time.sleep(2)

    raise SystemExit(
        "Qdrant did not become ready in time."
    )


def collection_exists() -> bool:
    response = requests.get(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
        timeout=10,
    )

    return response.status_code == 200


def wait_for_collection() -> None:
    print(f"Waiting for collection '{COLLECTION_NAME}'...")

    for _ in range(60):
        if collection_exists():
            print(
                f"Collection '{COLLECTION_NAME}' is available."
            )
            return

        time.sleep(2)

    raise SystemExit(
        f"Collection '{COLLECTION_NAME}' was not available "
        "after snapshot restore."
    )


def load_checksum() -> str | None:
    if not os.path.isfile(CHECKSUM_PATH):
        return None

    with open(
        CHECKSUM_PATH,
        encoding="utf-8",
    ) as checksum_file:
        checksum = checksum_file.read().strip()

    return checksum or None


def restore_snapshot() -> None:
    if not os.path.isfile(SNAPSHOT_PATH):
        raise SystemExit(
            f"Snapshot file not found: {SNAPSHOT_PATH}"
        )

    print(f"Collection '{COLLECTION_NAME}' does not exist.")
    print("Restoring snapshot...")

    payload: dict[str, str] = {
        "location": f"file://{SNAPSHOT_PATH}",
    }

    checksum = load_checksum()

    if checksum:
        payload["checksum"] = checksum

    response = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots/recover",
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    wait_for_collection()

    print("Snapshot restored successfully.")


def main() -> None:
    wait_for_qdrant()

    if collection_exists():
        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )
        print("Skipping snapshot restore.")
        return

    restore_snapshot()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"qdrant-init failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
