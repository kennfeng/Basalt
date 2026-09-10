import os

from ingest import ensure_seeded
from sample_data import SAMPLE_DOCUMENTS


def main() -> None:
    db_path = os.environ.get("BASALT_DB_PATH", "./basalt_db")
    seeded = ensure_seeded(
        db_path=db_path, texts=[text for _, text in SAMPLE_DOCUMENTS]
    )
    if seeded:
        print(f"Seeded sample documents into {db_path}")


if __name__ == "__main__":
    main()
