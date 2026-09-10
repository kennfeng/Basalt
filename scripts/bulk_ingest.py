import argparse
import json
from pathlib import Path

from ingest import BasaltIngestor


def _write_checkpoint_atomic(checkpoint: Path, offset: int) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    tmp.write_text(str(offset), encoding="utf-8")
    tmp.replace(checkpoint)


def bulk_ingest(
    source: Path,
    batch_size: int = 512,
    checkpoint: Path | None = None,
    resume: bool = False,
    db_path: str | None = None,
) -> int:
    source = Path(source)
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    start_offset = 0
    if resume and checkpoint is not None and checkpoint.exists():
        try:
            start_offset = int(checkpoint.read_text(encoding="utf-8").strip())
        except ValueError:
            start_offset = 0

    ingestor = BasaltIngestor(db_path=db_path) if db_path else BasaltIngestor()

    total_ingested = 0

    with source.open("rb") as f:
        f.seek(start_offset)
        batch_docs: list[str] = []
        batch_ids: list[str] = []
        while True:
            line_start = f.tell()
            line = f.readline()
            if not line:
                if batch_docs:
                    pos_after = f.tell()
                    if checkpoint is not None:
                        _write_checkpoint_atomic(checkpoint, pos_after)
                    try:
                        ingestor.collection.upsert(
                            documents=batch_docs, ids=batch_ids, metadatas=None
                        )
                    except Exception:
                        if checkpoint is not None:
                            _write_checkpoint_atomic(checkpoint, pos_after)
                        raise
                    total_ingested += len(batch_docs)
                    if checkpoint is not None:
                        _write_checkpoint_atomic(checkpoint, pos_after)
                else:
                    if checkpoint is not None and not checkpoint.exists():
                        _write_checkpoint_atomic(checkpoint, f.tell())
                break
            try:
                obj = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            doc_id = str(obj.get("id") or obj.get("doc_id") or f"line_{line_start}")
            abstract = (
                obj.get("abstract")
                or obj.get("text")
                or obj.get("document")
                or obj.get("content")
                or ""
            )
            if not isinstance(abstract, str):
                abstract = str(abstract)
            batch_docs.append(abstract)
            batch_ids.append(doc_id)
            if len(batch_docs) >= batch_size:
                pos_after = f.tell()
                if checkpoint is not None:
                    _write_checkpoint_atomic(checkpoint, pos_after)
                try:
                    ingestor.collection.upsert(
                        documents=batch_docs, ids=batch_ids, metadatas=None
                    )
                except Exception:
                    if checkpoint is not None:
                        _write_checkpoint_atomic(checkpoint, pos_after)
                    raise
                total_ingested += len(batch_docs)
                if checkpoint is not None:
                    _write_checkpoint_atomic(checkpoint, pos_after)
                batch_docs = []
                batch_ids = []

    return total_ingested


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk ingest JSONL corpus with checkpoint"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Path to JSONL corpus"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()
    source = Path(args.source)
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else source.with_suffix(source.suffix + ".checkpoint")
    )
    count = bulk_ingest(
        source=source,
        batch_size=args.batch_size,
        checkpoint=checkpoint,
        resume=args.resume,
        db_path=args.db_path,
    )
    print(f"Ingested {count} documents from {source}")


if __name__ == "__main__":
    main()
