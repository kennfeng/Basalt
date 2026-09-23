import argparse
import json
import os
from pathlib import Path

from ingest import BasaltIngestor


def _write_checkpoint_atomic(checkpoint: Path, offset: int) -> None:
    # save resume position so a retry skips finished bytes
    # write tmp plus fsync then replace for crash-safe updates
    # offset = byte position in source file, fsync = force to disk
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    tmp.write_text(str(offset), encoding="utf-8")
    try:
        with tmp.open("rb") as fh:
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        pass
    tmp.replace(checkpoint)
    try:
        dir_fd = os.open(str(checkpoint.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


def bulk_ingest(
    source: Path,
    batch_size: int = 512,
    checkpoint: Path | None = None,
    resume: bool = False,
    db_path: str | None = None,
) -> int:
    # load a large jsonl file into the vector store resumably
    # stream lines from offset, batch upserts, update checkpoint per batch
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

    ingestor = (
        BasaltIngestor(db_path=db_path) if db_path is not None else BasaltIngestor()
    )

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
                    try:
                        ingestor.collection.upsert(
                            documents=batch_docs, ids=batch_ids, metadatas=None
                        )
                    except Exception:
                        raise
                    total_ingested += len(batch_docs)
                    if checkpoint is not None:
                        _write_checkpoint_atomic(checkpoint, f.tell())
                else:
                    if checkpoint is not None and not checkpoint.exists():
                        _write_checkpoint_atomic(checkpoint, f.tell())
                break
            next_offset = f.tell()
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
                try:
                    ingestor.collection.upsert(
                        documents=batch_docs, ids=batch_ids, metadatas=None
                    )
                except Exception:
                    raise
                total_ingested += len(batch_docs)
                if checkpoint is not None:
                    _write_checkpoint_atomic(checkpoint, next_offset)
                batch_docs = []
                batch_ids = []

    return total_ingested


def main() -> None:
    # parse cli flags and run one bulk load with progress output
    # default checkpoint sits next to the source file for resume
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
        if args.checkpoint is not None
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
