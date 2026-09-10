import os
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer


def pre_pull(marker: Path) -> bool:
    if marker.exists():
        return False
    SentenceTransformer("all-MiniLM-L6-v2")
    CrossEncoder("BAAI/bge-reranker-base")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def main() -> None:
    hf_home = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    marker = Path(hf_home) / ".basalt_models_ready"
    pre_pull(marker)


if __name__ == "__main__":
    main()
