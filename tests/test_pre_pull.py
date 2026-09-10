import os
from unittest.mock import patch

from scripts import pre_pull


def test_pre_pull_skips_when_marker_exists(tmp_path):
    marker = tmp_path / ".basalt_models_ready"
    marker.write_text("ready")
    with (
        patch("scripts.pre_pull.SentenceTransformer") as mock_st,
        patch("scripts.pre_pull.CrossEncoder") as mock_ce,
    ):
        result = pre_pull.pre_pull(marker)
    assert result is False
    mock_st.assert_not_called()
    mock_ce.assert_not_called()


def test_pre_pull_loads_models_and_writes_marker(tmp_path):
    marker = tmp_path / ".basalt_models_ready"
    with (
        patch("scripts.pre_pull.SentenceTransformer") as mock_st,
        patch("scripts.pre_pull.CrossEncoder") as mock_ce,
    ):
        result = pre_pull.pre_pull(marker)
    assert result is True
    assert marker.exists()
    mock_st.assert_called_once_with("all-MiniLM-L6-v2")
    mock_ce.assert_called_once_with("BAAI/bge-reranker-base")


def test_main_uses_hf_home_env(tmp_path):
    hf_home = tmp_path / "hf"
    with (
        patch.dict(os.environ, {"HF_HOME": str(hf_home)}),
        patch("scripts.pre_pull.SentenceTransformer"),
        patch("scripts.pre_pull.CrossEncoder"),
    ):
        pre_pull.main()
    assert (hf_home / ".basalt_models_ready").exists()
