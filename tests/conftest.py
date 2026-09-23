import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = False
torch_mock.__spec__ = types.ModuleType("torch", None)
sys.modules["torch"] = torch_mock
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.utils"] = MagicMock()
