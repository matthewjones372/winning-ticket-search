"""Every Python block in the README runs, in order, against small stand-in data."""

import re
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lottery import ClassificationTrainer
from lottery.models import LeNet300100

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def _stand_ins() -> dict[str, object]:
    """The names a reader brings: their data, model and trainer."""
    data = TensorDataset(torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,)))
    loader = DataLoader(data, batch_size=16)
    return {
        "train_loader": loader,
        "val_loader": loader,
        "test_loader": loader,
        "device": torch.device("cpu"),
        "loss_fn": nn.CrossEntropyLoss(),
        "model": LeNet300100(),
        "MyModel": LeNet300100,
        "trainer": ClassificationTrainer(nn.CrossEntropyLoss(), loader, loader),
    }


def test_the_readme_has_python_blocks():
    assert len(BLOCKS) >= 10


def test_readme_python_blocks_run(tmp_path, monkeypatch):
    pytest.importorskip("torchao")
    pytest.importorskip("accelerate")
    monkeypatch.chdir(tmp_path)  # blocks write checkpoints and CSVs to relative paths
    torch.manual_seed(0)
    defined: dict[str, object] = {}
    for index, block in enumerate(BLOCKS):
        # Each block gets fresh stand-ins (an unpruned model, say), plus whatever the
        # blocks before it defined, such as `ticket` and `result`.
        namespace = {**defined, **_stand_ins()}
        try:
            exec(compile(block, f"README block {index}", "exec"), namespace)
        except Exception as exc:
            first_line = block.strip().splitlines()[0]
            pytest.fail(f"README block {index} ({first_line!r}) raised {exc!r}")
        stand_ins = _stand_ins()
        defined.update({k: v for k, v in namespace.items() if k not in stand_ins})
