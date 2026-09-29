"""The stratification variable must be computable without the model."""
import ast
from pathlib import Path

FORBIDDEN = ("neptune.model", "neptune.state", "neptune.dynamics", "neptune.heads",
             "neptune.training", "neptune.content.encoder", "neptune.baselines")
PKG = Path(__file__).resolve().parents[1] / "src" / "neptune"


def _imports(path: Path):
    pkg = ["neptune"] + list(path.relative_to(PKG).parent.parts)
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                yield node.module or ""
            else:
                base = pkg[: len(pkg) - (node.level - 1)]
                yield ".".join(base + ([node.module] if node.module else []))


def test_semantics_never_imports_the_model():
    for f in (PKG / "semantics").glob("*.py"):
        for name in _imports(f):
            assert not name.startswith(FORBIDDEN), f"{f.name} imports {name}"


def test_covariate_pipeline_never_imports_the_model():
    for f in (PKG / "pipeline" / "covariates.py", PKG / "pipeline" / "prepare.py",
              PKG / "content" / "raw_features.py", PKG / "common.py"):
        for name in _imports(f):
            assert not name.startswith(FORBIDDEN), f"{f.name} imports {name}"


def test_model_is_not_initialized_from_semantic_clusters():
    src = (PKG / "model.py").read_text()
    assert "centroid" not in src.split('"""', 2)[2] and "item_cluster" not in src
