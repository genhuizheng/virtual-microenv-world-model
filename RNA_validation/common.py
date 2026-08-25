from __future__ import annotations

import csv
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
try:
    import torch
except ModuleNotFoundError:  # Allows gene/clinical preparation audits in CPU-only utility environments.
    torch = None  # type: ignore[assignment]


MISSING = {"", "NA", "N/A", "NAN", "NONE", "NULL", "NOT REPORTED", "UNKNOWN", "--"}


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.upper() in MISSING else text


def strip_ensembl_version(value: object) -> str:
    text = clean(value)
    if text.upper().startswith("ENSG"):
        return re.sub(r"\.\d+$", "", text.upper())
    return text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model training or inference")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device: str) -> torch.device:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model training or inference")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def require_empty_output(path: str | Path, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace generated files.")
        resolved = path.resolve()
        if resolved == Path(resolved.anchor) or resolved == Path.cwd().resolve() or len(resolved.parts) < 3:
            raise ValueError(f"Refusing to overwrite unsafe output directory: {resolved}")
        shutil.rmtree(resolved)
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_gene_table(checkpoint: dict, model_genes: str | Path | None = None) -> pd.DataFrame:
    if model_genes is not None:
        table = read_tsv(model_genes)
        if "gene_id" not in table.columns:
            raise ValueError("--model-genes must contain gene_id")
        if "gene_symbol" not in table.columns:
            table["gene_symbol"] = table["gene_id"]
    else:
        ids = checkpoint.get("gene_ids")
        if not ids:
            raise ValueError("Checkpoint has no gene_ids; provide --model-genes genes.tsv")
        symbols = checkpoint.get("gene_symbols")
        if symbols is None or len(symbols) == 0:
            symbols = ids
        if len(symbols) != len(ids):
            raise ValueError("Checkpoint gene_ids and gene_symbols have different lengths")
        table = pd.DataFrame({"gene_id": ids, "gene_symbol": symbols})
    table = table[["gene_id", "gene_symbol"]].copy()
    table["gene_id"] = table["gene_id"].map(strip_ensembl_version)
    table["gene_symbol"] = table["gene_symbol"].map(clean)
    if table["gene_id"].eq("").any() or table["gene_id"].duplicated().any():
        raise ValueError("Model gene order contains blank or duplicated gene IDs")
    table.insert(0, "model_gene_index", np.arange(len(table), dtype=int))
    return table


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to load a world-model checkpoint")
    try:
        obj = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        obj = torch.load(Path(path), map_location=map_location)
    if not isinstance(obj, dict) or "model_state_dict" not in obj:
        raise ValueError(f"Not a supported world-model checkpoint: {path}")
    return obj


def build_world_model_from_checkpoint(checkpoint: dict, device: torch.device):
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to build the world model")
    from five_level_cell_world_model import build_model

    args = checkpoint.get("args", {})
    model = build_model(
        n_genes=int(checkpoint["n_genes"]),
        level=int(args.get("level", 6)),
        variant=str(args.get("variant", "default")),
        latent_dim=int(args.get("latent_dim", 128)),
        expression_hidden_dim=int(args.get("expression_hidden_dim", 1024)),
        expression_depth=int(args.get("expression_depth", 3)),
        representation_type=str(args.get("representation_type", "legacy_mlp")),
        model_dim=int(args.get("model_dim", 256)),
        depth=int(args.get("depth", 4)),
        heads=int(args.get("heads", 8)),
        num_tokens=int(args.get("num_tokens", 8)),
        dropout=float(args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


@dataclass
class AlignmentResult:
    source_to_model: np.ndarray
    method: list[str]
    resolved_model_gene: list[str]
    report: pd.DataFrame


class GeneAligner:
    """Resolve source genes and place them in the checkpoint's exact gene order."""

    def __init__(self, model_gene_table: pd.DataFrame, ranked_reference: str | Path | None = None):
        self.model = model_gene_table.reset_index(drop=True).copy()
        self.id_to_idx = {g: i for i, g in enumerate(self.model["gene_id"])}
        symbol_groups: dict[str, list[int]] = {}
        for i, symbol in enumerate(self.model["gene_symbol"].astype(str)):
            if clean(symbol):
                symbol_groups.setdefault(symbol.upper(), []).append(i)
        self.unique_symbol_to_idx = {s: v[0] for s, v in symbol_groups.items() if len(v) == 1}
        self.reference_symbol_to_id: dict[str, str] = {}
        if ranked_reference:
            ref = read_tsv(ranked_reference)
            required = {"input_symbol", "ensembl_gene_id"}
            if not required.issubset(ref.columns):
                raise ValueError(f"Ranked reference missing columns: {sorted(required - set(ref.columns))}")
            for row in ref.itertuples(index=False):
                symbol = clean(getattr(row, "input_symbol")).upper()
                gene_id = strip_ensembl_version(getattr(row, "ensembl_gene_id"))
                if symbol and gene_id in self.id_to_idx and symbol not in self.reference_symbol_to_id:
                    self.reference_symbol_to_id[symbol] = gene_id

    def resolve(self, gene_ids: Sequence[object], gene_symbols: Sequence[object]) -> AlignmentResult:
        if len(gene_ids) != len(gene_symbols):
            raise ValueError("gene_ids and gene_symbols must have equal length")
        target = np.full(len(gene_ids), -1, dtype=np.int64)
        methods: list[str] = []
        resolved: list[str] = []
        rows = []
        for i, (raw_id, raw_symbol) in enumerate(zip(gene_ids, gene_symbols)):
            gid = strip_ensembl_version(raw_id)
            symbol = clean(raw_symbol).upper()
            idx = -1
            method = "unresolved"
            if gid in self.id_to_idx:
                idx = self.id_to_idx[gid]
                method = "exact_ensembl"
            elif symbol in self.reference_symbol_to_id:
                resolved_id = self.reference_symbol_to_id[symbol]
                idx = self.id_to_idx[resolved_id]
                method = "ranked_symbol_reference"
            elif symbol in self.unique_symbol_to_idx:
                idx = self.unique_symbol_to_idx[symbol]
                method = "unique_checkpoint_symbol"
            target[i] = idx
            model_id = self.model.iloc[idx]["gene_id"] if idx >= 0 else ""
            methods.append(method)
            resolved.append(model_id)
            rows.append({
                "source_row": i,
                "source_gene_id": clean(raw_id),
                "source_gene_symbol": clean(raw_symbol),
                "model_gene_index": idx if idx >= 0 else "NA",
                "model_gene_id": model_id or "NA",
                "mapping_method": method,
            })
        return AlignmentResult(target, methods, resolved, pd.DataFrame(rows))

    def align_gene_by_sample(self, values: np.ndarray, alignment: AlignmentResult) -> np.ndarray:
        """Align [source_genes, samples], summing duplicate source mappings."""
        if values.ndim != 2 or values.shape[0] != len(alignment.source_to_model):
            raise ValueError("Expression shape does not match alignment")
        out = np.zeros((values.shape[1], len(self.model)), dtype=np.float32)
        for source_i, target_i in enumerate(alignment.source_to_model):
            if target_i >= 0:
                out[:, target_i] += values[source_i].astype(np.float32, copy=False)
        return out


def transform_counts(aligned_counts: np.ndarray, library_sizes: np.ndarray, transform: str) -> np.ndarray:
    aligned = np.asarray(aligned_counts, dtype=np.float32)
    library_sizes = np.asarray(library_sizes, dtype=np.float64).reshape(-1)
    if aligned.shape[0] != len(library_sizes):
        raise ValueError("One library size is required per sample")
    if transform == "none":
        return aligned
    denom = np.maximum(library_sizes, 1.0)[:, None]
    cpm = aligned / denom * 1_000_000.0
    if transform == "cpm":
        return cpm.astype(np.float32)
    if transform == "log1p_cpm":
        return np.log1p(cpm).astype(np.float32)
    if transform == "log1p_10k":
        normalized = aligned / denom * 10_000.0
        return np.log1p(normalized).astype(np.float32)
    raise ValueError(f"Unknown transform: {transform}")


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def endpoint_metrics(y_true: np.ndarray, y_pred: np.ndarray, x_source: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(len(y_true)):
        truth = y_true[i]
        pred = y_pred[i]
        source = x_source[i]
        rows.append({
            "row_index": i,
            "mse": float(np.mean((pred - truth) ** 2)),
            "mae": float(np.mean(np.abs(pred - truth))),
            "post_pearson": safe_pearson(pred, truth),
            "delta_pearson": safe_pearson(pred - source, truth - source),
            "predicted_delta_l2": float(np.linalg.norm(pred - source)),
            "observed_delta_l2": float(np.linalg.norm(truth - source)),
        })
    return pd.DataFrame(rows)


def parse_numeric(value: object) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def max_nonnegative(values: Iterable[object]) -> float | None:
    valid = [x for x in (parse_numeric(v) for v in values) if x is not None]
    return max(valid) if valid else None
