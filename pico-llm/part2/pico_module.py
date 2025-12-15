from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def load_pico_llm_module(pico_llm_py: str | Path) -> Any:
    pico_llm_py = Path(pico_llm_py)
    if not pico_llm_py.exists():
        raise FileNotFoundError(f"Cannot locate pico-llm training script at {pico_llm_py}")
    spec = importlib.util.spec_from_file_location("pico_llm_module", pico_llm_py)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class LoadedModel:
    module: Any
    model: torch.nn.Module
    model_type: str
    config: Dict[str, Any]


def build_model_from_checkpoint(module: Any, checkpoint: Dict[str, Any], device: torch.device) -> LoadedModel:
    config = checkpoint.get("config", {}) or {}
    model_type = checkpoint.get("model_type") or config.get("model_type")
    if not model_type:
        raise ValueError("Checkpoint missing 'model_type' metadata.")

    if model_type == "kgram_mlp_seq":
        model = module.KGramMLPSeqModel(
            vocab_size=config["vocab_size"],
            k=config["kgram_k"],
            embed_size=config["embed_size"],
            num_inner_layers=config.get("num_inner_layers", 1),
            chunk_size=config.get("chunk_size", 1),
        )
    elif model_type == "lstm_seq":
        model = module.LSTMSeqModel(
            vocab_size=config["vocab_size"],
            embed_size=config["embed_size"],
            hidden_size=config.get("hidden_size", config["embed_size"]),
        )
    elif model_type == "transformer":
        model = module.TransformerModel(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_blocks=config["n_blocks"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            dropout=config.get("dropout", 0.0),
            max_seq_len=config.get("max_seq_len", 1024),
            positional_embedding=config.get("positional_embedding", "learned"),
            rope_base=config.get("rope_base", 10000.0),
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' in checkpoint.")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return LoadedModel(module=module, model=model, model_type=model_type, config=config)


def load_checkpoint_model(pico_llm_py: str | Path, checkpoint_path: str | Path, device: torch.device) -> LoadedModel:
    module = load_pico_llm_module(pico_llm_py)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return build_model_from_checkpoint(module, checkpoint, device)


def pick_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)

