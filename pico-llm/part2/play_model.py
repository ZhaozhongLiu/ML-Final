from __future__ import annotations

import argparse
from pathlib import Path

from .pico_module import load_checkpoint_model, pick_device
from .tokenization import get_encoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play with a trained pico-llm checkpoint (one-shot or REPL).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint (e.g., transformer_dpo.pt).")
    p.add_argument("--pico_llm_py", type=str, default=str(Path(__file__).resolve().parents[1] / "pico-llm.py"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling p; set empty/<=0 for greedy.")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--mode", choices=["oneshot", "repl"], default="repl")
    p.add_argument("--prompt", type=str, default=None, help="Used in oneshot mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    enc = get_encoder()
    loaded = load_checkpoint_model(args.pico_llm_py, args.checkpoint, device)
    model = loaded.model

    top_p = None if args.top_p is None or args.top_p <= 0 else float(args.top_p)

    def _postprocess_generation(prompt: str, full_text: str) -> str:
        text = full_text
        if prompt and text.startswith(prompt):
            text = text[len(prompt) :]
        if "<|endoftext|>" in text:
            text = text.split("<|endoftext|>", 1)[0]
        return text.strip()

    if args.mode == "oneshot":
        if not args.prompt:
            raise SystemExit("--prompt is required in oneshot mode.")
        text, _ann = loaded.module.generate_text(
            model,
            enc,
            args.prompt,
            max_new_tokens=int(args.max_new_tokens),
            device=device,
            top_p=top_p,
        )
        print(_postprocess_generation(args.prompt, text))
        return

    # REPL mode
    print(f"[play] loaded={args.checkpoint} device={device} top_p={top_p} max_new_tokens={args.max_new_tokens}")
    print("[play] Enter a prompt (finish with Enter). Type /exit to quit.\n")
    while True:
        try:
            prompt = input("prompt> ")
        except EOFError:
            break
        prompt = prompt.strip("\n")
        if not prompt:
            continue
        if prompt.strip() in {"/exit", "exit", "quit"}:
            break
        text, _ann = loaded.module.generate_text(
            model,
            enc,
            prompt,
            max_new_tokens=int(args.max_new_tokens),
            device=device,
            top_p=top_p,
        )
        print("\n--- generation ---")
        print(_postprocess_generation(prompt, text))
        print("---------------\n")


if __name__ == "__main__":
    main()
