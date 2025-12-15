from __future__ import annotations

import argparse

from .chatgpt_api import ChatGPTClient, ChatGPTClientConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check OpenAI-compatible API connectivity for part2 dataset generation.")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--base_url_env", type=str, default="OPENAI_BASE_URL")
    p.add_argument("--use_response_format_json", type=int, default=1, help="1=true, 0=false")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_output_tokens", type=int, default=64)
    p.add_argument("--max_retries", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = ChatGPTClient(
        ChatGPTClientConfig(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            base_url_env=args.base_url_env,
            use_response_format_json=bool(int(args.use_response_format_json)),
            temperature=float(args.temperature),
            max_output_tokens=int(args.max_output_tokens),
            max_retries=int(args.max_retries),
        )
    )
    client.ping()
    print("OK")


if __name__ == "__main__":
    main()
