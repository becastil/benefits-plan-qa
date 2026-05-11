"""CLI entrypoint.

Usage:
    python main.py path/to/plan.pdf "What is the deductible for family coverage?"
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from qa import answer, build_index


def cli() -> int:
    if len(sys.argv) < 3:
        print("Usage: python main.py <plan.pdf> \"<question>\"", file=sys.stderr)
        return 2

    pdf_path = sys.argv[1]
    question = " ".join(sys.argv[2:])

    load_dotenv()

    print(f"Indexing {pdf_path} ...", file=sys.stderr)
    index = build_index(pdf_path)
    print(f"Indexed {len(index.chunks)} chunks. Asking Claude ...\n", file=sys.stderr)

    reply = answer(question, index)
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
