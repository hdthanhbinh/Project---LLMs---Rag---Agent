from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.processor import load_document, split_text


CHUNK_SIZES = [500, 1000, 1500, 2000]
CHUNK_OVERLAPS = [50, 100, 200]


def summarize_chunks(files: list[str], chunk_size: int, chunk_overlap: int) -> dict:
    start = time.perf_counter()
    docs = load_document(files)
    chunks = split_text(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elapsed = time.perf_counter() - start

    lengths = [len(chunk.page_content) for chunk in chunks]
    sources = sorted({chunk.metadata.get("source", "unknown") for chunk in chunks})

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunks),
        "avg_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "seconds": round(elapsed, 3),
        "sources": ", ".join(sources),
    }


def build_report(rows: list[dict], files: list[str]) -> str:
    lines = [
        "# Chunk Strategy Evaluation",
        "",
        "## Input files",
        "",
    ]
    lines.extend(f"- `{Path(file).name}`" for file in files)
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| chunk_size | chunk_overlap | chunks | avg chars | min chars | max chars | seconds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['chunk_size']} | {row['chunk_overlap']} | "
            f"{row['chunk_count']} | {row['avg_chars']} | "
            f"{row['min_chars']} | {row['max_chars']} | {row['seconds']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Fewer, larger chunks usually reduce retrieval/index overhead but may include more unrelated context.",
            "- Smaller chunks can improve precision, but may split relevant context across many chunks.",
            "- Use this report together with manual QA questions to choose the best setting for the dataset.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate chunk_size/chunk_overlap combinations for uploaded documents."
    )
    parser.add_argument("files", nargs="+", help="PDF/DOCX files to evaluate.")
    parser.add_argument(
        "--output",
        default="documentation/chunk_strategy_report.md",
        help="Markdown report path.",
    )
    args = parser.parse_args()

    rows = []
    for chunk_size in CHUNK_SIZES:
        for chunk_overlap in CHUNK_OVERLAPS:
            if chunk_overlap < chunk_size:
                rows.append(summarize_chunks(args.files, chunk_size, chunk_overlap))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(rows, args.files), encoding="utf-8")
    print(f"Wrote chunk strategy report to {output}")


if __name__ == "__main__":
    main()
