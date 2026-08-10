#!/usr/bin/env python3
"""Build an image-by-image HTML comparison and a machine-readable pairing report.

Candidate images are matched to source images by filename stem.  A terminal
``_rectified`` suffix is ignored for candidates, so both ``page.jpg`` and
``page_rectified.png`` match source image ``page.jpg``.  The source and
candidate directories are never modified.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote


IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
)
RECTIFIED_SUFFIX = "_rectified"


def _directory(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


def _source_index(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _image_files(directory):
        key = path.stem
        if key in index:
            raise ValueError(
                f"duplicate source basename {key!r}: {index[key]} and {path}"
            )
        index[key] = path
    if not index:
        raise ValueError(f"source directory contains no supported images: {directory}")
    return index


def _candidate_key(stem: str, source_keys: set[str]) -> str:
    # Prefer an exact basename.  This keeps a legitimate source named
    # ``foo_rectified`` addressable while still mapping the normal generated
    # output ``foo_rectified.png`` back to source ``foo.jpg``.
    if stem in source_keys:
        return stem
    if stem.endswith(RECTIFIED_SUFFIX):
        stripped = stem[: -len(RECTIFIED_SUFFIX)]
        if stripped:
            return stripped
    return stem


def _candidate_index(directory: Path, source_keys: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _image_files(directory):
        key = _candidate_key(path.stem, source_keys)
        if key in index:
            raise ValueError(
                f"duplicate candidate basename {key!r}: {index[key]} and {path}"
            )
        index[key] = path
    return index


def parse_candidate(value: str) -> tuple[str, Path]:
    """Parse one CLI ``NAME=DIR`` candidate specification."""

    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"candidate must use NAME=DIR syntax, got {value!r}"
        )
    name, raw_directory = value.split("=", 1)
    name = name.strip()
    raw_directory = raw_directory.strip()
    if not name:
        raise argparse.ArgumentTypeError("candidate name must not be empty")
    if not raw_directory:
        raise argparse.ArgumentTypeError(
            f"candidate directory must not be empty for {name!r}"
        )
    return name, Path(raw_directory)


def _relative_href(path: Path, html_directory: Path) -> str:
    relative = Path(os.path.relpath(path, start=html_directory)).as_posix()
    return quote(relative, safe="/")


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace one report atomically without leaving a partial final file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _image_cell(label: str, path: Path | None, html_directory: Path) -> str:
    escaped_label = html.escape(label)
    if path is None:
        return (
            '<figure class="image-cell missing">'
            f"<figcaption>{escaped_label}</figcaption>"
            '<div class="missing-message">缺失 / missing</div>'
            "</figure>"
        )
    href = html.escape(_relative_href(path, html_directory), quote=True)
    filename = html.escape(path.name)
    alt = html.escape(f"{label}: {path.name}", quote=True)
    return (
        '<figure class="image-cell">'
        f"<figcaption>{escaped_label}</figcaption>"
        f'<a href="{href}" target="_blank" rel="noopener">'
        f'<img src="{href}" alt="{alt}" loading="lazy" decoding="async">'
        "</a>"
        f'<div class="filename">{filename}</div>'
        "</figure>"
    )


def _render_html(
    report: dict[str, Any],
    *,
    output_html: Path,
    title: str,
    thumbnail_width: int,
) -> str:
    candidate_order = list(report["candidate_order"])
    column_count = 1 + len(candidate_order)
    summary_rows = []
    for name in candidate_order:
        counts = report["candidates"][name]
        summary_rows.append(
            "<tr>"
            f"<th>{html.escape(name)}</th>"
            f"<td>{counts['matched_count']}</td>"
            f"<td>{counts['missing_count']}</td>"
            f"<td>{counts['extra_count']}</td>"
            "</tr>"
        )

    image_rows = []
    html_directory = output_html.parent
    for row in report["rows"]:
        cells = [_image_cell("source", Path(row["source"]), html_directory)]
        cells.extend(
            _image_cell(
                name,
                Path(row["candidates"][name])
                if row["candidates"][name] is not None
                else None,
                html_directory,
            )
            for name in candidate_order
        )
        image_rows.append(
            '<section class="comparison-row">'
            f"<h2>{html.escape(row['key'])}</h2>"
            f'<div class="image-strip">{"".join(cells)}</div>'
            "</section>"
        )

    escaped_title = html.escape(title)
    pairing = report["pairing"]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; padding: 1.25rem; background: Canvas; color: CanvasText; }}
    h1 {{ margin-top: 0; }}
    .summary {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
    .summary th, .summary td {{ border: 1px solid #8888; padding: .35rem .7rem; text-align: right; }}
    .summary th:first-child {{ text-align: left; }}
    .comparison-row {{ border-top: 1px solid #8888; padding: 1rem 0 1.4rem; }}
    .comparison-row h2 {{ font: 600 1rem/1.3 ui-monospace, monospace; overflow-wrap: anywhere; }}
    .image-strip {{
      display: grid;
      grid-template-columns: repeat({column_count}, minmax({thumbnail_width}px, 1fr));
      gap: .8rem;
      overflow-x: auto;
      align-items: start;
    }}
    .image-cell {{ margin: 0; min-width: {thumbnail_width}px; }}
    .image-cell figcaption {{ font-weight: 700; margin-bottom: .35rem; }}
    .image-cell img {{
      display: block; width: 100%; max-height: 420px; object-fit: contain;
      background: #222; border: 1px solid #8888;
    }}
    .filename {{ margin-top: .3rem; font: .78rem/1.25 ui-monospace, monospace; overflow-wrap: anywhere; }}
    .missing {{ border: 2px dashed #c33; min-height: 9rem; padding: .65rem; box-sizing: border-box; }}
    .missing-message {{ color: #c33; font-weight: 700; padding-top: 2.5rem; text-align: center; }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <p>源图 {pairing['source_count']} 张；全部候选齐全 {pairing['complete_row_count']} 张；
     至少一个候选存在 {pairing['rows_with_any_candidate']} 张。点击缩略图打开原始文件。</p>
  <table class="summary">
    <thead><tr><th>候选</th><th>配对</th><th>缺失</th><th>多余</th></tr></thead>
    <tbody>{''.join(summary_rows)}</tbody>
  </table>
  {''.join(image_rows)}
</body>
</html>
"""


def generate_comparison(
    source_directory: str | Path,
    candidates: list[tuple[str, str | Path]],
    output_html: str | Path,
    *,
    output_json: str | Path | None = None,
    title: str = "Typical rectification comparison",
    thumbnail_width: int = 320,
) -> dict[str, Any]:
    """Generate the comparison artifacts and return the JSON report payload."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    if thumbnail_width < 64:
        raise ValueError("thumbnail_width must be at least 64 pixels")

    source_root = _directory(source_directory, label="source")
    source = _source_index(source_root)
    source_keys = set(source)

    candidate_order: list[str] = []
    candidate_roots: dict[str, Path] = {}
    candidate_images: dict[str, dict[str, Path]] = {}
    for name, raw_directory in candidates:
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("candidate name must not be empty")
        if normalized_name in candidate_roots:
            raise ValueError(f"duplicate candidate name: {normalized_name!r}")
        directory = _directory(raw_directory, label=f"candidate {normalized_name!r}")
        candidate_order.append(normalized_name)
        candidate_roots[normalized_name] = directory
        candidate_images[normalized_name] = _candidate_index(directory, source_keys)

    rows: list[dict[str, Any]] = []
    for key in sorted(source, key=lambda value: (value.casefold(), value)):
        rows.append(
            {
                "key": key,
                "source": str(source[key]),
                "candidates": {
                    name: (
                        str(candidate_images[name][key])
                        if key in candidate_images[name]
                        else None
                    )
                    for name in candidate_order
                },
            }
        )

    candidate_report: dict[str, Any] = {}
    for name in candidate_order:
        keys = set(candidate_images[name])
        matched = sorted(source_keys & keys, key=lambda value: (value.casefold(), value))
        missing = sorted(source_keys - keys, key=lambda value: (value.casefold(), value))
        extra = sorted(keys - source_keys, key=lambda value: (value.casefold(), value))
        candidate_report[name] = {
            "directory": str(candidate_roots[name]),
            "image_count": len(keys),
            "matched_count": len(matched),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing_keys": missing,
            "extra_keys": extra,
        }

    complete_rows = sum(
        all(row["candidates"][name] is not None for name in candidate_order)
        for row in rows
    )
    any_rows = sum(
        any(row["candidates"][name] is not None for name in candidate_order)
        for row in rows
    )
    total_pairings = sum(
        candidate_report[name]["matched_count"] for name in candidate_order
    )

    html_path = Path(output_html).expanduser().resolve()
    json_path = (
        Path(output_json).expanduser().resolve()
        if output_json is not None
        else html_path.with_suffix(".json")
    )
    if html_path == json_path:
        raise ValueError("HTML and JSON outputs must use different paths")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "source": {"directory": str(source_root), "count": len(source)},
        "candidate_order": candidate_order,
        "candidates": candidate_report,
        "pairing": {
            "source_count": len(source),
            "candidate_count": len(candidate_order),
            "total_pairings": total_pairings,
            "complete_row_count": complete_rows,
            "rows_with_any_candidate": any_rows,
        },
        "rows": rows,
        "output_html": str(html_path),
        "output_json": str(json_path),
    }
    rendered = _render_html(
        report,
        output_html=html_path,
        title=title,
        thumbnail_width=int(thumbnail_width),
    )
    _atomic_write_text(html_path, rendered)
    _atomic_write_text(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="directory of source images")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=parse_candidate,
        metavar="NAME=DIR",
        help="named candidate directory; repeat for multiple candidates",
    )
    parser.add_argument(
        "--output-html",
        "--output",
        dest="output_html",
        default="typical_comparison.html",
        help="destination HTML (default: typical_comparison.html)",
    )
    parser.add_argument(
        "--output-json",
        help="pairing JSON; default is OUTPUT_HTML with a .json suffix",
    )
    parser.add_argument("--title", default="Typical rectification comparison")
    parser.add_argument("--thumbnail-width", type=int, default=320)
    args = parser.parse_args()

    report = generate_comparison(
        args.source,
        args.candidate,
        args.output_html,
        output_json=args.output_json,
        title=args.title,
        thumbnail_width=args.thumbnail_width,
    )
    pairing = report["pairing"]
    print(
        f"wrote {report['output_html']} and {report['output_json']}; "
        f"source={pairing['source_count']} pairings={pairing['total_pairings']} "
        f"complete={pairing['complete_row_count']}"
    )


if __name__ == "__main__":
    main()
