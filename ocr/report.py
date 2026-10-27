"""HTML report so preprocessing variants and transcriptions can be compared locally."""

from __future__ import annotations

import html
from pathlib import Path

from ocr.engine_compare import EngineCompareResult
from ocr.pipeline import CompareResult


def write_report(result: CompareResult, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    by_method = {item.preprocessing_method: item for item in result.transcriptions}
    cards = []
    for meta in result.variant_metadata:
        name = meta["name"]
        transcription = by_method.get(name)
        if transcription is None:
            body = "<p>Not transcribed.</p>"
            status = "not-run"
            latency = ""
        elif transcription.status != "success":
            body = f"<pre class=\"error\">{html.escape(transcription.error or 'error')}</pre>"
            status = "error"
            latency = f"{transcription.latency_ms:.0f} ms"
        else:
            flags = transcription.quality_warnings or []
            warning_html = ""
            if flags:
                items = "".join(
                    f"<li>{html.escape(flag)}</li>" for flag in flags
                )
                warning_html = (
                    "<p class=\"quality\">Quality warnings (heuristic, not "
                    f"verified OCR errors):</p><ul class=\"quality\">{items}</ul>"
                )
            body = (
                f"{warning_html}<pre>{html.escape(transcription.text or '')}</pre>"
            )
            status = "success"
            latency = f"{transcription.latency_ms:.0f} ms"
        steps = ", ".join(meta.get("steps") or [])
        cards.append(
            f"""
            <article>
              <h2>{html.escape(name)}</h2>
              <p class="meta">{html.escape(status)} {html.escape(latency)} · {html.escape(steps)}</p>
              <img src="{html.escape(name)}.png" alt="{html.escape(name)} variant">
              {body}
            </article>
            """
        )
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OCR comparison</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; background: #f6f4ef; color: #1c1c1c; }}
    article {{ background: white; padding: 16px; margin: 0 0 16px; border: 1px solid #ddd; }}
    img {{ max-width: 100%; height: auto; background: #eee; }}
    pre {{ white-space: pre-wrap; font-family: Consolas, monospace; }}
    .error {{ color: #8a1f1f; }}
    .meta {{ color: #555; }}
    .quality {{ color: #8a5a12; }}
  </style>
</head>
<body>
  <h1>Local handwritten OCR</h1>
  <p>Transcriptions come from the local checkpoint. The image was not uploaded.</p>
  <p>Source file name is omitted from this report header. Variant images are beside their raw transcriptions.</p>
  {''.join(cards)}
</body>
</html>
"""
    path = directory / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


def write_engine_report(result: EngineCompareResult, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in result.notes)
    trocr_body = "<p>TrOCR was not requested.</p>"
    if result.trocr is not None:
        crops = []
        for crop in result.trocr.crops:
            text = crop.text or crop.error or crop.status
            crops.append(
                f"<li><strong>{html.escape(crop.crop_id)}</strong> "
                f"({html.escape(crop.status)}): "
                f"<pre>{html.escape(text or '')}</pre></li>"
            )
        trocr_body = (
            f"<p>Status: {html.escape(result.trocr.status)}</p>"
            f"<p>{html.escape(result.trocr.error or '')}</p>"
            f"<ul>{''.join(crops)}</ul>"
        )
    page_html = ""
    if result.trocr_full_page is not None:
        page = result.trocr_full_page
        warn = "".join(f"<li>{html.escape(item)}</li>" for item in page.warnings)
        body = page.text or page.error or page.status
        page_html = (
            f"<section><h2>TrOCR full page</h2>"
            f"<p>Status: {html.escape(page.status)}</p>"
            f"<ul>{warn}</ul>"
            f"<pre>{html.escape(body or '')}</pre></section>"
        )
    qwen_body = "<p>Qwen was not requested.</p>"
    if result.qwen is not None:
        blocks = []
        for item in result.qwen.transcriptions:
            if item.status != "success":
                blocks.append(
                    f"<pre class=\"error\">{html.escape(item.error or 'error')}</pre>"
                )
            else:
                blocks.append(f"<pre>{html.escape(item.text or '')}</pre>")
        qwen_body = "".join(blocks)
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OCR engine comparison</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; background: #f6f4ef; color: #1c1c1c; }}
    section {{ background: white; padding: 16px; margin: 0 0 16px; border: 1px solid #ddd; }}
    pre {{ white-space: pre-wrap; font-family: Consolas, monospace; }}
    .error {{ color: #8a1f1f; }}
  </style>
</head>
<body>
  <h1>Qwen and TrOCR</h1>
  <p>Mode: {html.escape(result.mode)}. review_required: {result.review_required} ({html.escape(result.comparison_kind or "n/a")}).</p>
  <p>Qwen remains the default full-page engine. Crop TrOCR and full-page TrOCR are separate paths. Neither replaces Qwen. review_required is a human-inspection flag, not an accuracy metric.</p>
  <ul>{notes}</ul>
  <section>
    <h2>Qwen</h2>
    {qwen_body}
  </section>
  <section>
    <h2>TrOCR crops</h2>
    {trocr_body}
  </section>
  {page_html}
</body>
</html>
"""
    path = directory / "engine_report.html"
    path.write_text(document, encoding="utf-8")
    return path


def write_hybrid_report(result, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cards = []
    for item in result.lines:
        crop_src = item.crop_path or ""
        img = (
            f'<img src="{html.escape(crop_src)}" alt="{html.escape(item.line_id)}">'
            if crop_src
            else ""
        )
        flags = "".join(f"<li>{html.escape(note)}</li>" for note in item.warnings)
        cards.append(
            f"""
            <article>
              <h3>{html.escape(item.line_id)} (order {item.reading_order})</h3>
              <p class="meta">bbox_processed {html.escape(str(item.bbox))}
              · bbox_oriented {html.escape(str(getattr(item, "bbox_oriented", [])))}
              · {html.escape(item.selected_engine)}
              · {html.escape(item.status)} · review={item.review_required}</p>
              <p>{html.escape(item.routing_reason)}</p>
              {img}
              <pre>{html.escape(item.raw_text or "")}</pre>
              <ul class="quality">{flags}</ul>
            </article>
            """
        )
    detect = "".join(f"<li>{html.escape(note)}</li>" for note in result.detection_warnings)
    overall = "".join(f"<li>{html.escape(note)}</li>" for note in result.warnings)
    steps = "".join(
        f"<li>{html.escape(step)}</li>" for step in (result.preprocess_steps or [])
    )
    original = ""
    if (directory / "original.png").is_file():
        original = '<p><img src="original.png" alt="oriented original"></p>'
    preprocessed = ""
    if (directory / "preprocessed.png").is_file():
        preprocessed = '<p><img src="preprocessed.png" alt="combined_gray preprocessed"></p>'
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hybrid prescription OCR</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; background: #f6f4ef; color: #1c1c1c; }}
    article {{ background: white; padding: 16px; margin: 0 0 16px; border: 1px solid #ddd; }}
    img {{ max-width: 100%; height: auto; background: #eee; }}
    pre {{ white-space: pre-wrap; font-family: Consolas, monospace; }}
    .meta {{ color: #555; }}
    .quality {{ color: #8a5a12; }}
    .error {{ color: #8a1f1f; }}
  </style>
</head>
<body>
  <h1>Hybrid prescription OCR</h1>
  <p>review_required is always true. This is not a medically verified transcription.</p>
  <p>Status: {html.escape(result.status)}. Lines: {result.line_count}.
  Fallbacks: {result.fallback_count}. Engines: {html.escape(str(result.engine_counts))}.</p>
  <p>Preprocess variant: {html.escape(str(result.preprocess_variant or "combined_gray"))}.
  Oriented size: {result.width}x{result.height}.
  Processed size: {result.processed_width}x{result.processed_height}.</p>
  <h2>Preprocessing steps</h2>
  <ul>{steps}</ul>
  <h2>Oriented original</h2>
  {original}
  <h2>Preprocessed (combined_gray)</h2>
  {preprocessed}
  <h2>Detection warnings</h2>
  <ul>{detect}</ul>
  <h2>Overall warnings</h2>
  <ul>{overall}</ul>
  <h2>Line crops</h2>
  {''.join(cards)}
  <h2>Merged transcription (reading order)</h2>
  <pre>{html.escape(result.merged_transcription or "")}</pre>
</body>
</html>
"""
    path = directory / "hybrid_report.html"
    path.write_text(document, encoding="utf-8")
    return path
