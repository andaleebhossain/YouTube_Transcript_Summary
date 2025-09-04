from __future__ import annotations
import os
import re
from typing import List, Tuple, Optional
from urllib.parse import urlparse, parse_qs

import gradio as gr

from youtube_transcript_api import YouTubeTranscriptApi as YT
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound

try:
    from pytube import YouTube as PyTube
    _HAS_PYTUBE = True
except Exception:
    _HAS_PYTUBE = False

from transformers import pipeline

MODEL_NAME = os.getenv("SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
_SUMMARIZER = pipeline("summarization", model=MODEL_NAME)


def extract_video_id(url: str) -> str:
    if not url:
        return ""
    try:
        pr = urlparse(url)
        if pr.netloc and "youtu" in pr.netloc:
            q = parse_qs(pr.query)
            if "v" in q and len(q["v"]) > 0 and len(q["v"][0]) == 11:
                return q["v"][0]
            path_parts = [p for p in pr.path.split('/') if p]
            if pr.netloc.startswith("youtu.be") and path_parts:
                cand = path_parts[0]
                return cand if len(cand) == 11 else ""
    except Exception:
        pass
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else ""


def _join_raw(raw: List[dict]) -> str:
    return " ".join(seg.get("text", "") for seg in raw if seg.get("text"))


def get_transcript_text(video_id: str, languages: Optional[List[str]] = None) -> str:
    languages = languages or ["en", "en-US", "en-GB"]

    if not hasattr(YT, "get_transcript"):
        api = YT()
        try:
            ft = api.fetch(video_id, languages=languages)
            return _join_raw(ft.to_raw_data())
        except (TranscriptsDisabled, NoTranscriptFound):
            tlist = api.list(video_id)  
            for tr in tlist:
                if getattr(tr, "language_code", "").startswith("en"):
                    return _join_raw(tr.fetch().to_raw_data())
            
            for tr in tlist:
                if getattr(tr, "is_translatable", False):
                    return _join_raw(tr.translate("en").fetch().to_raw_data())
            raise

    try:
        raw = YT.get_transcript(video_id, languages=languages)
        return _join_raw(raw)
    except (TranscriptsDisabled, NoTranscriptFound):
        tlist = YT.list_transcripts(video_id)
        for tr in tlist:
            if getattr(tr, "language_code", "").startswith("en"):
                return _join_raw(tr.fetch())
        for tr in tlist:
            if getattr(tr, "is_translatable", False):
                return _join_raw(tr.translate("en").fetch())
        raise


def get_transcript_via_pytube(url: str) -> str:
    if not _HAS_PYTUBE:
        raise RuntimeError("pytube not installed; run `pip install pytube` or disable this fallback.")
    yt = PyTube(url)
    cap = yt.captions.get_by_language_code('en') or yt.captions.get_by_language_code('a.en')
    if not cap:
        raise RuntimeError("No captions found via pytube (original or auto-generated).")
    import html
    import re as _re
    xml = cap.xml_captions
    text = _re.sub(r'<.*?>', '', xml)
    return html.unescape(text)


def _chunk(text: str, max_chars: int = 2200) -> List[str]:
    words = text.split()
    chunks, cur, cur_len = [], [], 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= max_chars:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def summarize_long_text(text: str, style: str = "concise") -> Tuple[str, List[str]]:
    style = (style or "concise").lower()
    if style == "detailed":
        min_len, max_len = 120, 260
    elif style == "short":
        min_len, max_len = 40, 120
    else:
        min_len, max_len = 80, 200

    parts = _chunk(text)
    partial = []
    for p in parts:
        s = _SUMMARIZER(p, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        partial.append(s)

    joined = " ".join(partial)
    final = _SUMMARIZER(joined, max_length=max_len + 40, min_length=min_len, do_sample=False)[0]["summary_text"]
    return final, partial


def to_bullets(text: str, k: int = 6) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    bullets = [f"• {s}" for s in sentences[: max(1, k)] if s]
    return "\n".join(bullets)


def fetch_metadata(url: str) -> str:
    if not _HAS_PYTUBE:
        return ""
    try:
        yt = PyTube(url)
        title = yt.title or ""
        chan = yt.author or ""
        length = yt.length or 0
        mins = length // 60
        secs = length % 60
        return f"Title: {title}\nChannel: {chan}\nDuration: {mins}m {secs}s"
    except Exception:
        return ""


def app(url: str, style: str, bullets_n: int, allow_pytube_fallback: bool):
    if not url:
        return ("Enter a YouTube URL.", "", "", "")

    vid = extract_video_id(url)
    if not vid:
        return ("Could not parse video ID. Check the URL.", "", "", "")

    meta = fetch_metadata(url)

    try:
        transcript = get_transcript_text(vid)
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        if allow_pytube_fallback:
            try:
                transcript = get_transcript_via_pytube(url)
            except Exception as e2:
                return (f"Transcript unavailable: {type(e).__name__}. Fallback also failed: {e2}", meta, "", "")
        else:
            return (f"Transcript unavailable: {type(e).__name__}. Try enabling fallback or pick a video with captions.", meta, "", "")
    except Exception as e:
        return (f"Transcript fetching error: {e}", meta, "", "")

    if not transcript or len(transcript.strip()) < 40:
        return ("Transcript is empty or too short to summarize.", meta, "", "")

    final_summary, chunk_summaries = summarize_long_text(transcript, style=style)
    bullets = to_bullets(final_summary, k=bullets_n)

    return (final_summary, meta, bullets, "\n\n".join(f"Chunk {i+1}: {s}" for i, s in enumerate(chunk_summaries)))


with gr.Blocks(title="YouTube Summarizer") as demo:
    gr.Markdown("""
    # YouTube Caption Analyzer
    Paste a YouTube URL to get an executive summary, key takeaways, and transparent per‑chunk notes.
    """)

    with gr.Row():
        url_in = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    with gr.Row():
        style = gr.Dropdown(
            label="Summary style",
            choices=["concise", "detailed", "short"],
            value="concise",
        )
        bullets_n = gr.Slider(1, 10, value=6, step=1, label="# of bullets")
        allow_pytube_fallback = gr.Checkbox(value=False, label="Allow pytube fallback if transcript blocked")

    run_btn = gr.Button("Summarize")

    exec_out = gr.Textbox(label="Executive Summary")
    meta_out = gr.Textbox(label="Video Metadata (optional)")
    bullets_out = gr.Textbox(label="Key Takeaways")
    chunks_out = gr.Textbox(label="Intermediate Chunk Summaries (transparency)")

    run_btn.click(app, inputs=[url_in, style, bullets_n, allow_pytube_fallback], outputs=[exec_out, meta_out, bullets_out, chunks_out])


if __name__ == "__main__":
    demo.launch()
