
"""
align_workflow.py
-----------------
Simple, modular pipeline to:
1) Transcribe audio with openai/whisper-large-v3-turbo (local, via 🤗 Transformers) and get word timestamps
2) Normalize & tokenize your *reference transcript* (with speaker labels)
3) Align your reference words to ASR words to inherit timestamps
4) Interpolate gaps and smooth, then attach speakers to words/sentences

Run:
  python align_workflow.py --audio path/to.wav --ref path/to_ref.json --out aligned.json

Install (example):
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu variant
  pip install transformers==4.* librosa==0.* numpy==1.* soundfile==0.*
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

# -----------------------------
# Section 0: Text normalization
# -----------------------------

import re
NUM_MAP = {
    # basic demo map; extend as needed
    "0": "zero","1": "one","2": "two","3": "three","4": "four","5": "five",
    "6": "six","7": "seven","8": "eight","9": "nine","10":"ten","11":"eleven",
    "12":"twelve","13":"thirteen","14":"fourteen","15":"fifteen","16":"sixteen",
    "17":"seventeen","18":"eighteen","19":"nineteen","20":"twenty","30":"thirty",
    "40":"forty","50":"fifty","60":"sixty","70":"seventy","80":"eighty","90":"ninety",
}

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    # unify quotes/dashes
    s = s.replace("—","-").replace("–","-").replace("’","'")
    # remove punctuation except apostrophes inside words
    s = re.sub(r"[^\w\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # expand simple numbers (very naive; replace with better num2words if needed)
    def repl_num(m):
        n = m.group(0)
        return NUM_MAP.get(n, n)
    s = re.sub(r"\b\d+\b", repl_num, s)
    return s

def tokenize_words(s: str) -> List[str]:
    # keep apostrophes for contractions
    return [w for w in re.findall(r"[a-zA-Z0-9]+'?[a-zA-Z0-9]+|[a-zA-Z0-9]+", s) if w]

# ---------------------------------
# Section 1: ASR via Transformers HF
# ---------------------------------

from transformers import pipeline

@dataclass
class ASRWord:
    word: str
    start: float
    end: float
    score: Optional[float] = None

def transcribe_with_whisper_hf(
    audio_path: str,
    model_id: str = "openai/whisper-large-v3-turbo",
    device: Optional[int] = None,
    chunk_length_s: float = 30.0,
    stride_length_s: float = 1.5,
    language: Optional[str] = None,
    task: str = "transcribe",
) -> List[ASRWord]:
    """
    Uses HF Transformers pipeline to get *word-level* timestamps.
    Returns a flat list of ASRWord with start/end (seconds).
    """
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype="auto",
        device=device if device is not None else 0 if torch.cuda.is_available() else -1,  # type: ignore
    )
    generate_kwargs = {"task": task}
    if language:
        generate_kwargs["language"] = language

    result = pipe(
        audio_path,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        return_timestamps="word",
        generate_kwargs=generate_kwargs,
    )
    # HF returns result["chunks"] with items: {'text': 'word', 'timestamp': (start,end), 'score': float?}
    words: List[ASRWord] = []
    for ch in result.get("chunks", []):
        ts = ch.get("timestamp", None)
        if not ts or ts[0] is None or ts[1] is None:
            continue
        words.append(ASRWord(word=normalize_text(ch["text"]), start=float(ts[0]), end=float(ts[1]), score=ch.get("score")))
    # enforce monotonicity & remove zero/neg durations
    prev = 0.0
    cleaned = []
    for w in words:
        start = max(w.start, prev)
        end = max(w.end, start + 1e-3)
        cleaned.append(ASRWord(w.word, start, end, w.score))
        prev = end
    return cleaned

# -------------------------------
# Section 2: Reference transcript
# -------------------------------

@dataclass
class RefSegment:
    speaker: str
    text: str

@dataclass
class RefWord:
    word: str
    idx: int  # global index in reference

def load_reference_json(path: str) -> List[RefSegment]:
    """
    Expect a JSON list like: [{"speaker": "A", "text": "Hello there."}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = []
    for item in data:
        segs.append(RefSegment(speaker=item.get("speaker", "UNK"), text=item.get("text", "")))
    return segs

def flatten_ref_words(segments: List[RefSegment]) -> Tuple[List[RefWord], List[Tuple[int,int]]]:
    """
    Returns:
      ref_words: flattened words with global indices
      seg_ranges: list of (start_idx, end_idx) for each input segment in ref_words space
    """
    ref_words: List[RefWord] = []
    seg_ranges: List[Tuple[int,int]] = []
    cur = 0
    for seg in segments:
        norm = normalize_text(seg.text)
        toks = tokenize_words(norm)
        start = cur
        for t in toks:
            ref_words.append(RefWord(word=t, idx=cur))
            cur += 1
        seg_ranges.append((start, cur))  # [start, end)
    return ref_words, seg_ranges

# ------------------------------------
# Section 3: Sequence alignment (words)
# ------------------------------------

@dataclass
class AlignedWord:
    word: str
    start: Optional[float]
    end: Optional[float]
    src: str  # "match" | "interp" | "gap"

def align_ref_to_asr(ref_words: List[RefWord], asr_words: List[ASRWord]) -> List[AlignedWord]:
    """
    Aligns tokens with difflib at the *word* level.
    Strategy:
      - Build SequenceMatcher over word strings
      - For equal blocks, copy timestamps
      - For ref-only runs (insertions), mark None (to be interpolated)
      - For asr-only runs (deletions), skip (no ref word)
    """
    a = [rw.word for rw in ref_words]
    b = [aw.word for aw in asr_words]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    aligned: List[AlignedWord] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                aw = asr_words[j]
                aligned.append(AlignedWord(word=ref_words[i].word, start=aw.start, end=aw.end, src="match"))
        elif tag == "replace":
            # mark ref words as gaps; we'll try local best-effort mapping by similarity inside window
            # naive: leave as gaps for interpolation
            for i in range(i1, i2):
                aligned.append(AlignedWord(word=ref_words[i].word, start=None, end=None, src="gap"))
        elif tag == "delete":
            # ref has words missing in ASR (unlikely here) -> mark gaps
            for i in range(i1, i2):
                aligned.append(AlignedWord(word=ref_words[i].word, start=None, end=None, src="gap"))
        elif tag == "insert":
            # words in ASR that aren't in ref; skip them (no aligned ref word)
            pass
    return aligned

# -----------------------------------
# Section 4: Interpolation & smoothing
# -----------------------------------

def interpolate_gaps(aligned: List[AlignedWord]) -> List[AlignedWord]:
    """
    For consecutive None spans, interpolate linearly between surrounding matched words.
    If leading/trailing gaps exist, extend from nearest known neighbor by word-duration median.
    """
    # collect indices of matched words
    idxs = [i for i, w in enumerate(aligned) if w.start is not None and w.end is not None]
    if not idxs:
        return aligned  # nothing we can do

    # median duration for fallback
    durs = [aligned[i].end - aligned[i].start for i in idxs]  # type: ignore
    dur_med = sorted(durs)[len(durs)//2] if durs else 0.25

    # leading gaps
    first = idxs[0]
    for i in range(first-1, -1, -1):
        aligned[i].start = max(0.0, aligned[first].start - (first - i) * dur_med)  # type: ignore
        aligned[i].end = aligned[i].start + dur_med
        aligned[i].src = "interp"

    # internal gaps
    for left, right in zip(idxs, idxs[1:]):
        t0 = aligned[left].end  # type: ignore
        t1 = aligned[right].start  # type: ignore
        gap = right - left - 1
        if gap <= 0:
            continue
        step = (t1 - t0) / (gap + 1)
        cur = t0
        for k in range(1, gap+1):
            i = left + k
            cur += step
            aligned[i].start = cur
            aligned[i].end = cur + max(1e-3, min(dur_med, step*0.9))
            aligned[i].src = "interp"

    # trailing gaps
    last = idxs[-1]
    for i in range(last+1, len(aligned)):
        start = aligned[last].end + (i - last - 1) * dur_med  # type: ignore
        aligned[i].start = start
        aligned[i].end = start + dur_med
        aligned[i].src = "interp"

    # enforce monotonic non-overlap
    prev_end = 0.0
    for w in aligned:
        if w.start is None or w.end is None:
            continue
        w.start = max(w.start, prev_end)
        if w.end <= w.start:
            w.end = w.start + 1e-3
        prev_end = w.end
    return aligned

def median_filter_words(aligned: List[AlignedWord], win: int = 3) -> None:
    if win < 3 or win % 2 == 0:
        return
    half = win // 2
    starts = [w.start for w in aligned]
    ends = [w.end for w in aligned]
    def med(vals, i):
        lo = max(0, i-half)
        hi = min(len(vals), i+half+1)
        window = [v for v in vals[lo:hi] if v is not None]
        if not window:
            return vals[i]
        return sorted(window)[len(window)//2]
    for i, w in enumerate(aligned):
        if w.start is not None:
            w.start = med(starts, i)
        if w.end is not None:
            w.end = med(ends, i)

# -----------------------------------
# Section 5: Speaker attachment
# -----------------------------------

@dataclass
class TimedWord:
    word: str
    start: float
    end: float
    speaker: str
    source: str  # "match" | "interp"

def attach_speakers(
    aligned: List[AlignedWord],
    segments: List[RefSegment],
    seg_ranges: List[Tuple[int,int]],
) -> List[TimedWord]:
    """
    Assign each aligned word the speaker of its original segment.
    """
    words: List[TimedWord] = []
    cursor = 0
    for (seg_idx, (lo, hi)) in enumerate(seg_ranges):
        spk = segments[seg_idx].speaker
        for i in range(lo, hi):
            aw = aligned[i]
            if aw.start is None or aw.end is None:
                # skip words that couldn't be timed (should be rare after interpolation)
                continue
            words.append(TimedWord(word=aw.word, start=float(aw.start), end=float(aw.end), speaker=spk, source="match" if aw.src=="match" else "interp"))
        cursor = hi
    return words

# -----------------------------------
# Section 6: Group into sentences
# -----------------------------------

def group_into_sentences(words: List[TimedWord], max_pause: float = 0.5) -> List[Dict[str, Any]]:
    """
    Simple pause-based sentence grouping + punctuation heuristics (since ref had punctuation removed).
    """
    if not words:
        return []
    sentences = []
    cur_start = words[0].start
    cur_words = [words[0]]
    for w_prev, w in zip(words, words[1:]):
        gap = w.start - w_prev.end
        if gap > max_pause or w.speaker != w_prev.speaker:
            sentences.append({
                "start": cur_start,
                "end": w_prev.end,
                "speaker": w_prev.speaker,
                "text": " ".join(x.word for x in cur_words),
                "source_frac_match": sum(1 for x in cur_words if x.source=="match")/len(cur_words)
            })
            cur_start = w.start
            cur_words = [w]
        else:
            cur_words.append(w)
    # flush
    last = words[-1]
    sentences.append({
        "start": cur_start,
        "end": last.end,
        "speaker": last.speaker,
        "text": " ".join(x.word for x in cur_words),
        "source_frac_match": sum(1 for x in cur_words if x.source=="match")/len(cur_words)
    })
    return sentences

# -----------------------------------
# Section 7: Orchestration
# -----------------------------------

def run_pipeline(audio_path: str, ref_json_path: str, model_id: str = "openai/whisper-large-v3-turbo") -> Dict[str, Any]:
    # 1) ASR
    asr_words = transcribe_with_whisper_hf(audio_path, model_id=model_id)
    # 2) Reference
    segments = load_reference_json(ref_json_path)
    ref_words, seg_ranges = flatten_ref_words(segments)
    # 3) Align
    aligned = align_ref_to_asr(ref_words, asr_words)
    aligned = interpolate_gaps(aligned)
    median_filter_words(aligned, win=3)
    # 4) Speakers
    timed_words = attach_speakers(aligned, segments, seg_ranges)
    # 5) Sentences
    sentences = group_into_sentences(timed_words, max_pause=0.6)
    return {
        "model": model_id,
        "audio": audio_path,
        "words": [asdict(w) for w in timed_words],
        "sentences": sentences,
        "meta": {
            "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "notes": "Word timestamps derived from HF pipeline; ref-to-ASR alignment via difflib + interpolation."
        }
    }

# -----------------------------------
# Section 8: CLI
# -----------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Path to audio file (wav/mp3/m4a...)")
    p.add_argument("--ref", required=True, help="Path to reference transcript JSON (speaker + text)")
    p.add_argument("--out", required=True, help="Path to output JSON")
    p.add_argument("--model", default="openai/whisper-large-v3-turbo", help="HF model id")
    args = p.parse_args()

    out = run_pipeline(args.audio, args.ref, model_id=args.model)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    # lazy import torch availability check used above
    import torch  # noqa: F401
    main()
