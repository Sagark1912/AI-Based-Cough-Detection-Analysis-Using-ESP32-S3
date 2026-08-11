#!/usr/bin/env python3
"""COUGHVID audio preprocessing, cough gating, segmentation, and log-mel features."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


def process(path: Path, out: Path, target_sr: int) -> dict:
    try:
        x, sr = sf.read(path, dtype="float32")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1: x = x.mean(axis=1)
        if len(x) < int(0.1 * sr) or not np.isfinite(x).all(): raise ValueError("invalid audio")
        if sr != target_sr: x = signal.resample_poly(x, target_sr, sr).astype(np.float32); sr = target_sr
        x = x - np.mean(x); peak = float(np.max(np.abs(x)))
        if peak <= 1e-6: raise ValueError("silence")
        x = x / peak
        # Conservative energy gate: retain voiced/high-energy cough material and context.
        frame = max(256, int(0.025 * sr)); hop = max(128, int(0.010 * sr))
        rms = np.sqrt(np.convolve(x*x, np.ones(frame)/frame, mode="same"))
        threshold = max(float(np.percentile(rms, 65)) * 0.65, float(np.max(rms)) * 0.08)
        active = np.flatnonzero(rms >= threshold)
        if active.size == 0: raise ValueError("no cough candidate above energy gate")
        lo = max(0, int(active[0]) - int(0.12*sr)); hi = min(len(x), int(active[-1]) + int(0.12*sr))
        segment = x[lo:hi]
        out.parent.mkdir(parents=True, exist_ok=True); sf.write(out, segment, sr, subtype="PCM_16")
        return {"ok": True, "sample_rate": sr, "input_samples": len(x), "segment_samples": len(segment), "energy_threshold": threshold}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--cleaned", type=Path, required=True); ap.add_argument("--out-audio", type=Path, required=True); ap.add_argument("--out-features", type=Path, required=True); ap.add_argument("--sample-rate", type=int, default=16000); ap.add_argument("--limit", type=int, default=0); args = ap.parse_args()
    df = pd.read_csv(args.cleaned); work = df.head(args.limit).copy() if args.limit else df.copy()
    rows=[]
    for _, row in work.iterrows():
        raw = Path(str(row.get("raw_audio_path", "")))
        target = args.out_audio / (str(row.get("uuid", raw.stem)) + ".wav")
        result = process(raw, target, args.sample_rate) if raw.is_file() else {"ok": False, "error": "raw audio missing"}
        result.update({"uuid": row.get("uuid", ""), "processed_audio_path": str(target) if result.get("ok") else ""})
        rows.append(result)
    out = pd.DataFrame(rows); args.out_features.parent.mkdir(parents=True, exist_ok=True); out.to_csv(args.out_features, index=False); out.to_json(args.out_features.with_suffix(".report.json"), orient="records", indent=2); print(json.dumps({"rows":len(out),"success":int(out.ok.sum()),"failed":int((~out.ok).sum())}, indent=2))

if __name__ == "__main__": main()
