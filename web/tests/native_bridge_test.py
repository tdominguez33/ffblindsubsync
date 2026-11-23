"""Native correctness test for the browser bridge's sync logic.

We cannot run Pyodide headlessly in this environment (no npm network), but the
bridge module is plain, runtime-agnostic Python driving the real ffsubsync, so a
native run against a known synthetic offset proves the sub-vs-sub path and the
bridge glue. The browser path reuses the exact same bridge + ffsubsync code.

Run with: uv run python web/tests/native_bridge_test.py
"""

import os
import sys
import tempfile

# Make the bridge importable and redirect its MEMFS-style workdir to a temp dir.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, REPO_ROOT)  # import the local (vendored) ffsubsync sources

import ffsubsync_bridge as bridge  # noqa: E402

bridge._WORK_DIR = tempfile.mkdtemp(prefix="ffs_bridge_")


def make_srt(offset_seconds: float, n: int = 40) -> bytes:
    """A simple SRT: one 1s cue every 2s, all shifted by ``offset_seconds``."""
    lines = []
    for i in range(n):
        start = i * 2.0 + offset_seconds
        end = start + 1.0
        lines.append(str(i + 1))
        lines.append("{} --> {}".format(_ts(start), _ts(end)))
        lines.append("line {}".format(i + 1))
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def make_srt_from(cues) -> bytes:
    """Serialize ``[(start, end, text), ...]`` to SRT bytes."""
    lines = []
    for i, (start, end, text) in enumerate(cues, 1):
        lines.append(str(i))
        lines.append("{} --> {}".format(_ts(start), _ts(end)))
        lines.append(text)
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def make_break_pair(break_shift: float = 8.0):
    """A reference + an input with a mid-file break (second half shifted late).

    Timings are deliberately *irregular* (varying gaps and durations) so that no
    single global offset can masquerade as correct by aliasing one cue onto another
    -- the split aligner is the only thing that can fix both halves.
    """
    gaps = [1.3, 2.7, 0.9, 3.1, 1.7, 2.2, 0.8, 2.9, 1.1, 3.3, 1.9, 2.4, 0.7, 3.0, 1.5]
    durs = [1.2, 0.9, 2.1, 1.4, 0.8, 1.9, 1.1, 2.3, 0.9, 1.6, 1.3, 2.0, 1.0, 1.8, 1.2]
    ref, t = [], 1.0
    for i, (g, d) in enumerate(zip(gaps, durs)):
        ref.append((t, t + d, "dialogue line number {}".format(i)))
        t += d + g
    broken = [
        (a + break_shift, b + break_shift, x) if a >= 20 else (a, b, x)
        for (a, b, x) in ref
    ]
    return make_srt_from(ref), make_srt_from(broken), ref


def _ts(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return "{:02d}:{:02d}:{:02d},{:03d}".format(h, m, s, ms)


def main() -> int:
    reference = make_srt(0.0)
    # Input subtitles arrive 5s late relative to the reference.
    delayed = make_srt(5.0)
    expected_offset = -5.0

    result = bridge.sync_subtitles(
        "reference.srt", reference, "input.srt", delayed
    )

    print("ok:", result["ok"])
    print("offset_seconds:", result["offset_seconds"])
    print("framerate_scale_factor:", result["framerate_scale_factor"])
    print("output_name:", result["output_name"])
    print("output_text[:120]:", repr(result["output_text"][:120]))
    if result["error"]:
        print("error:", result["error"])

    assert result["ok"], "sync should succeed"
    off = result["offset_seconds"]
    assert off is not None, "offset should be set"
    assert abs(off - expected_offset) < 0.25, (
        "expected offset ~%.2f, got %.3f" % (expected_offset, off)
    )
    assert result["output_text"].strip(), "output should be non-empty"
    assert "-->" in result["output_text"], "output should be SRT"
    print("\nPASS: detected offset %.3fs matches expected %.1fs" % (off, expected_offset))

    # --- split_sync path: a mid-file break a single offset cannot fix -------------
    import srt as _srt

    ref_bytes, broken_bytes, ref_cues = make_break_pair(break_shift=8.0)

    plain = bridge.sync_subtitles(
        "reference.srt", ref_bytes, "broken.srt", broken_bytes,
        max_offset_seconds=30,
    )
    split = bridge.sync_subtitles(
        "reference.srt", ref_bytes, "broken.srt", broken_bytes,
        split_sync=True, max_offset_seconds=30,
    )
    assert split["ok"], "split sync should succeed"

    ref_starts = [c[0] for c in ref_cues]
    plain_starts = [c.start.total_seconds() for c in _srt.parse(plain["output_text"])]
    split_starts = [c.start.total_seconds() for c in _srt.parse(split["output_text"])]
    plain_err = max(abs(a - b) for a, b in zip(ref_starts, plain_starts))
    split_err = max(abs(a - b) for a, b in zip(ref_starts, split_starts))
    print("mid-file break: single-offset max|Δ|=%.3fs, split max|Δ|=%.3fs"
          % (plain_err, split_err))
    assert plain_err > 5.0, "a single offset should NOT be able to fix a mid-file break"
    assert split_err < 0.1, "split_sync should correct both halves"
    print("PASS: split_sync fixes the mid-file break (Δ %.3fs) the single offset "
          "could not (Δ %.3fs)" % (split_err, plain_err))

    # --- auto_split path: safe on unbroken file, fixes real break ------------
    auto_clean = bridge.sync_subtitles(
        "reference.srt", reference, "input.srt", delayed,
        auto_split=True,
    )
    assert auto_clean["ok"], "auto_split on unbroken file should succeed"
    assert auto_clean["split_segments"] and len(auto_clean["split_segments"]) == 1, (
        "auto_split should produce 0 splits on unbroken input, got %r"
        % (auto_clean["split_segments"],)
    )
    assert abs(auto_clean["offset_seconds"] - expected_offset) < 0.25

    auto_break = bridge.sync_subtitles(
        "reference.srt", ref_bytes, "broken.srt", broken_bytes,
        auto_split=True, max_offset_seconds=30,
    )
    assert auto_break["ok"], "auto_split on broken file should succeed"
    auto_starts = [c.start.total_seconds() for c in _srt.parse(auto_break["output_text"])]
    auto_err = max(abs(a - b) for a, b in zip(ref_starts, auto_starts))
    assert auto_err < 0.1, "auto_split should fix mid-file break (got Δ %.3fs)" % auto_err
    assert len(auto_break["split_segments"]) == 2
    print("PASS: auto_split leaves clean files unbroken and fixes mid-file break (Δ %.3fs)" % auto_err)

    # --- PGS/VobSub/DVB timings path ----------------------------------------
    timings = [(i * 2.0, i * 2.0 + 1.0) for i in range(40)]
    pgs_res = bridge.sync_with_timings(timings, "input.srt", delayed)
    assert pgs_res["ok"], "sync_with_timings should succeed"
    assert abs(pgs_res["offset_seconds"] - expected_offset) < 0.25, (
        "expected timings offset ~%.2f, got %.3f" % (expected_offset, pgs_res["offset_seconds"])
    )
    print("PASS: sync_with_timings recovered offset %.3fs" % pgs_res["offset_seconds"])

    # --- Preserve .ass and .vtt output formats in Web -----------------------
    import pysubs2

    ass_in = pysubs2.SSAFile.from_string(delayed.decode("utf-8")).to_string("ass").encode("utf-8")
    ass_res = bridge.sync_subtitles("reference.srt", reference, "anime.ass", ass_in)
    assert ass_res["ok"], "ASS sync should succeed"
    assert ass_res["output_name"] == "anime.synced.ass"
    assert "[Script Info]" in ass_res["output_text"]
    assert "[V4+ Styles]" in ass_res["output_text"]

    vtt_in = pysubs2.SSAFile.from_string(delayed.decode("utf-8")).to_string("vtt").encode("utf-8")
    vtt_res = bridge.sync_subtitles("reference.srt", reference, "episode.vtt", vtt_in)
    assert vtt_res["ok"], "VTT sync should succeed"
    assert vtt_res["output_name"] == "episode.synced.vtt"
    assert vtt_res["output_text"].startswith("WEBVTT")
    print("PASS: web bridge preserves .ass and .vtt formats")

    # --- min_ref_cues rejects sparse embedded text refs ---------------------
    sparse_ref = make_srt(0.0, n=10)
    sparse_res = bridge.sync_subtitles(
        "embedded.srt", sparse_ref, "input.srt", delayed, min_ref_cues=50
    )
    assert not sparse_res["ok"], "sparse reference (<50 cues) should be rejected"
    assert "only 10 cues" in (sparse_res["error"] or "")
    print("PASS: min_ref_cues rejects sparse embedded text references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
