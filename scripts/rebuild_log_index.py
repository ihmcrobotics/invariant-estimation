"""Rebuild a lost robotData.dat by walking the zstd frames in robotData.bsz.

robotData.dat is 16 bytes per batch, big-endian (int64 timestamp, int64 offset).
Both halves are recoverable: zstd frames are self-delimiting, so the offsets come
from walking them; and the first 8 bytes of a decompressed tick are that tick's
timestamp (stored as raw long bits), so the batch timestamp comes from its first
tick.

Usage:  rebuild_index.py <logdir> <out.dat> [--verify] [--limit N]
"""
import sys, struct, numpy as np, zstandard

def walk(bsz_path, limit=None, want_timestamps=True):
    dctx = zstandard.ZstdDecompressor()
    offsets, stamps = [], []
    CHUNK = 1 << 22
    with open(bsz_path, "rb") as f:
        pos = 0
        pending = b""
        while True:
            if limit is not None and len(offsets) >= limit:
                break
            obj = dctx.decompressobj()
            out = []
            start = pos - len(pending)
            done = False
            while not done:
                if not pending:
                    pending = f.read(CHUNK)
                    if not pending:
                        break
                    pos += len(pending)
                chunk, pending = pending, b""
                try:
                    piece = obj.decompress(chunk)
                except zstandard.ZstdError as e:
                    raise SystemExit(f"frame {len(offsets)} at {start}: {e}")
                if piece and want_timestamps and not out:
                    out.append(piece[:8])
                if obj.eof:
                    pending = obj.unused_data
                    done = True
            if not done and not out:
                break
            offsets.append(start)
            stamps.append(struct.unpack(">q", out[0])[0] if out else 0)
    return np.array(stamps, dtype=">i8"), np.array(offsets, dtype=">i8")

logdir, out = sys.argv[1], sys.argv[2]
verify = "--verify" in sys.argv
limit = None
if "--limit" in sys.argv:
    limit = int(sys.argv[sys.argv.index("--limit") + 1])

stamps, offsets = walk(f"{logdir}/robotData.bsz", limit)
print(f"recovered {len(offsets)} frames", flush=True)

if verify:
    truth = np.fromfile(f"{logdir}/robotData.dat", dtype=">i8").reshape(-1, 2)
    n = len(offsets)
    to, ts = truth[:n, 1], truth[:n, 0]
    print(f"offsets  match: {np.array_equal(to, offsets.astype('int64'))}")
    print(f"stamps   match: {np.array_equal(ts, stamps.astype('int64'))}")
    bad = np.flatnonzero(to != offsets.astype('int64'))
    if bad.size:
        i = bad[0]
        print(f"  first mismatch at {i}: truth={to[i]} rebuilt={int(offsets[i])}")
else:
    np.stack([stamps, offsets], axis=1).astype(">i8").tofile(out)
    print(f"wrote {out}")
