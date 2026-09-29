"""Fetch and unpack MovieLens-25M.

Idempotent: returns immediately when the extracted marker exists.  Partial
downloads resume from a .part file.  The checksum is read from the official
`ml-25m.zip.md5` published next to the archive rather than hard-coded here, so
a wrong constant in this file can never silently accept a corrupted archive.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import requests

ML25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML25M_MD5_URL = ML25M_URL + ".md5"
CHUNK = 1 << 20


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _official_md5() -> str | None:
    try:
        r = requests.get(ML25M_MD5_URL, timeout=30)
        r.raise_for_status()
        token = r.text.strip().split()[0].lower()
        return token if len(token) == 32 else None
    except Exception as e:  # network hiccup: caller decides what to do
        print(f"[download] could not fetch official checksum ({e})")
        return None


def download_ml25m(root, verify: bool = True) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "ml-25m"
    if (extracted / "ratings.csv").exists() and (extracted / "genome-scores.csv").exists():
        print(f"[download] already extracted at {extracted}")
        return extracted

    zpath, part = root / "ml-25m.zip", root / "ml-25m.zip.part"
    if not zpath.exists():
        start = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        print(f"[download] fetching {ML25M_URL} (resume at {start/1e6:.1f} MB)")
        with requests.get(ML25M_URL, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0)) + start
            done = start
            with open(part, "ab" if start else "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r[download]   {done/1e6:7.1f}/{total/1e6:7.1f} MB", end="", flush=True)
        print()
        shutil.move(str(part), str(zpath))

    if verify:
        want = _official_md5()
        if want is None:
            raise RuntimeError(
                "could not obtain the official checksum; re-run later, or pass "
                "verify=False (--no-verify) if you have checked the archive yourself")
        got = _md5(zpath)
        if got != want:
            raise RuntimeError(f"md5 mismatch: got {got}, expected {want}. Delete {zpath} and retry.")
        print("[download] checksum OK")

    print("[download] extracting ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    if not (extracted / "ratings.csv").exists():
        raise RuntimeError(f"extraction finished but {extracted/'ratings.csv'} is missing")
    return extracted
