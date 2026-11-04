"""Download NSCH topical public-use Stata files from the U.S. Census Bureau.

Usage: python scripts/download_nsch.py 2024 [2023 ...]
Files land in data/raw/nsch_<year>/ (git-ignored).
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www2.census.gov/programs-surveys/nsch/datasets/{year}/nsch_{year}{suffix}_topical_Stata.zip"


def download(year: str) -> None:
    target = Path("data/raw") / f"nsch_{year}"
    target.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "e"):
        url = BASE.format(year=year, suffix=suffix)
        archive = target / Path(url).name
        try:
            urllib.request.urlretrieve(url, archive)
        except Exception:
            continue
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member.endswith(".dta") and "/" not in member and ".." not in member:
                    zf.extract(member, target)
        print(f"{year}: {archive} -> {[p.name for p in target.glob('*.dta')]}")
        return
    raise SystemExit(f"could not download NSCH {year}")


if __name__ == "__main__":
    for year in sys.argv[1:] or ["2024"]:
        download(year)
