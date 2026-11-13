"""Stage 1 - Data ingestion.

Sources (``ingestion.source`` in config):

* ``movielens`` - downloads the public **MovieLens 100K** dataset (943 users,
  1,682 movies, 100,000 ratings with timestamps). If the download fails and
  ``fallback_to_synthetic`` is true, synthetic data is generated instead.
* ``synthetic`` - generates documented synthetic data in the *exact* MovieLens
  file format (``u.user``, ``u.item``, ``u.data``), so the same parser is used.

The three raw tables are stored **unchanged, as text** in ``data/raw/`` (raw
zone = immutable copy of the source). To demonstrate the validation stage, a
small, fully documented set of dirty records can be injected; the list of
injected problems is saved to ``data/raw/injected_issues.csv``.
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.io import save_json
from src.utils.logger import get_logger

log = get_logger(__name__)

# 19 MovieLens genres in the column order used by u.item
GENRES = ["unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary",
          "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci_fi",
          "thriller", "war", "western"]
OCCUPATIONS = ["administrator", "artist", "doctor", "educator", "engineer", "entertainment", "executive",
               "healthcare", "homemaker", "lawyer", "librarian", "marketing", "none", "other", "programmer",
               "retired", "salesman", "scientist", "student", "technician", "writer"]
USER_COLS = ["user_id", "age", "gender", "occupation", "zip_code"]
ITEM_COLS = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + [f"genre_{g}" for g in GENRES]
INTERACTION_COLS = ["user_id", "item_id", "rating", "timestamp"]


# --------------------------------------------------------------------------- sources
def download_movielens(url: str, dest_dir: Path) -> Path:
    """Download and extract MovieLens 100K into ``dest_dir/ml-100k`` (skipped if already present)."""
    folder = dest_dir / "ml-100k"
    if (folder / "u.data").exists():
        log.info("MovieLens 100K already present in %s - skipping download", folder)
        return folder
    log.info("Downloading MovieLens 100K from %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:          # nosec - public dataset URL
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(dest_dir)
    log.info("Downloaded %.1f MB and extracted to %s", len(payload) / 1e6, folder)
    return folder


def generate_synthetic_movielens(dest_dir: Path, n_users: int, n_items: int, n_ratings: int, seed: int) -> Path:
    """Generate MovieLens-format synthetic data driven by a latent-taste model.

    Generative story (documented so the data is reproducible and explainable):
      * each user has an age, gender, occupation, a join date between Sep-1997 and Mar-1998,
        an activity level (log-normal, >= 20 ratings) and a preference weight per genre;
      * each item has 1-3 genres, a release year and a Zipf-like popularity;
      * a user picks items with probability proportional to popularity x genre affinity,
        rates them 1-5 from (user bias + item quality + genre affinity + noise) and the
        rating timestamps fall between the user's join date and 22-Apr-1998.
    """
    rng = np.random.default_rng(seed)
    folder = dest_dir / "ml-100k-synthetic"
    folder.mkdir(parents=True, exist_ok=True)

    # ---- users
    ages = np.clip(rng.normal(33, 12, n_users).round(), 7, 73).astype(int)
    genders = rng.choice(["M", "F"], n_users, p=[0.71, 0.29])
    occ = rng.choice(OCCUPATIONS, n_users)
    zips = [f"{z:05d}" for z in rng.integers(1000, 99999, n_users)]
    users = pd.DataFrame({"user_id": np.arange(1, n_users + 1), "age": ages, "gender": genders,
                          "occupation": occ, "zip_code": zips})

    # ---- items
    n_g = len(GENRES) - 1                                     # 'unknown' is not sampled
    genre_pop = rng.dirichlet(np.ones(n_g) * 0.8)
    item_genres = np.zeros((n_items, len(GENRES)), dtype=int)
    for i in range(n_items):
        k = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        gs = rng.choice(n_g, size=k, replace=False, p=genre_pop) + 1
        item_genres[i, gs] = 1
    years = np.clip(1998 - rng.exponential(8, n_items).round(), 1930, 1998).astype(int)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rel = [f"{rng.integers(1, 28):02d}-{months[rng.integers(0, 12)]}-{y}" for y in years]
    titles = [f"Synthetic Movie {i:04d} ({y})" for i, y in zip(range(1, n_items + 1), years)]
    popularity = 1.0 / np.arange(1, n_items + 1) ** 0.9
    rng.shuffle(popularity)
    quality = rng.normal(0, 0.6, n_items)

    # ---- users' tastes and activity
    taste = rng.dirichlet(np.ones(len(GENRES)) * 0.5, n_users)
    taste[:, 0] = 0
    activity = np.clip(rng.lognormal(4.2, 0.9, n_users), 20, 700)
    activity = (activity / activity.sum() * n_ratings).round().astype(int)
    activity = np.clip(activity, 20, n_items - 1)
    user_bias = rng.normal(0, 0.45, n_users)
    start = pd.Timestamp("1997-09-20").value // 10**9
    end = pd.Timestamp("1998-04-22").value // 10**9
    join = rng.uniform(start, end - 20 * 86400, n_users)

    rows = []
    for u in range(n_users):
        affinity = item_genres @ taste[u] / np.maximum(item_genres.sum(1), 1)
        p = popularity * (0.2 + affinity * 5)
        p /= p.sum()
        items = rng.choice(n_items, size=activity[u], replace=False, p=p)
        score = 3.4 + user_bias[u] + quality[items] + 2.5 * (affinity[items] - affinity.mean()) + rng.normal(0, 0.8, len(items))
        ratings = np.clip(np.rint(score), 1, 5).astype(int)
        u_end = min(end, join[u] + rng.uniform(5, 220) * 86400)
        ts = np.sort(rng.uniform(join[u], u_end, len(items))).astype(int)
        rows.append(pd.DataFrame({"user_id": u + 1, "item_id": items + 1, "rating": ratings, "timestamp": ts}))
    data = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed)

    # ---- write in the exact MovieLens 100K formats
    users.to_csv(folder / "u.user", sep="|", header=False, index=False)
    item_df = pd.DataFrame({"item_id": np.arange(1, n_items + 1), "title": titles, "release_date": rel,
                            "video_release_date": "", "imdb_url": "http://example.org/synthetic"})
    item_df = pd.concat([item_df, pd.DataFrame(item_genres)], axis=1)
    item_df.to_csv(folder / "u.item", sep="|", header=False, index=False, encoding="latin-1")
    data.to_csv(folder / "u.data", sep="\t", header=False, index=False)
    (folder / "README_SYNTHETIC.txt").write_text(generate_synthetic_movielens.__doc__ or "", encoding="utf-8")
    log.info("Generated synthetic MovieLens-format data: %d users, %d items, %d ratings", n_users, n_items, len(data))
    return folder


def load_movielens_files(folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse u.user / u.item / u.data as **text** (no type coercion - that is validation's job)."""
    users = pd.read_csv(folder / "u.user", sep="|", header=None, names=USER_COLS, dtype=str, encoding="latin-1")
    items = pd.read_csv(folder / "u.item", sep="|", header=None, names=ITEM_COLS, dtype=str, encoding="latin-1")
    inter = pd.read_csv(folder / "u.data", sep="\t", header=None, names=INTERACTION_COLS, dtype=str)
    return users, items, inter


# ------------------------------------------------------------------ dirty records
def inject_dirty_records(users: pd.DataFrame, items: pd.DataFrame, inter: pd.DataFrame,
                         seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Append a documented set of invalid records (one per validation rule) to the raw tables.

    Returns the three dirtied tables plus a manifest describing every injected issue.
    Also blanks a few *optional* fields (occupation, zip code, release date) - these are
    not errors but missing values that the preparation stage must impute.
    """
    rng = np.random.default_rng(seed)
    users, items, inter = users.copy(), items.copy(), inter.copy()
    manifest = []
    valid_u = users["user_id"].iloc[0]
    valid_i = items["item_id"].iloc[0]

    def add_inter(row: dict, issue: str) -> None:
        inter.loc[len(inter)] = [row.get(c) for c in INTERACTION_COLS]
        manifest.append({"table": "interactions", "issue": issue, **{k: row.get(k) for k in INTERACTION_COLS}})

    # interactions - one or more examples per rule
    for _ in range(3):
        add_inter({"user_id": None, "item_id": valid_i, "rating": "4", "timestamp": "880000000"}, "missing user_id")
    add_inter({"user_id": valid_u, "item_id": valid_i, "rating": None, "timestamp": "880000000"}, "missing rating")
    for r in ["0", "6", "7", "-1"]:
        add_inter({"user_id": valid_u, "item_id": valid_i, "rating": r, "timestamp": "880000001"}, "rating out of range 1-5")
    add_inter({"user_id": valid_u, "item_id": valid_i, "rating": "3.5", "timestamp": "880000002"}, "non-integer rating")
    add_inter({"user_id": valid_u, "item_id": valid_i, "rating": "five", "timestamp": "880000003"}, "wrong data type (rating)")
    add_inter({"user_id": "abc", "item_id": valid_i, "rating": "3", "timestamp": "880000004"}, "wrong data type (user_id)")
    add_inter({"user_id": valid_u, "item_id": valid_i, "rating": "3", "timestamp": "4102444800"}, "timestamp in the future (2100)")
    add_inter({"user_id": valid_u, "item_id": valid_i, "rating": "3", "timestamp": "100"}, "timestamp before 1995")
    add_inter({"user_id": valid_u, "item_id": valid_i, "rating": "3", "timestamp": "yesterday"}, "unparseable timestamp")
    for _ in range(3):
        add_inter({"user_id": "99999", "item_id": valid_i, "rating": "4", "timestamp": "880000005"}, "unknown user_id (referential integrity)")
    for _ in range(2):
        add_inter({"user_id": valid_u, "item_id": "88888", "rating": "4", "timestamp": "880000006"}, "unknown item_id (referential integrity)")
    for idx in rng.choice(len(inter) - 30, 5, replace=False):     # exact duplicates of real rows
        row = inter.iloc[int(idx)].to_dict()
        add_inter(row, "exact duplicate row")

    # users
    def add_user(row: dict, issue: str) -> None:
        users.loc[len(users)] = [row.get(c) for c in USER_COLS]
        manifest.append({"table": "users", "issue": issue, **{k: row.get(k) for k in USER_COLS}})
    dup = users.iloc[1].to_dict()
    add_user(dup, "duplicate user_id")
    add_user({"user_id": "5001", "age": "-3", "gender": "M", "occupation": "student", "zip_code": "12345"}, "age out of range")
    add_user({"user_id": "5002", "age": "250", "gender": "F", "occupation": "writer", "zip_code": "12345"}, "age out of range")
    add_user({"user_id": "5003", "age": "30", "gender": "X", "occupation": "artist", "zip_code": "12345"}, "invalid gender code")
    add_user({"user_id": "5004", "age": None, "gender": "M", "occupation": "artist", "zip_code": "12345"}, "missing age")
    add_user({"user_id": "5005", "age": "thirty", "gender": "M", "occupation": "artist", "zip_code": "12345"}, "wrong data type (age)")

    # items
    def add_item(row: dict, issue: str) -> None:
        items.loc[len(items)] = [row.get(c) for c in ITEM_COLS]
        manifest.append({"table": "items", "issue": issue, "item_id": row.get("item_id"), "title": row.get("title")})
    base = items.iloc[2].to_dict()
    add_item(base, "duplicate item_id")
    add_item({**base, "item_id": "7001", "title": None}, "missing title")
    add_item({**base, "item_id": "7002", "release_date": "31-Feb-1996"}, "invalid release date")
    add_item({**base, "item_id": "7003", "genre_action": "2"}, "genre flag not 0/1")

    # optional fields blanked -> must be imputed during preparation (not errors)
    for col, n in [("occupation", 5), ("zip_code", 3)]:
        rows = rng.choice(len(users) - 10, n, replace=False)
        users.loc[rows, col] = None
        manifest += [{"table": "users", "issue": f"missing optional field '{col}' (to be imputed)", "user_id": users.loc[r, "user_id"]} for r in rows]
    rows = rng.choice(len(items) - 10, 3, replace=False)
    items.loc[rows, "release_date"] = None
    manifest += [{"table": "items", "issue": "missing optional field 'release_date' (to be imputed)", "item_id": items.loc[r, "item_id"]} for r in rows]

    # shuffle the dirty interactions into the data so they are not trivially at the end
    inter = inter.sample(frac=1, random_state=seed).reset_index(drop=True)
    return users, items, inter, pd.DataFrame(manifest)


# --------------------------------------------------------------------------- stage
def run_ingestion(cfg: Dict) -> Dict:
    """Run the ingestion stage and return a summary dict (also saved as JSON)."""
    t0 = time.time()
    icfg, raw_dir, seed = cfg["ingestion"], cfg["paths"]["raw"], cfg["project"]["seed"]
    source = icfg["source"]
    folder, used_source = None, source

    if source == "movielens":
        try:
            folder = download_movielens(icfg["movielens_url"], raw_dir)
        except Exception as exc:                               # network problems, proxy, etc.
            if not icfg.get("fallback_to_synthetic", True):
                raise
            log.warning("MovieLens download failed (%s). Falling back to synthetic data.", exc)
            used_source = "synthetic (fallback)"
    if folder is None:
        s = icfg["synthetic"]
        folder = generate_synthetic_movielens(raw_dir, s["n_users"], s["n_items"], s["n_ratings"], seed)
        used_source = used_source if used_source != "movielens" else "synthetic"

    users, items, inter = load_movielens_files(folder)
    log.info("Loaded source files: %d users, %d items, %d interactions", len(users), len(items), len(inter))

    manifest = pd.DataFrame()
    if icfg.get("inject_dirty_records", True):
        users, items, inter, manifest = inject_dirty_records(users, items, inter, seed)
        manifest.to_csv(raw_dir / "injected_issues.csv", index=False)
        log.info("Injected %d documented data-quality issues (see data/raw/injected_issues.csv)", len(manifest))

    users.to_csv(raw_dir / "users.csv", index=False)
    items.to_csv(raw_dir / "items.csv", index=False)
    inter.to_csv(raw_dir / "interactions.csv", index=False)

    summary = {"source": used_source, "source_folder": str(folder), "n_users_raw": len(users),
               "n_items_raw": len(items), "n_interactions_raw": len(inter),
               "n_injected_issues": int(len(manifest)), "seconds": round(time.time() - t0, 2),
               "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_json(summary, raw_dir / "ingestion_summary.json")
    log.info("Ingestion done: %s", summary)
    return summary
