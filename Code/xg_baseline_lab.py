# /// script

# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.1",
#     "numpy>=2.5.1",
#     "pandas>=3.0.5",
#     "requests>=2.34.2",
#     "scikit-learn>=1.9.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # xG Baseline Lab — Week 4 (MSDS 696)

    **Goal this week:** scale to the full StatsBomb Open Data corpus, lock a
    match-grouped train/val/test split, and fit the logistic-regression
    baseline (distance + angle) that every later feature family must beat.

    **Ground rules carried over from Weeks 1–3**

    - Seed **696** everywhere randomness appears.
    - Barred leakage columns never enter the feature set: `shot_outcome`
      (used once to derive the label, then discarded), `shot_statsbomb_xg`
      (kept only as an external benchmark column), `shot_end_location`,
      `shot_deflected` (never extracted at all).
    - Calibration first: Brier, log loss, ECE, reliability curves. AUC is
      reported as a secondary ranking metric only.
    - The **test split is locked until Week 7** — every number in this
      notebook is computed on the validation split.

    Quick smoke test (a dozen matches, no parquet written):
    `XG_SMOKE=1 python xg_baseline_lab.py`
    """)
    return


@app.cell
def _():
    import json
    import math
    import os
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import requests

    SEED = 696
    RNG = np.random.default_rng(SEED)

    BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

    # competition_id -> display name (the four corpus competitions)
    COMPETITIONS = {
        11: "La Liga",
        2: "Premier League",
        37: "FA Women's Super League",
        43: "FIFA World Cup",
    }

    DATA_DIR = Path("data")
    CACHE_DIR = DATA_DIR / "statsbomb_cache"
    PARQUET_PATH = DATA_DIR / "xg_shots_full.parquet"
    FIG_DIR = Path("figures")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    SMOKE = os.environ.get("XG_SMOKE") == "1"
    SMOKE_MATCHES = 12

    EXCLUDE_PENALTIES = True  # penalties are a fixed-odds event; modeled separately if time allows
    SPLIT_FRACS = (0.70, 0.15, 0.15)  # train / val / test, allocated by shot count

    # StatsBomb pitch: 120 x 80, attacking goal at x = 120, posts at y = 36 / 44
    GOAL_X, POST_LOW, POST_HIGH, GOAL_CY = 120.0, 36.0, 44.0, 40.0
    return (
        BASE_URL,
        CACHE_DIR,
        COMPETITIONS,
        EXCLUDE_PENALTIES,
        FIG_DIR,
        GOAL_CY,
        GOAL_X,
        PARQUET_PATH,
        POST_HIGH,
        POST_LOW,
        RNG,
        SMOKE,
        SMOKE_MATCHES,
        SPLIT_FRACS,
        json,
        math,
        np,
        pd,
        requests,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Data acquisition

    Direct pulls from the StatsBomb `open-data` GitHub repo with a local
    JSON cache (`data/statsbomb_cache/`). The first full build downloads
    every event file once (~3,000+ matches, so expect it to run a while);
    every rerun after that is instant. The finished shot table is saved as
    `data/xg_shots_full.parquet` and loaded directly on later runs.
    """)
    return


@app.cell
def _(CACHE_DIR, json, requests, time):
    _session = requests.Session()

    def cached_get(url, cache_key):
        """GET a JSON resource, caching the raw text on disk."""
        path = CACHE_DIR / cache_key
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        last_err = None
        for attempt in range(3):
            try:
                resp = _session.get(url, timeout=60)
                resp.raise_for_status()
                path.write_text(resp.text, encoding="utf-8")
                return resp.json()
            except Exception as err:  # noqa: BLE001 — retry any transient failure
                last_err = err
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"failed to fetch {url}: {last_err}")

    return (cached_get,)


@app.cell
def _(GOAL_X, POST_HIGH, POST_LOW, math):
    def _tri_sign(ax, ay, bx, by, cx, cy):
        return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)

    def in_shot_cone(px, py, sx, sy):
        """Is point (px, py) inside the triangle shot-location -> both posts?"""
        d1 = _tri_sign(px, py, sx, sy, GOAL_X, POST_LOW)
        d2 = _tri_sign(px, py, GOAL_X, POST_LOW, GOAL_X, POST_HIGH)
        d3 = _tri_sign(px, py, GOAL_X, POST_HIGH, sx, sy)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def parse_shot(ev, match_id, comp_name, season_name):
        """One event dict -> one flat shot row (or None if unusable).

        Leakage discipline: `shot_end_location` and `shot_deflected` are never
        read. The outcome is consumed here to derive `is_goal` and is not
        stored. `statsbomb_xg` is stored under `benchmark_sb_xg` and is used
        for benchmarking only — never as a feature.
        """
        shot = ev.get("shot") or {}
        loc = ev.get("location") or [None, None]
        x, y = loc[0], loc[1]
        if x is None or y is None:
            return None
        outcome = (shot.get("outcome") or {}).get("name")

        row = {
            "match_id": match_id,
            "competition": comp_name,
            "season": season_name,
            "team": (ev.get("team") or {}).get("name"),
            "player": (ev.get("player") or {}).get("name"),
            "period": ev.get("period"),
            "minute": ev.get("minute"),
            "x": float(x),
            "y": float(y),
            "shot_type": (shot.get("type") or {}).get("name"),
            "body_part": (shot.get("body_part") or {}).get("name"),
            "technique": (shot.get("technique") or {}).get("name"),
            "play_pattern": (ev.get("play_pattern") or {}).get("name"),
            "under_pressure": bool(ev.get("under_pressure", False)),
            "first_time": bool(shot.get("first_time", False)),
            "one_on_one": bool(shot.get("one_on_one", False)),
            "open_goal": bool(shot.get("open_goal", False)),
            "follows_dribble": bool(shot.get("follows_dribble", False)),
            "aerial_won": bool(shot.get("aerial_won", False)),
            "is_goal": int(outcome == "Goal"),
            "benchmark_sb_xg": shot.get("statsbomb_xg"),
        }

        # Freeze-frame features precomputed at build time (used in a LATER
        # feature family, not by this week's baseline). Doing it here means
        # the raw nested freeze frames never need to be stored or re-pulled.
        gk_dist = None
        opp_in_cone = 0
        opp_within_5 = 0
        for pl in shot.get("freeze_frame") or []:
            ploc = pl.get("location") or [None, None]
            px, py = ploc[0], ploc[1]
            if px is None or py is None or pl.get("teammate"):
                continue
            dist = math.hypot(px - x, py - y)
            if (pl.get("position") or {}).get("name") == "Goalkeeper":
                gk_dist = dist
                continue
            if dist <= 5.0:
                opp_within_5 += 1
            if in_shot_cone(px, py, x, y):
                opp_in_cone += 1
        row["gk_dist"] = gk_dist
        row["opp_in_cone"] = opp_in_cone
        row["opp_within_5"] = opp_within_5
        return row

    return (parse_shot,)


@app.cell
def _(
    BASE_URL,
    COMPETITIONS,
    SMOKE,
    SMOKE_MATCHES,
    cached_get,
    parse_shot,
    pd,
):
    def build_corpus():
        comps = cached_get(f"{BASE_URL}/competitions.json", "competitions.json")
        targets = [c for c in comps if c["competition_id"] in COMPETITIONS]

        match_meta = []
        for comp in targets:
            cid, sid = comp["competition_id"], comp["season_id"]
            matches = cached_get(
                f"{BASE_URL}/matches/{cid}/{sid}.json", f"matches_{cid}_{sid}.json"
            )
            for m in matches:
                match_meta.append(
                    (m["match_id"], COMPETITIONS[cid], comp["season_name"])
                )

        if SMOKE:
            match_meta = match_meta[:SMOKE_MATCHES]
        print(f"building corpus from {len(match_meta)} matches ...")

        rows = []
        for i, (mid, comp_name, season_name) in enumerate(match_meta, 1):
            try:
                events = cached_get(
                    f"{BASE_URL}/events/{mid}.json", f"events_{mid}.json"
                )
            except Exception as err:  # noqa: BLE001 — skip, don't kill a long build
                print(f"  ! skipping match {mid}: {err}")
                continue
            for ev in events:
                if (ev.get("type") or {}).get("name") != "Shot":
                    continue
                row = parse_shot(ev, mid, comp_name, season_name)
                if row is not None:
                    rows.append(row)
            if i % 100 == 0 or i == len(match_meta):
                print(f"  {i}/{len(match_meta)} matches parsed, {len(rows)} shots")
        return pd.DataFrame(rows)

    return (build_corpus,)


@app.cell
def _(PARQUET_PATH, SMOKE, build_corpus, pd):
    if PARQUET_PATH.exists() and not SMOKE:
        shots_raw = pd.read_parquet(PARQUET_PATH)
        print(f"loaded cached corpus: {len(shots_raw):,} shots from {PARQUET_PATH}")
    else:
        shots_raw = build_corpus()
        if not SMOKE:
            shots_raw.to_parquet(PARQUET_PATH, index=False)
            print(f"saved corpus: {len(shots_raw):,} shots -> {PARQUET_PATH}")

    print(
        shots_raw.groupby("competition")["is_goal"]
        .agg(shots="size", goal_rate="mean")
        .round(4)
    )
    return (shots_raw,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Features (baseline scope only)

    Penalties are excluded from the modeling table (fixed-odds event; can
    be modeled separately later). The baseline uses exactly two features —
    **distance to goal center** and **the angle subtended by the posts** —
    so that every feature family added in Week 5 has an honest yardstick.
    """)
    return


@app.cell
def _(EXCLUDE_PENALTIES, GOAL_CY, GOAL_X, POST_HIGH, POST_LOW, np, shots_raw):
    df_model = shots_raw.copy()
    n_before_pen = len(df_model)
    if EXCLUDE_PENALTIES:
        df_model = df_model[df_model["shot_type"] != "Penalty"].copy()
    print(
        f"shots: {n_before_pen:,} raw -> {len(df_model):,} after penalty exclusion "
        f"({n_before_pen - len(df_model):,} penalties removed)"
    )

    df_model["distance"] = np.hypot(
        GOAL_X - df_model["x"], GOAL_CY - df_model["y"]
    )
    _a = np.arctan2(POST_HIGH - df_model["y"], GOAL_X - df_model["x"])
    _b = np.arctan2(POST_LOW - df_model["y"], GOAL_X - df_model["x"])
    _ang = np.abs(_a - _b)
    df_model["angle"] = np.minimum(_ang, 2 * np.pi - _ang)

    print(df_model[["distance", "angle"]].describe().round(3))
    return (df_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Grouped 70 / 15 / 15 split

    "Competition-grouped" is operationalized as: **grouped by match**
    (no match ever spans two splits) and **allocated within each
    competition**, so train/val/test each contain all four competitions in
    proportion. Match order is shuffled with seed 696 and matches are
    assigned by cumulative shot count. The test split is locked — nothing
    below touches it.
    """)
    return


@app.cell
def _(RNG, SPLIT_FRACS, df_model):
    def assign_splits(frame):
        frame = frame.copy()
        frame["split"] = ""
        tr_frac, va_frac, _te_frac = SPLIT_FRACS
        for _comp, grp in frame.groupby("competition"):
            counts = grp.groupby("match_id").size()
            mids = counts.index.to_numpy()
            mids = mids[RNG.permutation(len(mids))]
            cum = counts.loc[mids].cumsum() / counts.sum()
            split_of = {}
            for mid, frac in cum.items():
                if frac <= tr_frac:
                    split_of[mid] = "train"
                elif frac <= tr_frac + va_frac:
                    split_of[mid] = "val"
                else:
                    split_of[mid] = "test"
            frame.loc[frame["competition"] == _comp, "split"] = frame.loc[
                frame["competition"] == _comp, "match_id"
            ].map(split_of)
        return frame

    df_split = assign_splits(df_model)

    # sanity: no match in more than one split
    assert (df_split.groupby("match_id")["split"].nunique() == 1).all()

    split_table = (
        df_split.groupby(["split", "competition"])["is_goal"]
        .agg(shots="size", goal_rate="mean")
        .round(4)
    )
    print(split_table)
    print(
        df_split.groupby("split")["is_goal"]
        .agg(shots="size", goal_rate="mean")
        .round(4)
    )
    return (df_split,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Base rate + logistic-regression baseline

    Two reference points, both evaluated on **validation only**:

    1. **Base rate** — predict the train goal rate for every shot. Any
       model that can't beat this clearly has a bug.
    2. **LR baseline** — standardized distance + angle into a
       logistic regression. This is the number the Week 5 feature
       families have to move.

    StatsBomb's own xG is scored on the same validation rows as an
    **external benchmark** (it is never a feature).
    """)
    return


@app.cell
def _(df_split, np):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    BASELINE_FEATURES = ["distance", "angle"]

    train_df = df_split[df_split["split"] == "train"]
    val_df = df_split[df_split["split"] == "val"]

    base_rate = train_df["is_goal"].mean()
    y_val = val_df["is_goal"].to_numpy()
    p_base = np.full(len(val_df), base_rate)

    lr_pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000)),
        ]
    )
    lr_pipe.fit(train_df[BASELINE_FEATURES], train_df["is_goal"])
    p_lr = lr_pipe.predict_proba(val_df[BASELINE_FEATURES])[:, 1]

    # external benchmark on the same rows (drop the rare missing xG values)
    sb_mask = val_df["benchmark_sb_xg"].notna().to_numpy()
    p_sb = val_df["benchmark_sb_xg"].to_numpy(dtype=float)[sb_mask]
    y_sb = y_val[sb_mask]

    print(f"train n = {len(train_df):,}  (goal rate {base_rate:.4f})")
    print(f"val   n = {len(val_df):,}  (goal rate {y_val.mean():.4f})")
    _coefs = dict(
        zip(BASELINE_FEATURES, lr_pipe.named_steps["lr"].coef_[0].round(4))
    )
    print(f"LR coefficients (standardized): {_coefs}")
    return (
        brier_score_loss,
        log_loss,
        p_base,
        p_lr,
        p_sb,
        roc_auc_score,
        val_df,
        y_sb,
        y_val,
    )


@app.cell
def _(
    brier_score_loss,
    log_loss,
    np,
    p_base,
    p_lr,
    p_sb,
    pd,
    roc_auc_score,
    y_sb,
    y_val,
):
    def ece(y_true, p_pred, n_bins=10):
        """Expected calibration error, 10 equal-frequency bins."""
        y_true = np.asarray(y_true, dtype=float)
        p_pred = np.asarray(p_pred, dtype=float)
        order = np.argsort(p_pred)
        y_sorted, p_sorted = y_true[order], p_pred[order]
        total = len(p_sorted)
        err = 0.0
        for idx in np.array_split(np.arange(total), n_bins):
            if len(idx) == 0:
                continue
            err += (len(idx) / total) * abs(
                p_sorted[idx].mean() - y_sorted[idx].mean()
            )
        return err

    def score_row(name, y_true, p_pred, with_auc=True):
        p_clip = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
        return {
            "model": name,
            "n": len(p_clip),
            "brier": brier_score_loss(y_true, p_clip),
            "log_loss": log_loss(y_true, p_clip, labels=[0, 1]),
            "ece_10bin": ece(y_true, p_clip),
            "auc": roc_auc_score(y_true, p_clip)
            if with_auc and len(np.unique(p_clip)) > 1
            else np.nan,
        }

    metrics_val = pd.DataFrame(
        [
            score_row("base rate (constant)", y_val, p_base, with_auc=False),
            score_row("LR baseline (distance + angle)", y_val, p_lr),
            score_row("StatsBomb xG (external benchmark)", y_sb, p_sb),
        ]
    ).set_index("model")

    print(metrics_val.round(4))
    return (metrics_val,)


@app.cell
def _(FIG_DIR, np, p_lr, p_sb, pd, y_sb, y_val):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def reliability_points(y_true, p_pred, n_bins=10):
        frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p_pred)})
        frame["bin"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop")
        agg = frame.groupby("bin", observed=True).agg(
            p_mean=("p", "mean"), y_mean=("y", "mean")
        )
        return agg["p_mean"].to_numpy(), agg["y_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#666666", label="perfect calibration")
    for label, (yy, pp), color in [
        ("LR baseline", (y_val, p_lr), "#a3e635"),   # lime
        ("StatsBomb xG", (y_sb, p_sb), "#ef4444"),   # red
    ]:
        px, py = reliability_points(yy, pp)
        ax.plot(px, py, marker="o", lw=1.6, color=color, label=label)
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.tick_params(colors="#dddddd")
    ax.set_xlabel("mean predicted probability", color="#dddddd")
    ax.set_ylabel("observed goal rate", color="#dddddd")
    ax.set_title(
        "Reliability curve — validation split (10 quantile bins)",
        color="#ffffff",
    )
    ax.legend(facecolor="#0a0a0a", labelcolor="#dddddd", edgecolor="#444444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "week4_reliability.png", dpi=200)
    print(f"saved {FIG_DIR / 'week4_reliability.png'}")
    fig
    return


@app.cell
def _(df_split, metrics_val, val_df):
    from pathlib import Path as _Path

    _m = metrics_val.round(4)
    _lines = [
        "# Week 4 numbers (validation split, test locked)",
        "",
        f"- corpus after penalty exclusion: {len(df_split):,} shots "
        f"({df_split['match_id'].nunique():,} matches, "
        f"{df_split['competition'].nunique()} competitions)",
        f"- split sizes: "
        f"train {sum(df_split['split'] == 'train'):,} / "
        f"val {sum(df_split['split'] == 'val'):,} / "
        f"test {sum(df_split['split'] == 'test'):,} (locked)",
        f"- validation goal rate: {val_df['is_goal'].mean():.4f}",
        "",
    ]
    for _name, _row in _m.iterrows():
        _lines.append(
            f"- {_name}: Brier {_row['brier']:.4f} | log loss {_row['log_loss']:.4f}"
            f" | ECE {_row['ece_10bin']:.4f} | AUC {_row['auc']:.4f}"
        )
    _Path("week4_numbers.md").write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print("\n".join(_lines))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Read-out & next steps

    **How to read this week's table.** The LR baseline should sit clearly
    below the base rate on Brier/log loss and, being a well-specified
    low-dimensional GLM, should already be near-calibrated (small ECE).
    Expect StatsBomb's xG to beat the two-feature baseline on Brier — it
    sees body part, technique, and the freeze frame. That gap is exactly
    the budget the Week 5 feature families spend down.

    **Week 5 plan (stepwise family additions, one at a time, on top of
    this baseline):**

    1. body part + technique
    2. play pattern + context flags (first_time, one_on_one, open_goal,
       follows_dribble, aerial_won)
    3. pressure + freeze-frame (under_pressure, gk_dist, opp_in_cone,
       opp_within_5)

    Each family reports ΔBrier / ΔECE on validation; reliability curves
    per step decide whether Platt / isotonic recalibration is warranted.
    """)
    return


if __name__ == "__main__":
    app.run()
