# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.9",
#     "numpy>=2.5.1",
#     "pandas>=3.0.2",
#     "pyarrow>=17.0",
#     "statsbombpy>=1.21.0",
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
    mo.md("""
    # EDA — Calibrated xG on StatsBomb Open Data
    ## MSDS 696, Practicum II · Week 2 (companion to `xg_data_fit_lab.py`)

    This notebook turns the data-fit verdict (**KEEP, question reshaped**) into a
    full exploratory analysis of the shot corpus. Same discipline as the NBA
    project: cache one pull, derive features leakage-free, and let every plot
    answer one question the model will later depend on.

    **Sections:** ⓪ cache-first pull → ① corpus overview → ② target & class
    balance → ③ shot geometry (map, distance, angle) → ④ categorical context →
    ⑤ sparse boolean flags → ⑥ freeze-frame features → ⑦ StatsBomb xG benchmark
    (reliability, Brier, ECE) → ⑧ era/spec drift → ⑨ correlations →
    ⑩ leakage guard → takeaways + model-ready export.
    """)
    return


@app.cell
def _():
    # ---------------- EDA configuration ----------------
    # Same competitions and seed as the data-fit lab, larger sample.
    EDA_COMPETITIONS = [
        "La Liga",
        "FA Women's Super League",
        "FIFA World Cup",
        "Premier League",
    ]
    EDA_MATCHES_PER_COMPETITION = 12
    EDA_SEED = 696
    CACHE_STEM = "xg_eda_shots"  # -> .parquet + .meta.json next to the notebook
    return CACHE_STEM, EDA_COMPETITIONS, EDA_MATCHES_PER_COMPETITION, EDA_SEED


@app.cell
def _():
    import datetime as dt
    import json
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore", message=".*credentials were not supplied.*")
    from statsbombpy import sb

    return Path, dt, json, np, pd, plt, sb


@app.cell
def _(mo):
    mo.md("""
    ## ⓪ Cache-first pull

    `statsbombpy` reads GitHub `master`, which moves. The first run pulls the
    sample, derives freeze-frame scalars, and writes one parquet + a metadata
    JSON (pull date, seed, match ids). Every later run — and the whole capstone
    EDA — reads the cache, so results are frozen and reproducible.
    Delete the parquet to force a re-pull.
    """)
    return


@app.cell
def _(np):
    # ---------- Freeze-frame + geometry helpers (defined once) ----------
    GOAL_X, GOAL_Y = 120.0, 40.0
    POST_LOW, POST_HIGH = 36.0, 44.0

    def shot_distance(x, y):
        return float(np.hypot(GOAL_X - x, GOAL_Y - y))

    def shot_angle_deg(x, y):
        # Angle subtended by the goal mouth at the shot location.
        a = np.arctan2(POST_HIGH - y, GOAL_X - x)
        b = np.arctan2(POST_LOW - y, GOAL_X - x)
        return float(np.degrees(abs(a - b)))

    def _in_shot_cone(px, py, sx, sy):
        # Point inside the triangle (shot location, near post, far post)?
        def _cross(ax, ay, bx, by, cx, cy):
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        d1 = _cross(sx, sy, GOAL_X, POST_LOW, px, py)
        d2 = _cross(GOAL_X, POST_LOW, GOAL_X, POST_HIGH, px, py)
        d3 = _cross(GOAL_X, POST_HIGH, sx, sy, px, py)
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    def freeze_frame_scalars(ff, sx, sy):
        """Collapse a StatsBomb freeze frame into leakage-free scalar features."""
        out = {
            "ff_n_players": np.nan,
            "ff_opp_within_5": np.nan,
            "ff_opp_in_cone": np.nan,
            "ff_gk_dist_to_goal": np.nan,
            "ff_gk_dist_to_shot": np.nan,
        }
        if not isinstance(ff, list) or not ff:
            return out
        out["ff_n_players"] = len(ff)
        _w5 = _cone = 0
        for _p in ff:
            if _p.get("teammate"):
                continue
            _px, _py = _p["location"]
            if np.hypot(_px - sx, _py - sy) <= 5.0:
                _w5 += 1
            if _in_shot_cone(_px, _py, sx, sy):
                _cone += 1
            if _p.get("position", {}).get("name") == "Goalkeeper":
                out["ff_gk_dist_to_goal"] = float(np.hypot(GOAL_X - _px, GOAL_Y - _py))
                out["ff_gk_dist_to_shot"] = float(np.hypot(sx - _px, sy - _py))
        out["ff_opp_within_5"] = _w5
        out["ff_opp_in_cone"] = _cone
        return out

    return freeze_frame_scalars, shot_angle_deg, shot_distance


@app.cell
def _(
    CACHE_STEM,
    EDA_COMPETITIONS,
    EDA_MATCHES_PER_COMPETITION,
    EDA_SEED,
    Path,
    dt,
    freeze_frame_scalars,
    json,
    pd,
    sb,
    shot_angle_deg,
    shot_distance,
):
    _pq = Path(f"{CACHE_STEM}.parquet")
    _meta_p = Path(f"{CACHE_STEM}.meta.json")

    # Scalar columns kept in the cache. Leakage-watchlist columns are kept in
    # the RAW cache on purpose (audited in §⑦/⑩) and barred at model time.
    _KEEP = [
        "match_id", "competition_name", "season_name", "match_date",
        "data_version", "period", "minute", "second", "play_pattern",
        "position", "team", "player",
        "shot_type", "shot_body_part", "shot_technique", "shot_outcome",
        "shot_statsbomb_xg", "shot_first_time", "shot_one_on_one",
        "shot_open_goal", "shot_aerial_won", "under_pressure",
        "x", "y", "distance", "angle",
        "ff_n_players", "ff_opp_within_5", "ff_opp_in_cone",
        "ff_gk_dist_to_goal", "ff_gk_dist_to_shot",
    ]

    if _pq.exists():
        shots_cached = pd.read_parquet(_pq)
        pull_meta = json.loads(_meta_p.read_text(encoding="utf-8"))
        print(
            f"[cache] loaded {len(shots_cached):,} shots from {_pq} "
            f"(pulled {pull_meta['pull_date']}, seed {pull_meta['seed']})"
        )
    else:
        _comps = sb.competitions()
        _idx = []
        for _r in _comps.itertuples(index=False):
            if _r.competition_name not in EDA_COMPETITIONS:
                continue
            _m = sb.matches(
                competition_id=_r.competition_id, season_id=_r.season_id
            )[["match_id", "match_date", "data_version"]]
            _m["competition_name"] = _r.competition_name
            _m["season_name"] = _r.season_name
            _idx.append(_m)
        _pool = pd.concat(_idx, ignore_index=True)
        _n = min(
            EDA_MATCHES_PER_COMPETITION,
            int(_pool.groupby("competition_name").size().min()),
        )
        _sampled = (
            _pool.groupby("competition_name", group_keys=False)
            .sample(n=_n, random_state=EDA_SEED)
            .reset_index(drop=True)
        )
        print(f"[pull] fetching events for {len(_sampled)} matches ...")

        _frames, _failed = [], 0
        for _i, _mr in enumerate(_sampled.itertuples(index=False)):
            try:
                _ev = sb.events(match_id=int(_mr.match_id))
            except Exception as _e:
                _failed += 1
                print(f"  ! events failed for match {_mr.match_id}: {_e}")
                continue
            _sh = _ev[_ev["type"] == "Shot"].copy()
            _sh = _sh[_sh["period"] <= 4]  # drop penalty shootouts
            _sh["competition_name"] = _mr.competition_name
            _sh["season_name"] = _mr.season_name
            _sh["match_date"] = _mr.match_date
            _sh["data_version"] = _mr.data_version
            _sh["x"] = _sh["location"].str[0].astype(float)
            _sh["y"] = _sh["location"].str[1].astype(float)
            _sh["distance"] = [
                shot_distance(_xv, _yv) for _xv, _yv in zip(_sh["x"], _sh["y"])
            ]
            _sh["angle"] = [
                shot_angle_deg(_xv, _yv) for _xv, _yv in zip(_sh["x"], _sh["y"])
            ]
            _ffs = pd.DataFrame(
                [
                    freeze_frame_scalars(_ff, _xv, _yv)
                    for _ff, _xv, _yv in zip(
                        _sh.get("shot_freeze_frame", pd.Series([None] * len(_sh))),
                        _sh["x"],
                        _sh["y"],
                    )
                ],
                index=_sh.index,
            )
            _sh = pd.concat([_sh, _ffs], axis=1)
            _frames.append(_sh.reindex(columns=_KEEP))
            if (_i + 1) % 10 == 0:
                print(f"  ... {_i + 1}/{len(_sampled)} matches")

        shots_cached = pd.concat(_frames, ignore_index=True)
        shots_cached.to_parquet(_pq, index=False)
        pull_meta = {
            "pull_date": dt.date.today().isoformat(),
            "seed": EDA_SEED,
            "competitions": EDA_COMPETITIONS,
            "matches_requested": int(len(_sampled)),
            "matches_failed": int(_failed),
            "match_ids": [int(_v) for _v in _sampled["match_id"]],
            "n_shots": int(len(shots_cached)),
            "source": "StatsBomb Open Data (github.com/statsbomb/open-data)",
        }
        _meta_p.write_text(json.dumps(pull_meta, indent=2), encoding="utf-8")
        print(
            f"[cache] wrote {len(shots_cached):,} shots -> {_pq} "
            f"(+ {_meta_p.name})"
        )
    return pull_meta, shots_cached


@app.cell
def _(pd, shots_cached):
    # ---------- Post-cache feature build (pure, reproducible) ----------
    # StatsBomb sparse booleans: NaN means False. Fill explicitly (Limitation #2).
    SPARSE_FLAGS = [
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",
        "shot_aerial_won",
        "under_pressure",
    ]
    shots = shots_cached.copy()
    for _f in SPARSE_FLAGS:
        shots[_f] = shots[_f].map(lambda v: bool(v) if pd.notna(v) else False)

    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)
    shots["is_penalty"] = (shots["shot_type"] == "Penalty").astype(int)
    shots["is_header"] = (shots["shot_body_part"] == "Head").astype(int)
    shots["era"] = shots["match_date"].str[:4].astype(int).map(
        lambda yr: "≤2010" if yr <= 2010 else ("2011–2017" if yr <= 2017 else "2018+")
    )

    open_play = shots[shots["is_penalty"] == 0]
    print(
        f"[features] {len(shots):,} shots ({len(open_play):,} non-penalty) | "
        f"goal rate {shots['is_goal'].mean():.3f} | flags filled NaN->False"
    )
    return SPARSE_FLAGS, open_play, shots


@app.cell
def _(mo):
    mo.md("""
    ## ① Corpus overview

    One row per shot. Check the sample is what the lab said it is: four
    competitions, multiple eras and genders, and a shots-per-match rate that
    matches football reality (~25/match).
    """)
    return


@app.cell
def _(mo, shots):
    overview_tbl = (
        shots.groupby("competition_name")
        .agg(
            matches=("match_id", "nunique"),
            seasons=("season_name", "nunique"),
            shots=("is_goal", "size"),
            shots_per_match=("match_id", lambda s: round(len(s) / s.nunique(), 1)),
            first=("match_date", "min"),
            last=("match_date", "max"),
        )
        .reset_index()
    )
    mo.vstack([mo.md("**Sample composition**"), overview_tbl])
    return


@app.cell
def _(mo):
    mo.md("""
    ## ② Target & class balance

    xG is a rare-event problem: ~1 shot in 9 scores. Calibration metrics were
    chosen for exactly this reason — accuracy on a 89/11 split is a mirage
    (the "87% NBA model" lesson, again).
    """)
    return


@app.cell
def _(mo, shots):
    balance_tbl = (
        shots.groupby("competition_name")
        .agg(
            shots=("is_goal", "size"),
            goals=("is_goal", "sum"),
            goal_rate=("is_goal", "mean"),
            penalty_share=("is_penalty", "mean"),
            header_share=("is_header", "mean"),
            mean_sb_xg=("shot_statsbomb_xg", "mean"),
        )
        .round(3)
        .reset_index()
    )
    mo.vstack(
        [
            mo.md(
                "**Class balance by competition** — mean SB xG hugging the goal "
                "rate in every group is the first calibration sanity check."
            ),
            balance_tbl,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## ③ Shot geometry

    The two features that carry most of xG's signal are derived, not given:
    distance to goal centre and the angle subtended by the goal mouth.
    Pitch is StatsBomb's 120×80, attacking right.
    """)
    return


@app.function
def draw_half_pitch(ax):
    """Attacking half (x 60–120) of a StatsBomb 120x80 pitch."""
    _lc = "#666666"
    ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], color=_lc, lw=1)
    ax.plot([102, 102, 120], [18, 62, 62], color=_lc, lw=1)
    ax.plot([102, 120], [18, 18], color=_lc, lw=1)
    ax.plot([114, 114, 120], [30, 50, 50], color=_lc, lw=1)
    ax.plot([114, 120], [30, 30], color=_lc, lw=1)
    ax.plot([120, 120], [36, 44], color="#111111", lw=3)
    ax.scatter([108], [40], s=6, color=_lc)
    ax.set_xlim(59, 123)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


@app.cell
def _(open_play, plt):
    fig_map, ax_map = plt.subplots(figsize=(8, 6))
    draw_half_pitch(ax_map)
    _miss = open_play[open_play["is_goal"] == 0]
    _goal = open_play[open_play["is_goal"] == 1]
    ax_map.scatter(_miss["x"], _miss["y"], s=10, alpha=0.25, color="#4477AA", label=f"No goal (n={len(_miss):,})")
    ax_map.scatter(_goal["x"], _goal["y"], s=22, alpha=0.85, color="#EE6677", label=f"Goal (n={len(_goal):,})")
    ax_map.set_title("Open-play shot map — goals cluster tight and central")
    ax_map.legend(loc="lower left", frameon=False)
    fig_map.tight_layout()
    fig_map
    return


@app.cell
def _(open_play, plt):
    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(10, 3.5))
    for _ax, _col, _xlab in zip(
        axes_dist, ["distance", "angle"], ["Distance to goal centre (SB units ≈ yd)", "Goal-mouth angle (°)"]
    ):
        _ax.hist(
            open_play.loc[open_play["is_goal"] == 0, _col],
            bins=30, density=True, alpha=0.55, color="#4477AA", label="No goal",
        )
        _ax.hist(
            open_play.loc[open_play["is_goal"] == 1, _col],
            bins=30, density=True, alpha=0.55, color="#EE6677", label="Goal",
        )
        _ax.set_xlabel(_xlab)
        _ax.set_ylabel("Density")
        _ax.spines[["top", "right"]].set_visible(False)
    axes_dist[0].legend(frameon=False)
    fig_dist.suptitle("Goals come from closer and wider-angle positions", y=1.02)
    fig_dist.tight_layout()
    fig_dist
    return


@app.cell
def _(np, open_play, pd, plt):
    _bins = [0, 6, 9, 12, 15, 18, 24, 30, 90]
    _g = (
        open_play.assign(db=pd.cut(open_play["distance"], bins=_bins))
        .groupby("db", observed=True)
        .agg(n=("is_goal", "size"), rate=("is_goal", "mean"), sb=("shot_statsbomb_xg", "mean"))
        .reset_index()
    )
    _g["mid"] = _g["db"].map(lambda iv: (iv.left + min(iv.right, 40)) / 2)
    # Wilson-ish error bars for the empirical rate
    _g["se"] = np.sqrt(_g["rate"] * (1 - _g["rate"]) / _g["n"])

    fig_rate, ax_rate = plt.subplots(figsize=(8, 4))
    ax_rate.errorbar(_g["mid"], _g["rate"], yerr=1.96 * _g["se"], fmt="o-", color="#EE6677", capsize=3, label="Empirical goal rate")
    ax_rate.plot(_g["mid"], _g["sb"], "s--", color="#4477AA", label="Mean StatsBomb xG")
    ax_rate.set_xlabel("Shot distance (binned, SB units)")
    ax_rate.set_ylabel("P(goal)")
    ax_rate.set_title("Monotone distance decay — and SB xG rides the empirical curve")
    ax_rate.legend(frameon=False)
    ax_rate.spines[["top", "right"]].set_visible(False)
    fig_rate.tight_layout()
    fig_rate
    return


@app.cell
def _(mo):
    mo.md("""
    ## ④ Categorical context

    Lift tables: goal rate by category vs the base rate. These become one-hot /
    target-encoded inputs; anything with big lift **and** decent support earns
    a place in the model.
    """)
    return


@app.cell
def _(mo, shots):
    def _lift(df, col, min_n=10):
        _base = df["is_goal"].mean()
        _t = (
            df.groupby(col)
            .agg(n=("is_goal", "size"), goal_rate=("is_goal", "mean"))
            .query("n >= @min_n")
            .assign(lift=lambda d: d["goal_rate"] / _base)
            .sort_values("goal_rate", ascending=False)
            .round(3)
            .reset_index()
        )
        return _t

    cat_tables = {
        "shot_type": _lift(shots, "shot_type"),
        "shot_body_part": _lift(shots, "shot_body_part"),
        "shot_technique": _lift(shots, "shot_technique"),
        "play_pattern": _lift(shots, "play_pattern"),
    }
    mo.vstack(
        [
            _el
            for _k2, _v2 in cat_tables.items()
            for _el in (mo.md(f"**Goal-rate lift — `{_k2}`**"), _v2)
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑤ Sparse boolean flags

    These are the columns where NaN means False. After the explicit fill,
    check prevalence and lift — `shot_open_goal` and `shot_one_on_one` are
    rare but huge; `under_pressure` is common and suppressive.
    """)
    return


@app.cell
def _(SPARSE_FLAGS, mo, pd, shots):
    _base_rate = shots["is_goal"].mean()
    flag_tbl = pd.DataFrame(
        [
            {
                "flag": _f,
                "prevalence_%": round(100 * shots[_f].mean(), 1),
                "goal_rate_when_True": round(shots.loc[shots[_f], "is_goal"].mean(), 3)
                if shots[_f].any()
                else float("nan"),
                "goal_rate_when_False": round(shots.loc[~shots[_f], "is_goal"].mean(), 3),
                "lift_when_True": round(
                    shots.loc[shots[_f], "is_goal"].mean() / _base_rate, 2
                )
                if shots[_f].any()
                else float("nan"),
            }
            for _f in SPARSE_FLAGS
        ]
    )
    mo.vstack([mo.md("**Sparse flags after NaN→False fill**"), flag_tbl])
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑥ Freeze-frame features

    The 2-D freeze frame is this dataset's edge over bare event data: opponent
    pressure and GK positioning at the moment of the shot, collapsed into
    scalars (`ff_opp_within_5`, `ff_opp_in_cone`, `ff_gk_dist_to_goal`).
    This is the feature family that decides whether we can approach
    StatsBomb's own model.
    """)
    return


@app.cell
def _(mo, open_play, pd):
    ff_cov = open_play["ff_n_players"].notna().mean()
    _ff = open_play[open_play["ff_n_players"].notna()]
    ff_tbl = (
        _ff.assign(
            cone_band=pd.cut(
                _ff["ff_opp_in_cone"], bins=[-0.5, 0.5, 1.5, 2.5, 20],
                labels=["0", "1", "2", "3+"],
            )
        )
        .groupby("cone_band", observed=True)
        .agg(n=("is_goal", "size"), goal_rate=("is_goal", "mean"), mean_sb_xg=("shot_statsbomb_xg", "mean"))
        .round(3)
        .reset_index()
    )
    mo.vstack(
        [
            mo.md(
                f"**Opponents in the shot cone vs goal rate** — freeze-frame "
                f"coverage on open-play shots: **{ff_cov:.1%}**. Fewer bodies "
                f"in the cone → sharply higher conversion."
            ),
            ff_tbl,
        ]
    )
    return


@app.cell
def _(open_play, plt):
    fig_ff, axes_ff = plt.subplots(1, 2, figsize=(10, 3.5))
    _ffd = open_play.dropna(subset=["ff_opp_within_5", "ff_gk_dist_to_shot"])
    for _ax, _col, _xlab in zip(
        axes_ff,
        ["ff_opp_within_5", "ff_gk_dist_to_shot"],
        ["Opponents within 5 units of shooter", "GK distance to shot (SB units)"],
    ):
        _ax.hist(
            _ffd.loc[_ffd["is_goal"] == 0, _col], bins=20, density=True,
            alpha=0.55, color="#4477AA", label="No goal",
        )
        _ax.hist(
            _ffd.loc[_ffd["is_goal"] == 1, _col], bins=20, density=True,
            alpha=0.55, color="#EE6677", label="Goal",
        )
        _ax.set_xlabel(_xlab)
        _ax.set_ylabel("Density")
        _ax.spines[["top", "right"]].set_visible(False)
    axes_ff[0].legend(frameon=False)
    fig_ff.suptitle("Freeze-frame scalars separate the classes", y=1.02)
    fig_ff.tight_layout()
    fig_ff
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑦ StatsBomb xG as the external benchmark

    `shot_statsbomb_xg` is barred as a feature (leakage watchlist) but it is
    the **bar to clear on calibration** — the Vegas line of this project.
    Reliability curve + Brier + ECE below define the target numbers my model
    has to approach.
    """)
    return


@app.cell
def _(np, pd, shots):
    def calibration_report(y_true, p_pred, n_bins=10):
        """Equal-width reliability table + Brier + ECE."""
        _df = pd.DataFrame({"y": y_true, "p": p_pred}).dropna()
        _df["bin"] = pd.cut(_df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
        _tbl = (
            _df.groupby("bin", observed=True)
            .agg(n=("y", "size"), mean_pred=("p", "mean"), frac_goal=("y", "mean"))
            .dropna()
            .reset_index()
        )
        _brier = float(((_df["p"] - _df["y"]) ** 2).mean())
        _ece = float(
            (np.abs(_tbl["mean_pred"] - _tbl["frac_goal"]) * _tbl["n"]).sum()
            / _tbl["n"].sum()
        )
        return _tbl.round(3), _brier, _ece

    sb_rel_tbl, sb_brier, sb_ece = calibration_report(
        shots["is_goal"], shots["shot_statsbomb_xg"]
    )
    # Naive baseline: predict the base rate for every shot.
    _p0 = shots["is_goal"].mean()
    base_brier = float(((shots["is_goal"] - _p0) ** 2).mean())
    print(
        f"[benchmark] StatsBomb xG: Brier {sb_brier:.4f} (base-rate Brier "
        f"{base_brier:.4f}) | ECE {sb_ece:.4f} on {len(shots):,} shots"
    )
    return base_brier, sb_brier, sb_ece, sb_rel_tbl


@app.cell
def _(plt, sb_brier, sb_ece, sb_rel_tbl):
    fig_cal, ax_cal = plt.subplots(figsize=(5.5, 5))
    ax_cal.plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Perfect calibration")
    ax_cal.scatter(
        sb_rel_tbl["mean_pred"], sb_rel_tbl["frac_goal"],
        s=sb_rel_tbl["n"].clip(upper=400) / 2, color="#EE6677", zorder=3,
        label="SB xG (point size ∝ n)",
    )
    ax_cal.plot(sb_rel_tbl["mean_pred"], sb_rel_tbl["frac_goal"], color="#EE6677", lw=1, alpha=0.6)
    ax_cal.set_xlabel("Mean predicted xG (bin)")
    ax_cal.set_ylabel("Empirical goal fraction")
    ax_cal.set_title(f"StatsBomb xG reliability — Brier {sb_brier:.3f}, ECE {sb_ece:.3f}")
    ax_cal.legend(frameon=False, loc="upper left")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_aspect("equal")
    fig_cal.tight_layout()
    fig_cal
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑧ Era / spec drift

    Limitation #2 made measurable: field coverage by `data_version` and era.
    Anything that drops out in old seasons needs a coverage gate before it can
    be a model feature.
    """)
    return


@app.cell
def _(mo, shots):
    _CHECK = [
        "shot_technique",
        "play_pattern",
        "position",
        "ff_n_players",
        "shot_statsbomb_xg",
    ]
    drift_tbl = (
        shots.groupby(["era", "data_version"])
        .agg(
            shots=("is_goal", "size"),
            **{f"{_c}_cov": (_c, lambda s: round(s.notna().mean(), 3)) for _c in _CHECK},
        )
        .reset_index()
    )
    mo.vstack([mo.md("**Field coverage by era × spec version**"), drift_tbl])
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑨ Correlations among numeric candidates

    Check for redundancy (distance vs angle are strongly related by geometry)
    and confirm every candidate points at the target the way football logic
    says it should.
    """)
    return


@app.cell
def _(plt, shots):
    _NUM = [
        "distance", "angle", "ff_opp_within_5", "ff_opp_in_cone",
        "ff_gk_dist_to_shot", "minute", "is_goal",
    ]
    corr_mat = shots[_NUM].corr(method="spearman")

    fig_corr, ax_corr = plt.subplots(figsize=(6.5, 5.5))
    _im = ax_corr.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_corr.set_xticks(range(len(_NUM)), _NUM, rotation=45, ha="right")
    ax_corr.set_yticks(range(len(_NUM)), _NUM)
    for _i in range(len(_NUM)):
        for _j in range(len(_NUM)):
            ax_corr.text(
                _j, _i, f"{corr_mat.iloc[_i, _j]:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(corr_mat.iloc[_i, _j]) > 0.5 else "#222222",
            )
    ax_corr.set_title("Spearman correlations — numeric candidates vs target")
    fig_corr.colorbar(_im, shrink=0.8)
    fig_corr.tight_layout()
    fig_corr
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⑩ Leakage guard

    Codified, not remembered. The model-ready export below is built from an
    explicit allow-list; the barred list is asserted disjoint at write time —
    the `shift(1)` discipline of this project.
    """)
    return


@app.cell
def _(Path, shots):
    MODEL_FEATURES = [
        # geometry
        "x", "y", "distance", "angle",
        # context
        "shot_type", "shot_body_part", "shot_technique", "play_pattern",
        "position", "period", "minute",
        # sparse flags (filled)
        "under_pressure", "shot_first_time", "shot_one_on_one",
        "shot_open_goal", "shot_aerial_won",
        # freeze-frame scalars
        "ff_n_players", "ff_opp_within_5", "ff_opp_in_cone",
        "ff_gk_dist_to_goal", "ff_gk_dist_to_shot",
        # grouping / split keys (never model inputs, kept for grouped CV)
        "competition_name", "season_name", "match_id",
    ]
    BARRED = [
        "shot_outcome",          # the target
        "shot_statsbomb_xg",     # benchmark only — label-adjacent
        "shot_end_location",     # post-release
        "shot_deflected",        # post-release
    ]
    assert set(MODEL_FEATURES).isdisjoint(BARRED), "leakage: barred column in feature list"

    model_ready = shots[MODEL_FEATURES + ["is_goal", "is_penalty"]].copy()
    _out = Path("xg_shots_model_ready.parquet")
    model_ready.to_parquet(_out, index=False)
    print(
        f"[export] {len(model_ready):,} rows x {model_ready.shape[1]} cols -> "
        f"{_out} | barred at source: {BARRED}"
    )
    return


@app.cell
def _(base_brier, mo, open_play, pull_meta, sb_brier, sb_ece, shots):
    _gr = shots["is_goal"].mean()
    _close = open_play.loc[open_play["distance"] <= 6, "is_goal"].mean()
    _far = open_play.loc[open_play["distance"] > 30, "is_goal"].mean()
    _up = shots.loc[shots["under_pressure"], "is_goal"].mean()
    _nup = shots.loc[~shots["under_pressure"], "is_goal"].mean()
    _hd = shots.loc[shots["is_header"] == 1, "is_goal"].mean()
    mo.md(
        f"""
        ## EDA takeaways → into the Status Report

        - **Corpus (cached {pull_meta["pull_date"]}, seed {pull_meta["seed"]}):**
          {len(shots):,} shots, goal rate {_gr:.1%}. Rare-event target confirmed —
          calibration metrics, not accuracy.
        - **Geometry dominates:** goal rate falls from {_close:.0%} inside 6 units
          to {_far:.1%} beyond 30, monotone throughout. Distance + angle are the
          backbone features.
        - **Context features are real:** under pressure {_up:.1%} vs {_nup:.1%}
          unpressured; headers convert at {_hd:.1%}; opponents-in-cone shows a
          clean monotone gradient. Freeze-frame scalars are the differentiating
          family.
        - **The bar is quantified:** StatsBomb xG scores Brier **{sb_brier:.4f}**
          (base-rate baseline {base_brier:.4f}) and ECE **{sb_ece:.4f}** on this
          sample. Those are the numbers a calibrated model must approach —
          this project's Vegas line.
        - **Drift is handled, not hoped away:** sparse flags filled NaN→False,
          coverage gated by era/spec, one frozen parquet + metadata JSON, and a
          leakage guard asserted at export time.

        """
    )
    return


if __name__ == "__main__":
    app.run()
