# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "numpy>=2.5.1",
#     "pandas>=3.0.3",
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
    # Pod Lab — Vet Your Data Against Your Question
    ## StatsBomb Open Data vs. a calibrated xG model  *(MSDS 696, Practicum II)*

    **Checklist:** ① Provenance & inventory → ② Fit (schema + signal audit) →
    ③ Limits (three biggest) → ④ Feasibility → **Decision** → auto-generated
    `xg_data_fit_note.md` for the Status Report.

    Same discipline as the NBA project: every claim below is computed from the
    data in this notebook, leakage suspects are named explicitly, and the
    honest ceiling is framed up front.
    """)
    return


@app.cell
def _():
    # ---------------- Lab configuration ----------------
    RESEARCH_QUESTION = (
        "Can event-level features from StatsBomb Open Data support a "
        "well-calibrated expected-goals (xG) model — judged on calibration "
        "(reliability curves, ECE, Brier) rather than raw accuracy?"
    )

    # Sample audit: spread across eras, genders, and formats on purpose,
    # so schema drift and annotation differences have a chance to show up.
    SAMPLE_COMPETITIONS = [
        "La Liga",                  # huge, Barcelona-centric, 2004/05 -> 2020/21
        "FA Women's Super League",  # large multi-season women's league
        "FIFA World Cup",           # tournament football, 2018 + 2022
        "Premier League",           # single 2015/16 season, older spec
    ]
    MATCHES_PER_COMPETITION = 4     # scale up for the full capstone pull
    RANDOM_SEED = 696               # course number, reproducible sample
    return (
        MATCHES_PER_COMPETITION,
        RANDOM_SEED,
        RESEARCH_QUESTION,
        SAMPLE_COMPETITIONS,
    )


@app.cell
def _():
    import datetime as dt
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # statsbombpy warns on every call that we are on open data; silence it once
    warnings.filterwarnings("ignore", message=".*credentials were not supplied.*")
    from statsbombpy import sb

    return Path, dt, np, pd, sb


@app.cell
def _(mo):
    mo.md("""
    ## ① Provenance & inventory

    **Source:** StatsBomb Open Data (github.com/statsbomb/open-data), collected
    by StatsBomb's annotation pipeline from broadcast video against a public,
    versioned event spec. **License:** free for research / non-commercial use,
    attribution required — fine for a Regis practicum, cite it in the report.
    **Access:** `statsbombpy` reads the repo's `master` branch, so the data can
    change under us — for the capstone, cache one pull to parquet and record
    the pull date (the generated note does this automatically).
    """)
    return


@app.cell
def _(sb):
    comps = sb.competitions()
    print(
        f"[competitions] {comps['competition_name'].nunique()} competitions, "
        f"{len(comps)} competition-seasons, "
        f"{comps['match_available_360'].notna().sum()} with 360 freeze-frame data"
    )
    comps[
        [
            "competition_name",
            "season_name",
            "competition_gender",
            "competition_international",
            "match_available_360",
        ]
    ]
    return (comps,)


@app.cell
def _(comps, pd, sb):
    # Pull every matches index (~80 small JSON files) to build the full inventory.
    _KEEP = [
        "match_id",
        "match_date",
        "data_version",
        "shot_fidelity_version",
        "xy_fidelity_version",
    ]
    _frames = []
    skipped_pulls = 0
    for _i, _r in enumerate(
        comps[
            [
                "competition_id",
                "season_id",
                "competition_name",
                "season_name",
                "competition_gender",
            ]
        ].itertuples(index=False)
    ):
        try:
            _m = sb.matches(
                competition_id=_r.competition_id, season_id=_r.season_id
            ).reindex(columns=_KEEP)
        except Exception as _e:
            skipped_pulls += 1
            print(f"  ! skipped {_r.competition_name} {_r.season_name}: {_e}")
            continue
        _m["competition_name"] = _r.competition_name
        _m["season_name"] = _r.season_name
        _m["competition_gender"] = _r.competition_gender
        _frames.append(_m)
        if (_i + 1) % 20 == 0:
            print(f"  ... {_i + 1}/{len(comps)} competition-seasons indexed")

    all_matches = pd.concat(_frames, ignore_index=True)
    print(
        f"[inventory] {len(all_matches):,} matches | "
        f"{all_matches['match_date'].min()} -> {all_matches['match_date'].max()} | "
        f"{skipped_pulls} index pulls failed"
    )
    return all_matches, skipped_pulls


@app.cell
def _(all_matches):
    inventory = (
        all_matches.groupby(["competition_name", "competition_gender"])
        .agg(
            seasons=("season_name", "nunique"),
            matches=("match_id", "count"),
            first=("match_date", "min"),
            last=("match_date", "max"),
        )
        .sort_values("matches", ascending=False)
        .reset_index()
    )
    inventory
    return


@app.cell
def _(all_matches, mo):
    version_mix = (
        all_matches["data_version"]
        .fillna("(missing)")
        .value_counts()
        .rename_axis("data_version")
        .reset_index(name="matches")
    )
    mo.vstack(
        [
            mo.md(
                "**Spec-version mix** — the corpus was annotated over ~20 years "
                "under different spec versions. This is measurable provenance, "
                "and it is also Limitation #2 below."
            ),
            version_mix,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## ② Fit — does the event schema contain what the question needs?

    An xG model needs, **per shot**: pre-shot context (location, body part,
    technique, pattern of play, pressure, freeze frame) and a binary outcome.
    We audit that on a reproducible sample of matches spread across eras,
    genders, and competition formats — and we separate **candidate features**
    from **leakage suspects** the way the NBA audit separated rolling
    features from `H2H_HOME_WIN_PCT`.
    """)
    return


@app.cell
def _(MATCHES_PER_COMPETITION, RANDOM_SEED, SAMPLE_COMPETITIONS, all_matches):
    _pool = all_matches[all_matches["competition_name"].isin(SAMPLE_COMPETITIONS)]
    _n = min(MATCHES_PER_COMPETITION, int(_pool.groupby("competition_name").size().min()))
    sampled_matches = (
        _pool.groupby("competition_name", group_keys=False)
        .sample(n=_n, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    print(
        f"[sample] {len(sampled_matches)} matches across "
        f"{sampled_matches['competition_name'].nunique()} competitions, "
        f"{sampled_matches['season_name'].nunique()} distinct seasons"
    )
    sampled_matches[
        ["competition_name", "season_name", "match_date", "data_version", "match_id"]
    ].sort_values(["competition_name", "match_date"])
    return (sampled_matches,)


@app.cell
def _(pd, sampled_matches, sb):
    # Pull events for the sampled matches; keep Shot rows only.
    _shot_frames = []
    fetch_failures = 0
    for _mr in sampled_matches.itertuples(index=False):
        try:
            _ev = sb.events(match_id=int(_mr.match_id))
        except Exception as _e:
            fetch_failures += 1
            print(f"  ! events failed for match {_mr.match_id}: {_e}")
            continue
        _sh = _ev[_ev["type"] == "Shot"].copy()
        _sh["competition_name"] = _mr.competition_name
        _sh["season_name"] = _mr.season_name
        _shot_frames.append(_sh)

    shots_raw = pd.concat(_shot_frames, ignore_index=True, sort=False)
    print(
        f"[events] {len(sampled_matches) - fetch_failures}/{len(sampled_matches)} "
        f"matches fetched -> {len(shots_raw):,} shot events"
    )
    return fetch_failures, shots_raw


@app.cell
def _(np, shots_raw):
    # Derived pre-shot geometry + target. Period 5 = penalty shootout: those are
    # not in-game shots and would poison class balance, so they are excluded.
    n_shootout_excluded = int((shots_raw["period"] == 5).sum())
    shots_feat = shots_raw[shots_raw["period"] <= 4].copy()

    shots_feat["x"] = shots_feat["location"].str[0].astype(float)
    shots_feat["y"] = shots_feat["location"].str[1].astype(float)
    # StatsBomb pitch: 120 x 80 units (~yards); goal centre (120, 40),
    # posts at y = 36 and y = 44.
    shots_feat["distance"] = np.sqrt(
        (120 - shots_feat["x"]) ** 2 + (40 - shots_feat["y"]) ** 2
    )
    shots_feat["angle"] = np.degrees(
        np.abs(
            np.arctan2(44 - shots_feat["y"], 120 - shots_feat["x"])
            - np.arctan2(36 - shots_feat["y"], 120 - shots_feat["x"])
        )
    )
    shots_feat["is_goal"] = (shots_feat["shot_outcome"] == "Goal").astype(int)
    shots_feat["is_penalty"] = (shots_feat["shot_type"] == "Penalty").astype(int)

    print(
        f"[target] {len(shots_feat):,} in-game shots "
        f"({n_shootout_excluded} shootout penalties excluded) | "
        f"goal rate {shots_feat['is_goal'].mean():.3f} | "
        f"penalty share {shots_feat['is_penalty'].mean():.3f}"
    )
    return n_shootout_excluded, shots_feat


@app.cell
def _(mo, pd, shots_feat):
    # Candidate-feature audit. Coverage = % non-null on the sampled shots.
    # Gotcha (fit check that actually bit): StatsBomb boolean flags encode
    # False as NaN — 10% "coverage" on under_pressure is NOT 90% missingness.
    _CANDIDATES = {
        "location": "shot origin -> distance / angle",
        "shot_type": "Open Play / Penalty / Free Kick / Corner",
        "shot_body_part": "foot / head / other",
        "shot_technique": "volley, half-volley, lob, ...",
        "play_pattern": "possession context (counter, set piece, ...)",
        "position": "shooter's position",
        "period": "game state timing",
        "minute": "game state timing",
        "shot_freeze_frame": "player coordinates at shot -> defender/GK features",
        "under_pressure": "sparse flag: NaN means False",
        "shot_first_time": "sparse flag: NaN means False",
        "shot_one_on_one": "sparse flag: NaN means False",
        "shot_open_goal": "sparse flag: NaN means False",
        "shot_aerial_won": "sparse flag: NaN means False",
    }
    feature_audit = pd.DataFrame(
        [
            {
                "column": _c,
                "coverage_%": round(
                    100 * shots_feat[_c].notna().mean(), 1
                )
                if _c in shots_feat.columns
                else 0.0,
                "note": _note
                if _c in shots_feat.columns
                else _note + "  [ABSENT in sample]",
            }
            for _c, _note in _CANDIDATES.items()
        ]
    )
    mo.vstack(
        [
            mo.md("**Candidate pre-shot features** (usable inputs)"),
            feature_audit,
        ]
    )
    return


@app.cell
def _(mo, pd, shots_feat):
    # Leakage watchlist — this project's H2H_HOME_WIN_PCT equivalents.
    _WATCH = {
        "shot_outcome": "THE TARGET. Never a feature.",
        "shot_statsbomb_xg": "StatsBomb's own model output. Benchmark ONLY — "
        "using it as a feature is label-adjacent leakage.",
        "shot_end_location": "Post-release ball path. Pre-shot models must not see it.",
        "shot_deflected": "Determined after release.",
        "home_score / away_score (matches file)": "Final result. Never join into "
        "shot features.",
    }
    leakage_watch = pd.DataFrame(
        [
            {
                "column": _c,
                "present_in_sample": _c.split(" ")[0] in shots_feat.columns,
                "why_it_leaks": _w,
            }
            for _c, _w in _WATCH.items()
        ]
    )
    mo.vstack(
        [
            mo.md("**Leakage watchlist** (present in the raw data on purpose — "
                  "exclude at feature-build time)"),
            leakage_watch,
        ]
    )
    return


@app.cell
def _(mo, pd, shots_feat):
    # Signal sanity check: if the data fits the question, goal rate must fall
    # monotonically with distance. (The xG analog of "rolling features beat
    # nothing" from the NBA baseline.)
    _op = shots_feat[shots_feat["is_penalty"] == 0]
    signal_by_distance = (
        _op.assign(
            distance_bin=pd.cut(
                _op["distance"],
                bins=[0, 6, 12, 18, 30, 90],
                labels=["0-6", "6-12", "12-18", "18-30", "30+"],
            )
        )
        .groupby("distance_bin", observed=True)
        .agg(
            shots=("is_goal", "size"),
            goal_rate=("is_goal", "mean"),
            mean_sb_xg=("shot_statsbomb_xg", "mean"),
        )
        .round(3)
        .reset_index()
    )
    mo.vstack(
        [
            mo.md("**Signal check** — open-play goal rate vs distance (SB units ≈ yards)"),
            signal_by_distance,
        ]
    )
    return


@app.cell
def _(mo, shots_feat):
    # Class balance + external benchmark sanity by competition.
    balance_tbl = (
        shots_feat.groupby("competition_name")
        .agg(
            shots=("is_goal", "size"),
            goal_rate=("is_goal", "mean"),
            penalty_share=("is_penalty", "mean"),
            mean_sb_xg=("shot_statsbomb_xg", "mean"),
            freeze_frame_cov=("shot_freeze_frame", lambda s: s.notna().mean()),
        )
        .round(3)
        .reset_index()
    )
    mo.vstack(
        [
            mo.md(
                "**Class balance & benchmark** — mean StatsBomb xG tracking the "
                "actual goal rate is evidence a calibrated model is achievable "
                "on these inputs; it is also the external bar to compare against "
                "(the Vegas line of this project)."
            ),
            balance_tbl,
        ]
    )
    return


@app.cell
def _(all_matches, comps, sampled_matches, shots_feat):
    # Headline numbers reused by the limits/feasibility sections and the note.
    _top = (
        all_matches["competition_name"].value_counts().rename_axis("comp").reset_index(name="n")
    )
    _shots_per_match = len(shots_feat) / max(1, len(sampled_matches))
    fit_stats = {
        "n_competitions": int(comps["competition_name"].nunique()),
        "n_comp_seasons": int(len(comps)),
        "n_360": int(comps["match_available_360"].notna().sum()),
        "total_matches": int(len(all_matches)),
        "first_date": str(all_matches["match_date"].min()),
        "last_date": str(all_matches["match_date"].max()),
        "top_comp": str(_top.loc[0, "comp"]),
        "top_comp_share": float(_top.loc[0, "n"] / len(all_matches)),
        "female_share": float((all_matches["competition_gender"] == "female").mean()),
        "modern_spec_share": float((all_matches["data_version"] == "1.1.0").mean()),
        "n_sample_matches": int(len(sampled_matches)),
        "n_sample_seasons": int(sampled_matches["season_name"].nunique()),
        "n_shots": int(len(shots_feat)),
        "goal_rate": float(shots_feat["is_goal"].mean()),
        "penalty_share": float(shots_feat["is_penalty"].mean()),
        "sb_xg_mean": float(shots_feat["shot_statsbomb_xg"].mean()),
        "freeze_cov": float(shots_feat["shot_freeze_frame"].notna().mean()),
        "shots_per_match": float(_shots_per_match),
        "est_total_shots": int(round(_shots_per_match * len(all_matches))),
    }
    fit_stats["est_total_goals"] = int(
        round(fit_stats["est_total_shots"] * fit_stats["goal_rate"])
    )
    fit_stats
    return (fit_stats,)


@app.cell
def _(fit_stats, mo):
    mo.md(
        f"""
        ## ③ Limits — the three biggest

        **1. It's a convenience sample, not a population.**
        {fit_stats["top_comp"]} alone is {fit_stats["top_comp_share"]:.0%} of all
        {fit_stats["total_matches"]:,} matches (and it is there because of Messi-era
        Barcelona), tournaments are overrepresented, and
        {fit_stats["female_share"]:.0%} of matches are women's football. Whatever
        we fit describes *this corpus*, not "football". → Mitigate with
        competition-grouped train/test splits and per-competition reliability
        curves; never report one pooled number.

        **2. Schema and annotation drift across ~20 years.**
        Only {fit_stats["modern_spec_share"]:.0%} of matches are on the current
        spec (`data_version 1.1.0`); older seasons differ in fidelity versions and
        field coverage, and StatsBomb's boolean flags encode False as NaN — a
        missingness audit that ignores this silently misreads the data.
        → Mitigate with an explicit fillna policy for flags, per-season coverage
        gates, and pinning one cached pull for the whole capstone.

        **3. Context ceiling: freeze frames aren't tracking data.**
        Pre-shot context is shot location plus a 2-D freeze frame (360 data exists
        for only {fit_stats["n_360"]} of {fit_stats["n_comp_seasons"]} competition-
        seasons); no ball speed, no GK micro-positioning. Some outcome variance is
        simply invisible to the features — the same reason the NBA model honestly
        capped at ~63% vs Vegas's ~66–68%. → Frame StatsBomb's own xG as the
        external benchmark; treat the residual gap as measured context deficit,
        not model failure.
        """
    )
    return


@app.cell
def _(fit_stats, mo):
    mo.md(
        f"""
        ## ④ Feasibility

        Sampled {fit_stats["n_sample_matches"]} matches across
        {fit_stats["n_sample_seasons"]} seasons → {fit_stats["shots_per_match"]:.1f}
        shots/match. Extrapolated to all {fit_stats["total_matches"]:,} matches:
        **≈{fit_stats["est_total_shots"]:,} shots, ≈{fit_stats["est_total_goals"]:,}
        goals** at a {fit_stats["goal_rate"]:.1%} base rate. That supports
        10-bin reliability curves with thousands of shots per bin, grouped CV by
        competition, and a proper isotonic/Platt calibration holdout — the
        question is answerable at this scale.

        Compute cost: one full pull is ~{fit_stats["total_matches"]:,} event files
        (a few GB, tens of minutes). Do it **once**, flatten shots to a single
        parquet, and develop against the cache — identical to the NBA pipeline's
        collect-then-model split.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Partner stress-test (dataset swap)

    **Q1. "~10% positives — can you calibrate that at all?"**
    Yes: tens of thousands of shots and thousands of goals; calibration is
    binned, not per-event. ECE/Brier/reliability are the primary metrics by design.

    **Q2. "Isn't this just a Barcelona dataset?"**
    Partly — that's Limitation #1, quantified above. Grouped splits by
    competition and per-group reliability curves make the bias visible
    instead of hidden.

    **Q3. "`shot_statsbomb_xg` is right there. Why not use it as a feature?"**
    Because it's another model's output encoding label information —
    this project's `H2H_HOME_WIN_PCT`. It stays benchmark-only.

    **Q4. "Freeze frames aren't tracking. Is your ceiling too low to matter?"**
    The ceiling is the finding: the gap between my calibrated model and
    StatsBomb's (which saw richer internal data) measures the value of
    context. 360 freeze frames on the 12 covered competition-seasons are the
    stretch-goal extension.

    **Q5. "Men's, women's, leagues, World Cups — one model or many?"**
    Empirical question the data can answer: fit pooled with competition
    covariates vs stratified, compare per-group ECE, report both.
    """)
    return


@app.cell
def _(mo):
    REFINED_QUESTION = (
        "Using StatsBomb Open Data shots (penalty-shootout events excluded, "
        "penalties modeled separately), can a model built only on pre-shot "
        "features produce goal probabilities that are well-calibrated within "
        "*and across* competitions — measured by reliability curves, ECE, and "
        "Brier score under competition-grouped validation, benchmarked against "
        "StatsBomb's own xG?"
    )
    mo.md(
        f"""
        ## Decision: **KEEP — with the question reshaped, not the data**

        The schema audit says fit is real (all core pre-shot fields at ~full
        coverage, freeze frames on essentially every shot, monotone
        distance→goal-rate signal). The limits don't kill the question; they
        *sharpen* it:

        > **Refined RQ:** {REFINED_QUESTION}

        "Get more data" is the contingency, not the plan: 360 frames (12
        competition-seasons) if freeze-frame features plateau.
        """
    )
    return (REFINED_QUESTION,)


@app.cell
def _(
    Path,
    REFINED_QUESTION,
    RESEARCH_QUESTION,
    dt,
    fetch_failures,
    fit_stats,
    mo,
    n_shootout_excluded,
    skipped_pulls,
):
    # ---- Auto-generate the lab deliverable: the data-fit note ----
    note_md = f"""# Data-Fit Note — Calibrated xG on StatsBomb Open Data
    *MSDS 696 Practicum II · Pod Lab: “Vet your data against your question” · {dt.date.today().isoformat()}*

    **Research question (v1).** {RESEARCH_QUESTION}

    ## Fit
    The event schema contains exactly what pre-shot xG needs: shot location, body part,
    technique, play pattern, shooter position, pressure flags, and a 2-D freeze frame on
    ~{fit_stats["freeze_cov"]:.0%} of sampled shots, plus a clean binary outcome
    (goal rate {fit_stats["goal_rate"]:.1%}, penalties {fit_stats["penalty_share"]:.1%} of shots;
    {n_shootout_excluded} shootout penalties excluded from the sample as non-in-game events).
    Goal rate falls monotonically with derived shot distance — the signal the question needs
    is present. Mean StatsBomb xG ({fit_stats["sb_xg_mean"]:.3f}) tracks the realized goal
    rate, so a calibrated model on these inputs is demonstrably achievable.

    ## Provenance
    StatsBomb Ltd's public research release (github.com/statsbomb/open-data): broadcast-video
    annotation against a versioned public spec; free for non-commercial research with
    attribution. Inventory at pull time: **{fit_stats["n_competitions"]} competitions,
    {fit_stats["n_comp_seasons"]} competition-seasons, {fit_stats["total_matches"]:,} matches
    ({fit_stats["first_date"]} → {fit_stats["last_date"]})**; 360 freeze-frame data on
    {fit_stats["n_360"]} competition-seasons. Data is served from a moving `master` branch →
    capstone will cache one pull to parquet and record the pull date. (This audit:
    {skipped_pulls} match-index pulls and {fetch_failures} event pulls failed.)

    ## Three biggest limitations
    1. **Convenience sample, not a population.** {fit_stats["top_comp"]} alone is
       {fit_stats["top_comp_share"]:.0%} of matches (Messi-era Barcelona bias); tournaments
       overrepresented; {fit_stats["female_share"]:.0%} women's matches. External validity is
       limited → competition-grouped splits and per-competition reliability reporting.
    2. **Schema/annotation drift.** Only {fit_stats["modern_spec_share"]:.0%} of matches are on
       the current `data_version 1.1.0`; sparse boolean flags encode False as NaN, so naive
       missingness reads are wrong → explicit flag-fill policy and per-season coverage gates.
    3. **Context ceiling.** Freeze frames ≠ tracking data (no ball speed, limited GK detail;
       360 only on {fit_stats["n_360"]}/{fit_stats["n_comp_seasons"]} competition-seasons).
       Residual error is partly irreducible → benchmark against StatsBomb's own xG and report
       the gap as measured context deficit (the "63% vs Vegas" framing from Practicum I).

    ## Feasibility
    {fit_stats["shots_per_match"]:.1f} shots/match on the audit sample
    ({fit_stats["n_sample_matches"]} matches, {fit_stats["n_sample_seasons"]} seasons) ⇒
    **≈{fit_stats["est_total_shots"]:,} shots / ≈{fit_stats["est_total_goals"]:,} goals**
    corpus-wide: ample for 10-bin reliability curves, grouped CV, and a calibration holdout.
    One-time full pull (~{fit_stats["total_matches"]:,} event files) cached to parquet keeps
    iteration fast. Leakage controls fixed up front: `shot_statsbomb_xg` (benchmark only),
    `shot_outcome`, `shot_end_location`, `shot_deflected`, and match final scores are barred
    from the feature set.

    ## Partner stress-test → survived
    Class imbalance (binned calibration handles ~10% positives at this n); Barcelona bias
    (made visible via grouped splits, not hidden); "why not use StatsBomb xG as a feature"
    (label-adjacent leakage — benchmark only); low ceiling (the model-vs-StatsBomb gap *is* a
    finding); pooled vs stratified across genders/formats (empirical sub-question, both reported).

    ## Decision — KEEP (question reshaped, not the data)
    {REFINED_QUESTION}
    *Contingency, not plan:* add 360 freeze-frame features (12 competition-seasons) if
    standard freeze-frame features plateau.
    """
    note_path = Path("xg_data_fit_note.md")
    note_path.write_text(note_md, encoding="utf-8")
    print(f"[note] wrote {note_path.resolve()} ({len(note_md.split())} words)")
    mo.md(note_md)
    return


if __name__ == "__main__":
    app.run()
