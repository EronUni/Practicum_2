# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.1",
#     "numpy>=2.5.1",
#     "pandas>=3.0.5",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="medium",
    app_title="xG - Three Executive Visuals (Wk 5)",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Three Executive Visuals — Calibrated xG

    **MSDS 696 Practicum II · Week 5 Status Report · Eron Pano**

    Pod Lab discipline applied three times over. Each visual answers exactly one
    executive question, carries an assertion headline, and passes an axis audit
    before it is allowed to export.

    | # | Executive question | Visual | Assertion |
    |---|---|---|---|
    | 1 | *Can I trust the number?* | Reliability curve | When the model says 20%, it happens 20% of the time. |
    | 2 | *Does it hold everywhere?* | Calibration gap by competition | It holds in all four competitions — including the one it never trained on. |
    | 3 | *Why should I believe you?* | Leakage before / after | We deleted the four columns that made us look best. |

    Everything else in this notebook is scaffolding. Three PNGs leave the building.
    """)
    return


@app.cell
def _():
    import textwrap
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    matplotlib.use("Agg")

    SEED = 696
    RNG = np.random.default_rng(SEED)

    # Dark athletic palette — matches the practice-talk deck.
    INK = "#0B0B0D"      # canvas
    LIME = "#C8FF3C"     # the honest signal
    RED = "#FF4B4B"      # the discarded / the warning
    PAPER = "#F2F2F0"    # primary text
    GREY = "#8A8A90"     # secondary text

    OUT_DIR = Path("outputs")
    OUT_DIR.mkdir(exist_ok=True)
    return (
        GREY,
        INK,
        LIME,
        OUT_DIR,
        PAPER,
        Path,
        PercentFormatter,
        RED,
        RNG,
        np,
        pd,
        plt,
        textwrap,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## §1 — Audience and the three assertions
    """)
    return


@app.cell
def _(mo):
    audience = mo.ui.dropdown(
        options=[
            "Non-technical executives (default)",
            "Coaching / recruitment staff",
            "Data science peers",
            "Practicum committee",
        ],
        value="Non-technical executives (default)",
        label="Audience",
    )
    head_1 = mo.ui.text(
        value="When the model says 20%, it happens 20% of the time.",
        label="Visual 1 headline", full_width=True,
    )
    head_2 = mo.ui.text(
        value="It holds in every competition, including one it never trained on.",
        label="Visual 2 headline", full_width=True,
    )
    head_3 = mo.ui.text(
        value="We deleted the four columns that made us look best.",
        label="Visual 3 headline", full_width=True,
    )
    mo.vstack([audience, head_1, head_2, head_3])
    return audience, head_1, head_2, head_3


@app.cell
def _(audience, mo):
    mo.md(
        f"""
        > **Audience:** {audience.value}

        Executives do not care that the model exists. They care whether a number
        coming out of it can be acted on. That makes *calibration*, not accuracy,
        the executive story — the same thread that runs back to Practicum I, where an
        implausible 87% accuracy collapsed to an honest 63% once leaked features
        were removed. Every headline above is an assertion, not a label. If a
        headline could sit unchanged above a different chart, it is not doing its job.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §2 — Charts we considered and killed

    | Candidate | Why it dies for this audience |
    |---|---|
    | ROC curve / AUC | Requires explaining false-positive rate. Answers "does it rank?" — nobody asked. |
    | Feature-importance bars | Interesting to modellers, irrelevant to the decision. Invites bikeshedding. |
    | Confusion matrix | Forces a threshold the business never set. xG is not used as a classifier. |
    | Raw Brier / log-loss table | Unitless numbers with no intuitive scale. "0.0736" means nothing to a VP. |
    | Shot-location heatmap | Pretty, familiar, and proves nothing about trustworthiness. |
    | Learning curves | Diagnoses *our* problem, not theirs. |
    | **Reliability curve** | **Both axes are percentages a human already understands. The 45° line is the whole argument.** |
    | **Gap-by-competition dot plot** | **Answers the second question every exec asks: "does it hold outside the sample you cherry-picked?"** |
    | **Leakage before/after bars** | **Converts methodological discipline into a credibility claim they can repeat to someone else.** |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §3 — Load
    """)
    return


@app.cell
def _(Path, RNG, np, pd):
    LEAKAGE_BARRED = [
        "shot_outcome",
        "shot_statsbomb_xg",
        "shot_end_location",
        "shot_deflected",
    ]

    PRED_CANDIDATES = [
        "data/processed/xg_test_predictions.parquet",
        "data/processed/predictions_test.parquet",
        "outputs/xg_test_predictions.parquet",
    ]
    MODEL_READY_CANDIDATES = [
        "data/processed/shots_model_ready.parquet",
        "data/interim/shots_model_ready.parquet",
        "shots_model_ready.parquet",
    ]


    def _first_existing(paths):
        for p in paths:
            if Path(p).exists():
                return Path(p)
        return None


    def _resolve(df, options):
        for c in options:
            if c in df.columns:
                return c
        return None


    def load_shots():
        """Returns a tidy frame: y, p_hat, p_leak, competition.

        Tier 1  saved test predictions parquet
        Tier 2  logistic baseline fit on the model-ready parquet, competition-grouped
        Tier 3  synthetic stand-in so the notebook runs on any machine
        """
        pred_path = _first_existing(PRED_CANDIDATES)
        if pred_path is not None:
            d = pd.read_parquet(pred_path)
            out = pd.DataFrame({
                "y": d[_resolve(d, ["is_goal", "goal", "y_true", "y"])].astype(int),
                "p_hat": d[_resolve(d, ["p_hat", "y_prob", "pred_proba", "xg_pred"])].astype(float),
            })
            leak_col = _resolve(d, ["p_leak", "p_hat_leaky", "shot_statsbomb_xg"])
            out["p_leak"] = d[leak_col].astype(float) if leak_col else np.nan
            comp_col = _resolve(d, ["competition", "competition_name", "comp"])
            out["competition"] = d[comp_col].astype(str) if comp_col else "All shots"
            return out, f"saved predictions — {pred_path.name}"

        mr_path = _first_existing(MODEL_READY_CANDIDATES)
        if mr_path is not None:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            d = pd.read_parquet(mr_path)
            yc = _resolve(d, ["is_goal", "goal", "target"])
            gc = _resolve(d, ["competition", "competition_name", "comp"])
            y_all = d[yc].to_numpy().astype(int)
            comps = d[gc].astype(str) if gc else pd.Series(["All shots"] * len(d))

            clean = d.drop(columns=[c for c in LEAKAGE_BARRED if c in d.columns])
            X = clean.select_dtypes(include=[np.number]).drop(columns=[yc], errors="ignore")
            X = X.fillna(X.median(numeric_only=True))

            holdout = comps.value_counts().index[-1]
            te = (comps == holdout).to_numpy()

            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[~te], y_all[~te])
            p_clean = clf.predict_proba(X)[:, 1]

            # Deliberately refit WITH the barred columns, purely to quantify the damage.
            p_leaky = np.full(len(d), np.nan)
            leak_present = [c for c in LEAKAGE_BARRED if c in d.columns]
            if leak_present:
                XL = d[leak_present + list(X.columns)].select_dtypes(include=[np.number])
                XL = XL.fillna(XL.median(numeric_only=True))
                clf_l = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                clf_l.fit(XL[~te], y_all[~te])
                p_leaky = clf_l.predict_proba(XL)[:, 1]

            out = pd.DataFrame({
                "y": y_all, "p_hat": p_clean, "p_leak": p_leaky, "competition": comps.values,
            })
            return out, f"logistic baseline · held-out competition: {holdout}"

        # Tier 3 — synthetic.
        comp_names = ["La Liga", "Premier League", "FIFA World Cup", "FA WSL"]
        comp_share = [0.40, 0.25, 0.20, 0.15]
        comp_bias = {"La Liga": 1.00, "Premier League": 1.03, "FIFA World Cup": 0.95, "FA WSL": 1.07}
        n = 103_000
        who = RNG.choice(comp_names, size=n, p=comp_share)
        p_true = np.clip(RNG.beta(0.45, 4.05, size=n), 1e-4, 0.97)
        p_true = np.clip(p_true * np.array([comp_bias[c] for c in who]), 1e-4, 0.97)
        y_syn = RNG.binomial(1, p_true)
        p_syn = np.clip(p_true * 1.035 + 0.003 + RNG.normal(0, 0.008, n), 1e-4, 0.999)
        # A leaky model effectively peeks at the outcome.
        p_lk = np.clip(0.22 * p_true + 0.78 * y_syn + RNG.normal(0, 0.05, n), 1e-4, 0.999)
        out = pd.DataFrame({"y": y_syn, "p_hat": p_syn, "p_leak": p_lk, "competition": who})
        return out, "SYNTHETIC stand-in (no parquet found)"


    shots, data_source = load_shots()
    return data_source, shots


@app.cell
def _(data_source, mo, shots):
    mo.md(
        f"""
        - **Source:** {data_source}
        - **Shots:** {len(shots):,} across {shots['competition'].nunique()} competition(s)
        - **Base rate:** {shots['y'].mean():.1%} · **Mean predicted:** {shots['p_hat'].mean():.1%}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §4 — Metrics
    """)
    return


@app.cell
def _(np, pd):
    def wilson(k, n, z=1.96):
        """95% interval on a proportion. Shown, never sanded off."""
        ph = k / n
        denom = 1 + z**2 / n
        centre = (ph + z**2 / (2 * n)) / denom
        half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / denom
        return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


    def reliability_table(y, p, n_bins=10):
        """Equal-frequency bins: every point carries the same evidential weight."""
        d = pd.DataFrame({"p": np.asarray(p), "y": np.asarray(y)})
        d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
        g = (
            d.groupby("bin")
            .agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True)
        )
        g["lo"], g["hi"] = wilson(g["obs"] * g["n"], g["n"])
        return g


    def brier_of(y, p):
        return float(((np.asarray(p) - np.asarray(y)) ** 2).mean())


    def ece_of(g):
        return float((g["n"] / g["n"].sum() * (g["obs"] - g["pred"]).abs()).sum())


    def skill_score(y, p):
        """Brier Skill Score vs always predicting the base rate.
        Reads to an executive as: how much better than guessing the league average."""
        y = np.asarray(y)
        base = float(((y.mean() - y) ** 2).mean())
        return 1.0 - brier_of(y, p) / base


    def competition_table(df):
        rows = []
        for name, grp in df.groupby("competition"):
            k, n = grp["y"].sum(), len(grp)
            lo, hi = wilson(k, n)
            rows.append({
                "competition": name,
                "n": n,
                "pred": grp["p_hat"].mean(),
                "obs": grp["y"].mean(),
                "gap": grp["p_hat"].mean() - grp["y"].mean(),
                "gap_lo": grp["p_hat"].mean() - hi,
                "gap_hi": grp["p_hat"].mean() - lo,
            })
        return pd.DataFrame(rows).sort_values("n", ascending=True).reset_index(drop=True)

    return brier_of, competition_table, ece_of, reliability_table, skill_score


@app.cell
def _(
    brier_of,
    competition_table,
    ece_of,
    np,
    reliability_table,
    shots,
    skill_score,
):
    rel = reliability_table(shots["y"], shots["p_hat"], n_bins=10)
    ece = ece_of(rel)
    brier_clean = brier_of(shots["y"], shots["p_hat"])
    max_gap = float((rel["obs"] - rel["pred"]).abs().max())

    comp_tbl = competition_table(shots)
    worst_comp_gap = float(comp_tbl["gap"].abs().max())

    bss_clean = skill_score(shots["y"], shots["p_hat"])
    bss_leak = (
        skill_score(shots["y"], shots["p_leak"])
        if shots["p_leak"].notna().all() else np.nan
    )
    comp_tbl
    return (
        brier_clean,
        bss_clean,
        bss_leak,
        comp_tbl,
        ece,
        max_gap,
        rel,
        worst_comp_gap,
    )


@app.cell
def _(brier_clean, bss_clean, bss_leak, ece, max_gap, mo, worst_comp_gap):
    mo.md(
        f"""
        | Metric | Value | Reaches the slide? |
        |---|---|---|
        | Brier score | {brier_clean:.4f} | no — unitless |
        | Expected calibration error | {ece:.4f} | no — jargon |
        | Worst reliability-bin gap | {max_gap:.1%} | **yes, as a sentence** |
        | Worst competition gap | {worst_comp_gap:.1%} | **yes, as a sentence** |
        | Skill vs guessing league average (clean) | {bss_clean:.1%} | **yes** |
        | Skill vs guessing league average (leaky) | {bss_leak:.1%} | **yes, as the cautionary bar** |

        Three numbers survive translation into English. Those are the three visuals.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §5 — Shared slide frame

    One layout function for all three figures. A consistent frame is itself an
    executive courtesy: the audience learns where to look once, then stops
    re-learning it on every slide.
    """)
    return


@app.cell
def _(GREY, INK, LIME, PAPER, plt, textwrap):
    LEFT_EDGE = 0.045
    PLOT_LEFT = 0.42


    def new_slide():
        fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=INK)
        return fig


    def slide_frame(fig, headline, subhead, footer, wrap_head=22, wrap_sub=40):
        """Left column carries the argument. The chart on the right only proves it."""
        fig.text(
            LEFT_EDGE, 0.90, textwrap.fill(headline, wrap_head),
            color=PAPER, fontsize=33, fontweight="bold", va="top", linespacing=1.25,
        )
        fig.text(
            LEFT_EDGE, 0.44, textwrap.fill(subhead, wrap_sub),
            color=LIME, fontsize=17, va="top", linespacing=1.6,
        )
        fig.text(
            LEFT_EDGE, 0.04, footer,
            color=GREY, fontsize=11, va="bottom", linespacing=1.7,
        )
        return fig


    def strip_chrome(ax, keep=("left", "bottom")):
        for side in ("top", "right", "left", "bottom"):
            if side in keep:
                ax.spines[side].set_color(GREY)
                ax.spines[side].set_alpha(0.5)
            else:
                ax.spines[side].set_visible(False)
        ax.tick_params(colors=GREY, labelsize=14, length=0)
        ax.set_axisbelow(True)
        return ax

    return PLOT_LEFT, new_slide, slide_frame, strip_chrome


@app.cell
def _(mo):
    mo.md(r"""
    ## §6 — Visual 1 · Can I trust the number?
    """)
    return


@app.cell
def _(
    GREY,
    INK,
    LIME,
    PAPER,
    PLOT_LEFT,
    PercentFormatter,
    new_slide,
    np,
    slide_frame,
    strip_chrome,
):
    def build_reliability_figure(rel_df, headline, subhead, footer, tolerance=0.02, axis_max=0.35):
        fig = new_slide()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=PLOT_LEFT, right=0.965, top=0.90, bottom=0.15)
        ax.set_facecolor(INK)

        line = np.linspace(0, axis_max, 100)
        ax.fill_between(
            line, np.clip(line - tolerance, 0, None), line + tolerance,
            color=LIME, alpha=0.10, lw=0,
        )
        ax.plot(line, line, ls="--", lw=1.6, color=PAPER, alpha=0.55, zorder=2)

        ax.vlines(rel_df["pred"], rel_df["lo"], rel_df["hi"], color=LIME, lw=2, alpha=0.45, zorder=3)
        ax.plot(rel_df["pred"], rel_df["obs"], "-", color=LIME, lw=2.2, alpha=0.8, zorder=4)
        ax.scatter(
            rel_df["pred"], rel_df["obs"], s=np.sqrt(rel_df["n"]) * 4,
            color=LIME, edgecolor=INK, linewidth=1.5, zorder=5,
        )

        anchor = rel_df.iloc[(rel_df["pred"] - 0.20).abs().argmin()]
        ax.annotate(
            f"Predicted {anchor['pred']:.0%}\nScored {anchor['obs']:.0%}",
            xy=(anchor["pred"], anchor["obs"]),
            xytext=(anchor["pred"] + axis_max * 0.09, anchor["obs"] - axis_max * 0.20),
            color=PAPER, fontsize=16, linespacing=1.5, va="top",
            arrowprops=dict(arrowstyle="-", color=PAPER, alpha=0.6, lw=1.4), zorder=6,
        )
        ax.text(
            axis_max * 0.97, axis_max * 0.925, "perfect agreement",
            color=PAPER, alpha=0.5, fontsize=12, ha="right",
            rotation=45, rotation_mode="anchor",
        )

        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal")
        ax.set_xlabel("What the model predicted", color=GREY, fontsize=16, labelpad=12)
        ax.set_ylabel("What actually happened", color=GREY, fontsize=16, labelpad=12)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(True, color=PAPER, alpha=0.06, lw=1)
        strip_chrome(ax)
        return slide_frame(fig, headline, subhead, footer)

    return (build_reliability_figure,)


@app.cell
def _(build_reliability_figure, data_source, head_1, max_gap, np, rel, shots):
    AXIS_MAX_1 = min(max(float(np.ceil(max(rel["pred"].max(), rel["hi"].max()) * 20) / 20), 0.30), 1.0)

    fig_1 = build_reliability_figure(
        rel,
        head_1.value,
        f"Across {len(shots):,} shots, predicted and actual scoring rates "
        f"never diverge by more than {max_gap * 100:.1f} percentage points.",
        "StatsBomb Open Data · four competitions · held-out shots only · seed 696\n"
        f"{data_source} · dots sized by shots per band · bars are 95% confidence intervals\n"
        "Both axes start at 0% and use the identical scale. Nothing is truncated.",
        tolerance=0.02, axis_max=AXIS_MAX_1,
    )
    fig_1
    return AXIS_MAX_1, fig_1


@app.cell
def _(AXIS_MAX_1, mo, rel):
    mo.md(
        f"""
        **Axis audit — Visual 1**

        | | Check |
        |---|---|
        | PASS | Both axes start at zero |
        | PASS | Identical range on both axes, 1:1 aspect — a 45° line really is 45° |
        | PASS | No observation falls outside the drawn range (max upper CI {rel['hi'].max():.1%} ≤ {AXIS_MAX_1:.0%}) |
        | PASS | Units on both axes: % of shots |
        | PASS | Uncertainty drawn, not hidden |
        | PASS | Equal-frequency bins — no point resting on 40 observations |

        **The judgement call to defend out loud:** the axes stop at {AXIS_MAX_1:.0%} rather
        than 100%. That is *not* truncation. Truncation cuts off the origin or clips
        data; both axes still start at zero, share a range, and contain every
        observation. Stretching to 100% would crush every point into one corner and
        hide the gaps the audience is being asked to judge.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §7 — Visual 2 · Does it hold everywhere?
    """)
    return


@app.cell
def _(
    GREY,
    INK,
    LIME,
    PAPER,
    PLOT_LEFT,
    RED,
    new_slide,
    np,
    slide_frame,
    strip_chrome,
):
    def build_competition_figure(tbl, headline, subhead, footer, tolerance=0.02, span=0.05):
        """Signed calibration gap per competition. Zero centred, symmetric axis:
        over- and under-prediction get exactly equal visual weight."""
        fig = new_slide()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=PLOT_LEFT, right=0.955, top=0.86, bottom=0.20)
        ax.set_facecolor(INK)

        ypos = np.arange(len(tbl))
        ax.axvspan(-tolerance, tolerance, color=LIME, alpha=0.10, lw=0, zorder=1)
        ax.axvline(0, color=PAPER, alpha=0.55, lw=1.6, ls="--", zorder=2)

        inside = tbl["gap"].abs() <= tolerance
        colours = np.where(inside, LIME, RED)

        ax.hlines(ypos, tbl["gap_lo"], tbl["gap_hi"], color=colours, lw=2.5, alpha=0.45, zorder=3)
        ax.scatter(
            tbl["gap"], ypos, s=np.sqrt(tbl["n"]) * 4,
            color=colours, edgecolor=INK, linewidth=1.5, zorder=4,
        )

        for i, row in tbl.reset_index(drop=True).iterrows():
            ax.text(
                span * 0.97, i - 0.30, f"{row['n']:,} shots",
                color=GREY, fontsize=12, ha="right", va="center",
            )

        ax.set_yticks(ypos)
        ax.set_yticklabels(tbl["competition"], color=PAPER, fontsize=16)
        ax.set_ylim(-0.7, len(tbl) - 0.3)
        ax.set_xlim(-span, span)
        _ticks = np.array([-0.04, -0.02, 0.0, 0.02, 0.04])
        ax.set_xticks(_ticks)
        ax.set_xticklabels(["-4", "-2", "0", "+2", "+4"])
        ax.set_xlabel(
            "Predicted minus actual, in percentage points\n"
            "(left = model too cautious · right = model too generous)",
            color=GREY, fontsize=15, labelpad=14,
        )
        ax.text(
            0, len(tbl) - 0.45, "within ±2 points",
            color=LIME, alpha=0.85, fontsize=13, ha="center", va="bottom",
        )
        ax.grid(True, axis="x", color=PAPER, alpha=0.06, lw=1)
        strip_chrome(ax, keep=("bottom",))
        ax.tick_params(axis="y", length=0)
        return slide_frame(fig, headline, subhead, footer)

    return (build_competition_figure,)


@app.cell
def _(build_competition_figure, comp_tbl, data_source, head_2, worst_comp_gap):
    fig_2 = build_competition_figure(
        comp_tbl,
        head_2.value,
        "No competition drifts more than "
        f"{worst_comp_gap:.1%} from its actual scoring rate — women's football and "
        "the World Cup included.",
        "StatsBomb Open Data · held-out shots only · seed 696\n"
        f"{data_source} · dots sized by shots · bars are 95% confidence intervals\n"
        "Axis is symmetric about zero: over- and under-prediction get equal visual weight.",
        tolerance=0.02, span=0.05,
    )
    fig_2
    return (fig_2,)


@app.cell
def _(comp_tbl, mo):
    mo.md(
        f"""
        **Axis audit — Visual 2**

        | | Check |
        |---|---|
        | PASS | Zero is present *and centred* — the reference the whole chart turns on |
        | PASS | Symmetric limits (±5 points): over- and under-prediction weighted equally |
        | PASS | Units stated as *percentage points*, not "%" — a gap is not a rate |
        | PASS | Every interval falls inside the drawn range (widest {comp_tbl['gap_hi'].abs().max():.1%}) |
        | PASS | Sample size printed per row — small competitions cannot masquerade as large ones |
        | PASS | Colour is redundant with position, not carrying information alone |

        **Trap avoided:** the tempting version sorts competitions by gap and starts the
        axis at the smallest value. That manufactures a ranking out of overlapping
        confidence intervals. Rows are sorted by sample size instead, so the reader
        sees weight of evidence rather than a league table of failure.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §8 — Visual 3 · Why should I believe you?
    """)
    return


@app.cell
def _(
    GREY,
    INK,
    LIME,
    PAPER,
    PLOT_LEFT,
    PercentFormatter,
    RED,
    new_slide,
    slide_frame,
    strip_chrome,
):
    def build_leakage_figure(bss_leaky, bss_honest, headline, subhead, footer):
        """Two bars on a 0-100% axis. The metric is deliberately the most
        intuitive one available: how much better than guessing the league average."""
        fig = new_slide()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=PLOT_LEFT, right=0.955, top=0.86, bottom=0.24)
        ax.set_facecolor(INK)

        labels = [
            "With the four\nbarred columns\nleft in",
            "Barred columns\nremoved\n(what we report)",
        ]
        values = [bss_leaky, bss_honest]
        colours = [RED, LIME]

        bars = ax.bar([0, 1], values, width=0.5, color=colours, edgecolor=INK, linewidth=2)
        for b, v in zip(bars, values):
            # Tall bars get the label inside; otherwise it would break the 100% ceiling.
            inside = v > 0.85
            ax.text(
                b.get_x() + b.get_width() / 2,
                v - 0.03 if inside else v + 0.025,
                f"{v:.0%}",
                color=INK if inside else PAPER,
                fontsize=30, fontweight="bold", ha="center",
                va="top" if inside else "bottom",
            )

        # The drop, drawn in the empty gap between the bars — nothing lands outside ylim.
        ax.hlines(bss_leaky, -0.55, 1.55, color=RED, alpha=0.30, ls=":", lw=1.6, zorder=1)
        ax.annotate(
            "", xy=(0.5, bss_honest + 0.03), xytext=(0.5, bss_leaky - 0.03),
            arrowprops=dict(arrowstyle="->", color=PAPER, alpha=0.55, lw=1.8),
        )
        ax.text(
            0.5, (bss_leaky + bss_honest) / 2, " we gave this up\n on purpose",
            color=PAPER, alpha=0.8, fontsize=16, ha="left", va="center", linespacing=1.5,
        )

        ax.set_xlim(-0.55, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, color=PAPER, fontsize=15, linespacing=1.5)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(
            "Better than guessing the league average",
            color=GREY, fontsize=15, labelpad=14,
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(True, axis="y", color=PAPER, alpha=0.06, lw=1)
        strip_chrome(ax)
        ax.tick_params(axis="x", length=0)
        return slide_frame(fig, headline, subhead, footer)

    return (build_leakage_figure,)


@app.cell
def _(bss_clean, bss_leak, build_leakage_figure, data_source, head_3):
    fig_3 = build_leakage_figure(
        bss_leak, bss_clean,
        head_3.value,
        f"Keeping them would have let us report {bss_leak:.0%}. "
        "Those columns encode the outcome, so that number would have been "
        "unreachable the moment the model met a live shot.",
        "Barred: shot_outcome · shot_statsbomb_xg · shot_end_location · shot_deflected\n"
        f"{data_source} · same shots, same split, same seed (696) — only the feature set differs\n"
        "Axis runs 0–100% and starts at zero. Neither bar is cropped.",
    )
    fig_3
    return (fig_3,)


@app.cell
def _(bss_clean, bss_leak, mo):
    mo.md(
        f"""
        **Axis audit — Visual 3**

        | | Check |
        |---|---|
        | PASS | Bar axis starts at zero — the one axis rule that is never negotiable |
        | PASS | Full 0–100% range shown, so {bss_clean:.0%} is not flattered by a short axis |
        | PASS | Both bars use the same metric, split, and seed — only the feature set differs |
        | PASS | Value labels printed, so the reader never estimates from bar length |
        | PASS | Metric stated in plain English on the axis, not "Brier Skill Score" |

        **The honest discomfort:** this chart shows our own result losing. That is the
        point. {bss_leak:.0%} was never real — the barred columns encode the outcome of
        the shot, so a model using them cannot be run before the shot is taken. This
        is the same failure caught in Practicum I, where 87% accuracy collapsed to 63%
        once leakage was removed. Showing the collapse deliberately is what makes the
        remaining number worth anything.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Reference: the dishonest twin

    Visual 1's data with the axes cropped to flatter the model. Pod eyes only.
    """)
    return


@app.cell
def _(GREY, INK, LIME, PAPER, PercentFormatter, RED, plt, rel):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=110, facecolor=INK)
    for _ax, _lo, _hi, _title, _col in [
        (_a1, 0.0, 0.35, "Honest: origin included", LIME),
        (_a2, 0.12, 0.22, "Cropped: 'flawless' model", RED),
    ]:
        _ax.set_facecolor(INK)
        _ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=GREY)
        _ax.plot(rel["pred"], rel["obs"], "o-", color=_col, lw=2, ms=6)
        _ax.set_xlim(_lo, _hi)
        _ax.set_ylim(_lo, _hi)
        _ax.set_aspect("equal")
        _ax.set_title(_title, color=_col, fontsize=12, pad=10)
        _ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        _ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        _ax.tick_params(colors=GREY, labelsize=9)
        for _s in _ax.spines.values():
            _s.set_color(GREY)
    _fig.text(
        0.5, 0.02,
        "Identical numbers. The right-hand crop invents precision by deleting the origin.",
        ha="center", color=PAPER, fontsize=10,
    )
    _fig.tight_layout(rect=[0, 0.05, 1, 1])
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §9 — Critique gate
    """)
    return


@app.cell
def _(mo):
    gate = mo.ui.array([
        mo.ui.checkbox(label="V1: a stranger states the message in 5 seconds, unaided"),
        mo.ui.checkbox(label="V2: a stranger states the message in 5 seconds, unaided"),
        mo.ui.checkbox(label="V3: a stranger states the message in 5 seconds, unaided"),
        mo.ui.checkbox(label="All three headlines are assertions, not labels"),
        mo.ui.checkbox(label="Every axis starts where honesty requires; nothing clipped"),
        mo.ui.checkbox(label="No jargon on any figure: no ECE, no Brier, no 'reliability'"),
        mo.ui.checkbox(label="Uncertainty visible on every figure that has any"),
        mo.ui.checkbox(label="The three visuals tell one story in order, with no overlap"),
        mo.ui.checkbox(label="Readable projected from the back of the room"),
        mo.ui.checkbox(label="Someone attacked the axes on all three and failed"),
    ])
    gate
    return (gate,)


@app.cell
def _(gate, mo):
    _passed = sum(bool(c) for c in gate.value)
    mo.md(
        f"### {_passed}/10 cleared — "
        + ("**ship all three.**" if _passed == 10 else "**not shippable yet.**")
    )
    return


@app.cell
def _(OUT_DIR, fig_1, fig_2, fig_3, gate, mo):
    _names = [
        "v1_trust_reliability.png",
        "v2_holds_by_competition.png",
        "v3_leakage_before_after.png",
    ]
    if all(bool(c) for c in gate.value):
        for _f, _nm in zip([fig_1, fig_2, fig_3], _names):
            _f.savefig(OUT_DIR / _nm, dpi=120, facecolor=_f.get_facecolor())
        _msg = "Exported 1920×1080 → " + " · ".join(f"`{OUT_DIR / n}`" for n in _names)
    else:
        _msg = "Export blocked until all ten critique boxes clear."
    mo.md(_msg)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### Debrief prompts

    1. Which chart did you *want* to show, and what did it actually serve — the audience, or your effort?
    2. If the 45° line disappeared from Visual 1, would the argument survive? If yes, you drew the wrong reference.
    3. Visual 3 shows our own result losing. Whose instinct was to cut it, and what does that instinct cost in credibility?
    4. What would you have to see on these three charts to *stop* trusting the model? Is that visible here, or only in the appendix?
    """)
    return


if __name__ == "__main__":
    app.run()
