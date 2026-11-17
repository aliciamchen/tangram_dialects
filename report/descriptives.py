"""
The descriptive statistics the article quotes, written as LaTeX macros.

Reads the processed tables, the measures and the per-game group-specificity
estimates stats/primary/group_specificity.R wrote, computes the descriptive
summaries (counts, attrition, endpoint means, survey responses) and writes
them as macros to results/macros/overview.tex, exploratory.tex and
survey.tex. Nothing here fits a model; report/inferential.py writes the
inferential files (checks, primary, secondary) from the estimates tables.

Usage:
    uv run python -m report.descriptives --dataset full

Tests: report/test_descriptives.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.dataset_paths import (
    FILTERED_UTTERANCES_FILE,
    add_dataset_argument,
    dataset_dirs,
    filtered_utterances_status,
    stale_filtered_message,
)
from report.bootstrap import bootstrap_mean
from report.plot_style import CONDITION_ORDER
from preprocessing.filter_nonreferential import CLASSIFIED_FILE, attach_labels
from preprocessing.preprocessing import (
    EXIT_SURVEY_FILE,
    PARTICIPANT_EXCLUSIONS_FILE,
    TECHNICAL_EXCLUSIONS_FILE,
    read_participant_exclusions,
    read_technical_exclusions,
)
from report.stats_tex import (
    add_allow_removed_argument,
    condition_macro_name,
    fmt_count,
    fmt_stat,
    stats_file,
    write_stats_files,
)

SOURCE = "report/descriptives.py"

# ── Attrition ────────────────────────────────────────────────
# Who was removed and why, and which games ran to completion, are columns the
# pipeline writes (`removed`, `removalReason` and `removedNoFault` on
# players.csv, `ranToCompletion` and `completeNine` on games.csv; the
# definitions are in preprocessing/preprocessing.py). The helpers here read
# them for the real games and add the one group-level fact.
MIN_GROUP_SIZE = 2  # the server removes a group's last member once it falls below this


def _true(series: pd.Series) -> pd.Series:
    return series.map(lambda v: v in (True, "True", "true", "TRUE")).astype(bool)


def in_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the four conditions (test games have none)."""
    return df[df["condition"].isin(CONDITION_ORDER)]


def utterances_file(tables_dir: Path) -> str:
    """The utterance file the analyses read: the filtered one when current, as config.R chooses."""
    status, detail = filtered_utterances_status(tables_dir)
    if status == "current":
        return FILTERED_UTTERANCES_FILE
    if status == "absent":
        return "speaker_utterances.csv"
    raise SystemExit(f"Error: {stale_filtered_message(status, detail)}")


def load_tables(dirs) -> dict:
    """Every table the descriptives read, as the R model scripts read them."""
    inputs, t, measures, estimates = dirs.data, dirs.tables, dirs.measures, dirs.estimates
    gs_path = estimates / "primary" / "group_specificity.csv"
    gs_jaccard_path = estimates / "primary" / "group_specificity_jaccard.csv"
    tables = {
        "games": pd.read_csv(t / "games.csv"),
        "players": pd.read_csv(t / "players.csv", low_memory=False),
        "trials": pd.read_csv(t / "trials.csv", low_memory=False),
        "messages": pd.read_csv(t / "messages.csv", low_memory=False),
        "speaker_utts": pd.read_csv(t / utterances_file(t)),
        "social_guesses": pd.read_csv(t / "social_guesses.csv", low_memory=False),
        "exit_survey": pd.read_csv(t / EXIT_SURVEY_FILE) if (t / EXIT_SURVEY_FILE).exists() else pd.DataFrame(),
        "dropouts": pd.read_csv(t / "dropouts.csv") if (t / "dropouts.csv").exists() else pd.DataFrame(),
        "participant_exclusions": read_participant_exclusions(inputs / PARTICIPANT_EXCLUSIONS_FILE),
        "technical_exclusions": read_technical_exclusions(inputs / TECHNICAL_EXCLUSIONS_FILE),
        "adjacent_sim": pd.read_csv(measures / "adjacent_similarities.csv")
        if (measures / "adjacent_similarities.csv").exists() else pd.DataFrame(),
        "phase_change": pd.read_csv(measures / "phase_change_similarities.csv")
        if (measures / "phase_change_similarities.csv").exists() else pd.DataFrame(),
        "gs_by_game": pd.read_csv(gs_path) if gs_path.exists() else pd.DataFrame(),
        "gs_by_game_jaccard": pd.read_csv(gs_jaccard_path) if gs_jaccard_path.exists() else pd.DataFrame(),
        "ingroup_advantage": pd.read_csv(dirs.model_inputs / "ingroup_advantage.csv")
        if (dirs.model_inputs / "ingroup_advantage.csv").exists() else pd.DataFrame(),
        "classified": pd.read_csv(dirs.classifier / CLASSIFIED_FILE, low_memory=False)
        if (dirs.classifier / CLASSIFIED_FILE).exists() else pd.DataFrame(),
    }
    return tables


# ── Attrition ────────────────────────────────────────────────


def player_attrition(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per player of a real game: condition, removed, reason, no_fault.

    The removal facts are the pipeline's; `reason` is the player's
    `removalReason`, and "completed" for a player who finished.
    """
    real = games[games["condition"].notna()][["gameId", "condition"]]
    out = players.merge(real, on="gameId", how="inner")
    out["removed"] = _true(out["removed"])
    out["reason"] = out["removalReason"].where(out["removed"], "completed")
    out["no_fault"] = _true(out["removedNoFault"])
    return out[["gameId", "playerId", "condition", "originalGroup", "removed", "reason", "no_fault"]]


def group_attrition(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per original group of a real game, with whether it was lost (fewer than two active members at the end)."""
    pa = player_attrition(players, games)
    out = (
        pa.groupby(["gameId", "condition", "originalGroup"], dropna=False)
        .agg(players=("removed", "size"), active_at_end=("removed", lambda r: int((~r).sum())), removed=("removed", "sum"))
        .reset_index()
    )
    out["lost"] = out["active_at_end"] < MIN_GROUP_SIZE
    return out


def game_status(games: pd.DataFrame) -> pd.DataFrame:
    """One row per real game: `ran_to_completion` and `complete_nine`, the pipeline's `ranToCompletion` and `completeNine`."""
    out = games[games["condition"].notna()][["gameId", "condition"]].copy()
    out["ran_to_completion"] = _true(games.loc[out.index, "ranToCompletion"])
    out["complete_nine"] = _true(games.loc[out.index, "completeNine"])
    return out


def quiz_failures(dropouts: pd.DataFrame) -> int:
    if dropouts.empty or "exitReason" not in dropouts.columns:
        return 0
    return int((dropouts["exitReason"] == "quiz failed").sum())


# ── Descriptions ─────────────────────────────────────────────

DESCRIPTION_KEYS = ["gameId", "playerId", "phaseNum", "blockNum", "target"]


def described_speaker_trials(trials: pd.DataFrame, speaker_utts: pd.DataFrame) -> pd.DataFrame:
    """The speaker trials of the four conditions, each with whether it has an analyzable description.

    A description is analyzable when the speaker's trial has a row in the
    utterance table: a speaker who said nothing, an excluded participant or a
    speaker whose messages were all classified non-referential has none.
    """
    speakers = in_conditions(trials[trials["role"] == "speaker"]).copy()
    described = speaker_utts[DESCRIPTION_KEYS].drop_duplicates().assign(has_description=True)
    for key in ("phaseNum", "blockNum"):
        speakers[key] = pd.to_numeric(speakers[key], errors="coerce")
        described[key] = pd.to_numeric(described[key], errors="coerce")
    merged = speakers.merge(described, on=DESCRIPTION_KEYS, how="left")
    merged["has_description"] = merged["has_description"].fillna(False).astype(bool)
    return merged


def description_coverage(trials: pd.DataFrame, speaker_utts: pd.DataFrame) -> tuple[int, int]:
    """Speaker trials of the four conditions without an analyzable description, and all speaker trials."""
    merged = described_speaker_trials(trials, speaker_utts)
    return int((~merged["has_description"]).sum()), len(merged)


def description_coverage_by_condition(trials: pd.DataFrame, speaker_utts: pd.DataFrame) -> pd.DataFrame:
    """Per condition: speaker trials, those without an analyzable description, and their proportion."""
    merged = described_speaker_trials(trials, speaker_utts)
    out = (
        merged.groupby("condition")["has_description"]
        .agg(speaker_trials="size", without=lambda d: int((~d).sum()))
        .reset_index()
    )
    out["prop_without"] = out["without"] / out["speaker_trials"]
    out["condition"] = pd.Categorical(out["condition"], CONDITION_ORDER, ordered=True)
    return out.sort_values("condition").reset_index(drop=True)


# ── Response opportunities ───────────────────────────────────

TASK_MACRO_NAMES = {"referential": "Ref", "social": "Social"}
PHASE_MACRO_NAMES = {1: "PhaseOne", 2: "PhaseTwo"}


def response_opportunities(trials: pd.DataFrame, social_guesses: pd.DataFrame) -> pd.DataFrame:
    """Per task, phase and condition: the response opportunities and what became of them.

    The opportunities are the rows the accuracy models use, as preprocessing.py
    coded them: listener rows of trials.csv with `responseOpportunity`, and
    the eligible listener rows of social_guesses.csv. `scored` leaves out the
    observations a participant exclusion or a documented technical failure
    made missing. The on-time rate, the accuracy, the accuracy among on-time
    answers and the late arrivals are over the scored opportunities. A late
    answer is coded 0, so it counts against accuracy and is not an on-time
    answer.
    """
    frames = []
    ref = trials[(trials["role"] == "listener") & _true(trials["responseOpportunity"])]
    frames.append(ref.assign(task="referential", late=_true(ref["lateClick"])))
    if not social_guesses.empty:
        soc = social_guesses[_true(social_guesses["responseOpportunity"])]
        frames.append(soc.assign(task="social", late=_true(soc["lateSocialGuess"])))
    rows = []
    for df in frames:
        df = in_conditions(df).copy()
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
        df["phaseNum"] = pd.to_numeric(df["phaseNum"], errors="coerce").astype(int)
        df["scored"] = df["correct"].notna()
        df["answered"] = df["scored"] & _true(df["onTime"])
        for (task, phase, condition), part in df.groupby(["task", "phaseNum", "condition"]):
            scored = part[part["scored"]]
            answered = part[part["answered"]]
            rows.append({
                "task": task,
                "phaseNum": int(phase),
                "condition": condition,
                "opportunities": len(part),
                "scored": len(scored),
                "on_time": len(answered),
                "on_time_rate": len(answered) / len(scored) if len(scored) else np.nan,
                "accuracy": scored["correct"].mean() if len(scored) else np.nan,
                "accuracy_on_time": answered["correct"].mean() if len(answered) else np.nan,
                "late": int(scored["late"].sum()),
                "late_rate": scored["late"].mean() if len(scored) else np.nan,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["condition"] = pd.Categorical(out["condition"], CONDITION_ORDER, ordered=True)
    return out.sort_values(["task", "phaseNum", "condition"]).reset_index(drop=True)


def response_opportunity_macros(table: pd.DataFrame) -> dict:
    """One macro family per task, phase and condition, e.g. overviewRefPhaseOneReferSeparatedAccuracy."""
    macros = {}
    for _, r in table.iterrows():
        prefix = (
            f"overview{TASK_MACRO_NAMES[r['task']]}{PHASE_MACRO_NAMES[int(r['phaseNum'])]}"
            f"{condition_macro_name(str(r['condition']))}"
        )
        macros.update({
            f"{prefix}Opportunities": fmt_count(r["opportunities"]),
            f"{prefix}Scored": fmt_count(r["scored"]),
            f"{prefix}OnTimeRate": fmt_stat(r["on_time_rate"]),
            f"{prefix}Accuracy": fmt_stat(r["accuracy"]),
            f"{prefix}AccuracyOnTime": fmt_stat(r["accuracy_on_time"]),
            f"{prefix}Late": fmt_count(r["late"]),
            f"{prefix}LateRate": fmt_stat(r["late_rate"]),
        })
    return macros


# ── Message classifier and screening flags ───────────────────


def classifier_macros(messages: pd.DataFrame, classified: pd.DataFrame) -> dict:
    """How many speaker messages there are and how many the classifier labeled
    non-referential, which are the ones left out of the descriptions.

    The labels are joined onto the current messages.csv by `attach_labels`,
    the join the filter itself uses, so the counts are of the messages the
    analyses read. Nothing for a dataset without the classifier's labels.
    """
    if classified.empty or messages.empty:
        return {}
    labeled = in_conditions(attach_labels(messages, classified))
    speaker = labeled[labeled["senderRole"] == "speaker"]
    if speaker.empty:
        return {}
    removed = int((~speaker["is_referential"].astype(bool)).sum())
    return {
        "overviewNSpeakerMessages": fmt_count(len(speaker)),
        "overviewNNonReferentialMessages": fmt_count(removed),
        "overviewPctNonReferentialMessages": fmt_stat(100 * removed / len(speaker), 1),
    }


def length_flag_count(players: pd.DataFrame, games: pd.DataFrame) -> int | None:
    """Players of real games flagged for descriptions that lengthened over
    Phase 1 (`lengthIncreaseFlag`, coded by preprocessing.flag_length_increase);
    None for a dataset without the flag."""
    if "lengthIncreaseFlag" not in players.columns:
        return None
    real = players[players["gameId"].isin(games.loc[games["condition"].notna(), "gameId"])]
    return int(_true(real["lengthIncreaseFlag"]).sum())


# ── Reshuffled groups ────────────────────────────────────────

RESHUFFLE_COUNTS = ["reshuffleGroups", "reshuffleTrios", "reshuffleTriosOk", "reshufflePairs"]


def reshuffle_macros(trials: pd.DataFrame) -> dict:
    """The groups the reshuffle formed in Phase 2 of the mixed conditions.

    The server records what each trial's assignment achieved, and trials.csv
    carries it on every row of the round: the groups formed, the three-person
    ones, those of them whose speaker had exactly one listener from their own
    original group, and the two-person ones. The macros are the sums over
    rounds. Only reshuffled rounds carry the record, so nothing else is
    counted, and a dataset without the record gets no macro.
    """
    if not set(RESHUFFLE_COUNTS) <= set(trials.columns):
        return {}
    rounds = in_conditions(trials).dropna(subset=RESHUFFLE_COUNTS).drop_duplicates(["gameId", "roundId"])
    if rounds.empty:
        return {}
    total = rounds[RESHUFFLE_COUNTS].sum()
    return {
        "overviewReshuffleNTrials": fmt_count(len(rounds)),
        "overviewReshuffleNGroups": fmt_count(total["reshuffleGroups"]),
        "overviewReshuffleNTrios": fmt_count(total["reshuffleTrios"]),
        "overviewReshuffleNTriosOk": fmt_count(total["reshuffleTriosOk"]),
        "overviewReshuffleNPairs": fmt_count(total["reshufflePairs"]),
    }


# ── Games, attrition and exclusions by condition ─────────────


def condition_counts(t: dict) -> pd.DataFrame:
    """Per condition: games started, run to completion and complete with nine players; players, players
    removed (and removed through no fault of their own), original groups lost, and excluded participants."""
    games, players = t["games"], t["players"]
    status = game_status(games)
    pa = player_attrition(players, games)
    ga = group_attrition(players, games)
    excluded = pa[pa["playerId"].isin(t["participant_exclusions"]["playerId"])] if not t["participant_exclusions"].empty else pa.iloc[0:0]
    out = pd.DataFrame({"condition": CONDITION_ORDER})
    out["games"] = out["condition"].map(status.groupby("condition").size())
    out["games_ran_to_completion"] = out["condition"].map(status.groupby("condition")["ran_to_completion"].sum())
    out["games_complete"] = out["condition"].map(status.groupby("condition")["complete_nine"].sum())
    out["players"] = out["condition"].map(pa.groupby("condition").size())
    out["removed"] = out["condition"].map(pa.groupby("condition")["removed"].sum())
    out["removed_no_fault"] = out["condition"].map(pa.groupby("condition")["no_fault"].sum())
    out["groups_lost"] = out["condition"].map(ga.groupby("condition")["lost"].sum())
    out["excluded_participants"] = out["condition"].map(excluded.groupby("condition").size())
    count_cols = [c for c in out.columns if c != "condition"]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    return out[out["games"] > 0].reset_index(drop=True)


def condition_count_macros(counts: pd.DataFrame) -> dict:
    """The per-condition counts as macros, e.g. overviewNGamesCompleteSocialMixed."""
    names = {
        "games": "NGames",
        "games_ran_to_completion": "NGamesRanToCompletion",
        "games_complete": "NGamesComplete",
        "players": "NPlayers",
        "removed": "NRemoved",
        "removed_no_fault": "NRemovedNoFault",
        "groups_lost": "NGroupsLost",
        "excluded_participants": "NExcludedParticipants",
    }
    macros = {}
    for _, r in counts.iterrows():
        cond = condition_macro_name(str(r["condition"]))
        for column, name in names.items():
            macros[f"overview{name}{cond}"] = int(r[column])
    return macros


# ── Macros ───────────────────────────────────────────────────


def overview_macros(t: dict) -> dict:
    games, players, trials = t["games"], t["players"], t["trials"]
    pa = player_attrition(players, games)
    ga = group_attrition(players, games)
    status = game_status(games)
    excluded_rows = in_conditions(trials)
    excluded_rows = excluded_rows[_true(excluded_rows["excluded"])] if "excluded" in trials.columns else excluded_rows.iloc[0:0]
    without, speaker_trials = description_coverage(trials, t["speaker_utts"])
    listeners = trials[trials["role"] == "listener"]
    by_condition = {}
    for _, r in description_coverage_by_condition(trials, t["speaker_utts"]).iterrows():
        cond = condition_macro_name(str(r["condition"]))
        by_condition[f"overviewNNoDescription{cond}"] = fmt_count(r["without"])
        by_condition[f"overviewPropNoDescription{cond}"] = fmt_stat(r["prop_without"], 3)
    by_condition.update(condition_count_macros(condition_counts(t)))
    by_condition.update(response_opportunity_macros(response_opportunities(trials, t["social_guesses"])))
    by_condition.update(classifier_macros(t["messages"], t.get("classified", pd.DataFrame())))
    flagged = length_flag_count(players, games)
    if flagged is not None:
        by_condition["overviewNLengthFlagged"] = flagged
    return by_condition | {
        "overviewNExcludedParticipants": len(t["participant_exclusions"]),
        "overviewNExcludedTrialRows": len(excluded_rows),
        "overviewNTechnicalExclusions": len(t["technical_exclusions"]),
        "overviewPropNoDescription": fmt_stat(without / speaker_trials, 3),
        "overviewNGames": int(games["condition"].notna().sum()),
        "overviewNPlayers": len(pa),
        "overviewNTrials": fmt_count(len(trials)),
        "overviewNMessages": fmt_count(len(t["messages"])),
        "overviewNSpeakerUtts": fmt_count(len(t["speaker_utts"])),
        "overviewNMessageWords": fmt_count(count_words(t["messages"]["text"])),
        "overviewNSpeakerUttWords": fmt_count(pd.to_numeric(t["speaker_utts"]["uttLength"], errors="coerce").sum()),
        "overviewNRemoved": int(pa["removed"].sum()),
        "overviewNRemovedNoFault": int(pa["no_fault"].sum()),
        "overviewNGroupsLost": int(ga["lost"].sum()),
        "overviewNGamesComplete": int(status["complete_nine"].sum()),
        "overviewNGamesRanToCompletion": int(status["ran_to_completion"].sum()),
        "overviewNDropouts": len(t["dropouts"]),
        **session_minutes_macros(t["players"], t["games"]),
        **reshuffle_macros(trials),
        "overviewNQuizFailed": quiz_failures(t["dropouts"]),
        "overviewNNoOpportunity": int((~_true(listeners["responseOpportunity"])).sum()),
    }


def count_words(texts: pd.Series) -> int:
    """The words in a column of texts, counted as the pipeline counts a
    description's `uttLength`: the runs of characters separated by whitespace."""
    return int(texts.fillna("").astype(str).str.split().str.len().sum())


def session_minutes_macros(players: pd.DataFrame, games: pd.DataFrame) -> dict:
    """The median time in the game, in whole minutes, with the 10th and 90th
    percentiles, over the players of real games who have a recorded duration
    (players who finished and players removed early alike)."""
    if "minutesSpent" not in players.columns:
        return {}
    real = games[games["condition"].notna()][["gameId"]]
    minutes = pd.to_numeric(players.merge(real, on="gameId", how="inner")["minutesSpent"], errors="coerce").dropna()
    if minutes.empty:
        return {}
    return {
        "overviewMinutesN": fmt_count(len(minutes)),
        "overviewMinutesMedian": fmt_count(round(minutes.median())),
        "overviewMinutesLowerDecile": fmt_count(round(minutes.quantile(0.1))),
        "overviewMinutesUpperDecile": fmt_count(round(minutes.quantile(0.9))),
    }


def endpoint_macros(prefix: str, df: pd.DataFrame, value: str, digits: int = 3) -> dict:
    """The mean of a measure in its first and last observed game block, with the two block numbers and the observation count.

    A measure that only starts once a pair of descriptions exists (adjacent
    similarity) is thereby reported from its own first block, not from block 1.
    """
    d = in_conditions(df).copy()
    d[value] = pd.to_numeric(d[value], errors="coerce")
    d["gameBlock"] = pd.to_numeric(d["gameBlock"], errors="coerce")
    d = d[np.isfinite(d[value]) & d["gameBlock"].notna()]
    if d.empty:
        return {}
    first, last = int(d["gameBlock"].min()), int(d["gameBlock"].max())
    return {
        f"{prefix}N": fmt_count(len(d)),
        f"{prefix}FirstBlock": first,
        f"{prefix}LastBlock": last,
        f"{prefix}First": fmt_stat(d.loc[d["gameBlock"] == first, value].mean(), digits),
        f"{prefix}Last": fmt_stat(d.loc[d["gameBlock"] == last, value].mean(), digits),
    }


def phase1_specificity_by_condition(gs_by_game: pd.DataFrame) -> pd.DataFrame:
    """Games, mean and SD of the estimable final Phase 1 group-specificity estimates, by condition."""
    if gs_by_game.empty:
        return pd.DataFrame(columns=["condition", "games", "mean", "sd"])
    gs = gs_by_game[(gs_by_game["window"] == "phase1_final") & _true(gs_by_game["estimable"])]
    gs = in_conditions(gs)
    out = (
        gs.groupby("condition")["coefficient"]
        .agg(games="size", mean="mean", sd=lambda s: s.std(ddof=1))
        .reset_index()
    )
    out["condition"] = pd.Categorical(out["condition"], CONDITION_ORDER, ordered=True)
    return out.sort_values("condition").reset_index(drop=True)


def phase1_final_words_by_condition(speaker_utts: pd.DataFrame) -> pd.DataFrame:
    """Words per description in each game's final three Phase 1 blocks, averaged within game first, by condition."""
    return final_words_by_condition(speaker_utts, phase=1)


def phase_specificity_change_by_condition(gs_by_game: pd.DataFrame) -> pd.DataFrame:
    """Group specificity at the end of each phase, by condition: the estimable
    games and their mean at each endpoint, and over the games estimable at both
    endpoints the mean Phase 2 minus Phase 1 change with a game-level bootstrap
    interval and the ratio of the paired Phase 2 mean to the paired Phase 1 mean.

    The change and ratio describe what the primary H1 and H2 contrasts test:
    descriptive summaries of how much specificity moved in each condition,
    beside the model-based contrasts, which compare conditions only.
    """
    cols = ["condition", "games_phase1", "mean_phase1", "games_phase2", "mean_phase2", "sd_phase2",
            "games_paired", "change", "change_lower", "change_upper", "ratio"]
    if gs_by_game.empty:
        return pd.DataFrame(columns=cols)
    gs = in_conditions(gs_by_game[_true(gs_by_game["estimable"])])
    wide = gs.pivot_table(index=["gameId", "condition"], columns="window", values="coefficient").reset_index()
    for w in ("phase1_final", "phase2_final"):
        if w not in wide:
            wide[w] = np.nan
    rows = []
    for cond, d in wide.groupby("condition", observed=True):
        paired = d.dropna(subset=["phase1_final", "phase2_final"]).copy()
        row = {
            "condition": cond,
            "games_phase1": int(d["phase1_final"].notna().sum()),
            "mean_phase1": d["phase1_final"].mean(),
            "games_phase2": int(d["phase2_final"].notna().sum()),
            "mean_phase2": d["phase2_final"].mean(),
            "sd_phase2": d["phase2_final"].std(ddof=1),
            "games_paired": len(paired),
            "change": np.nan, "change_lower": np.nan, "change_upper": np.nan, "ratio": np.nan,
        }
        if len(paired) >= 2:
            paired["change"] = paired["phase2_final"] - paired["phase1_final"]
            est = bootstrap_mean(paired, "change")
            row.update(change=est.mean, change_lower=est.lower, change_upper=est.upper,
                       ratio=paired["phase2_final"].mean() / paired["phase1_final"].mean())
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols)
    out["condition"] = pd.Categorical(out["condition"], CONDITION_ORDER, ordered=True)
    return out.sort_values("condition").reset_index(drop=True)


def listener_accuracy_by_speaker_group(ingroup_advantage: pd.DataFrame) -> pd.DataFrame:
    """Phase 2 referential accuracy of listeners hearing a speaker from their own
    original group and from another group, by mixed condition: the mean over
    listeners with a game-level bootstrap interval, from the per-listener table
    the in-group advantage model is fit to. The separated condition has no
    out-group speakers and is not in the table.
    """
    cols = ["condition", "listener_type", "listeners", "accuracy", "lower", "upper"]
    if ingroup_advantage.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for cond, d in in_conditions(ingroup_advantage).groupby("condition", observed=True):
        for col, label in (("acc_in", "in_group"), ("acc_out", "out_group")):
            valid = d.dropna(subset=[col])
            if len(valid) < 2:
                continue
            est = bootstrap_mean(valid, col)
            rows.append({"condition": cond, "listener_type": label, "listeners": est.n_obs,
                         "accuracy": est.mean, "lower": est.lower, "upper": est.upper})
    out = pd.DataFrame(rows, columns=cols)
    out["condition"] = pd.Categorical(out["condition"], CONDITION_ORDER, ordered=True)
    return out.sort_values(["condition", "listener_type"]).reset_index(drop=True)


def listener_accuracy_at_phase2_ends(trials: pd.DataFrame) -> pd.DataFrame:
    """Referential accuracy in the first and in the last block of Phase 2 with a
    speaker from the listener's own original group and from another group, by
    mixed condition: the mean over listener response opportunities with a
    game-level bootstrap interval. The rows and the summary are those of the
    accuracy panel of report/figures/block_trajectories.py, read at the two
    ends of the phase. The separated condition has no out-group speakers and
    is not in the table.
    """
    cols = ["condition", "listener_type", "block", "accuracy", "lower", "upper", "games", "opportunities"]
    d = in_conditions(trials)
    d = d[(d["role"] == "listener") & _true(d["responseOpportunity"])
          & (pd.to_numeric(d["phaseNum"], errors="coerce") == 2)].copy()
    d["correct"] = pd.to_numeric(d["correct"], errors="coerce")
    d = d[d["correct"].notna()]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["block"] = pd.to_numeric(d["block"]).astype(int)
    d["in_group"] = _true(d["inGroupSpeaker"])
    mixed = d.loc[~d["in_group"], "condition"].unique()
    ends = {"first": d["block"].min(), "last": d["block"].max()}
    rows = []
    for cond, part in d[d["condition"].isin(mixed)].groupby("condition", observed=True):
        for end, block in ends.items():
            for in_group, label in ((True, "in_group"), (False, "out_group")):
                cell = part[(part["block"] == block) & (part["in_group"] == in_group)]
                if cell.empty:
                    continue
                est = bootstrap_mean(cell, "correct")
                rows.append({"condition": cond, "listener_type": label, "block": end, "accuracy": est.mean,
                             "lower": est.lower, "upper": est.upper, "games": est.n_games, "opportunities": est.n_obs})
    out = pd.DataFrame(rows, columns=cols)
    out["condition"] = pd.Categorical(out["condition"], CONDITION_ORDER, ordered=True)
    return out.sort_values(["condition", "block", "listener_type"]).reset_index(drop=True)


def specificity_measure_agreement(gs_by_game: pd.DataFrame, gs_by_game_jaccard: pd.DataFrame) -> dict:
    """The Pearson correlation between a game's group specificity estimated from
    the sentence embeddings and from content-word overlap (Jaccard), over the
    game endpoints estimable on both measures, with their count: the points of
    report/figures/measure_agreement.py."""
    if gs_by_game.empty or gs_by_game_jaccard.empty:
        return {}
    keys = ["gameId", "window"]
    sbert = in_conditions(gs_by_game[_true(gs_by_game["estimable"])])[keys + ["coefficient"]]
    jaccard = in_conditions(gs_by_game_jaccard[_true(gs_by_game_jaccard["estimable"])])[keys + ["coefficient"]]
    both = sbert.merge(jaccard, on=keys, suffixes=("_sbert", "_jaccard")).dropna()
    if len(both) < 3:
        return {}
    return {
        "exploratoryGsMeasuresR": fmt_stat(np.corrcoef(both["coefficient_sbert"], both["coefficient_jaccard"])[0, 1], 2),
        "exploratoryGsMeasuresN": fmt_count(len(both)),
    }


def final_words_by_condition(speaker_utts: pd.DataFrame, phase: int) -> pd.DataFrame:
    """Words per description in the final three blocks of `phase`, averaged within game first, by condition.

    The final three blocks are the pipeline's `finalWindow` flag on the
    utterances, the window of the endpoint similarity measures.
    """
    if "finalWindow" not in speaker_utts.columns:
        raise SystemExit("Error: the utterances have no finalWindow column; rerun the preprocessing (make process) so it is written")
    utts = in_conditions(speaker_utts[pd.to_numeric(speaker_utts["phaseNum"]) == phase])
    last_three = utts[_true(utts["finalWindow"])]
    per_game = last_three.groupby(["condition", "gameId"])["uttLength"].mean().reset_index(name="words")
    out = per_game.groupby("condition")["words"].agg(games="size", mean_words="mean", sd="std").reset_index()
    out["condition"] = pd.Categorical(out["condition"], CONDITION_ORDER, ordered=True)
    return out.sort_values("condition").reset_index(drop=True)


def semantic_drift_macros(phase_change: pd.DataFrame) -> dict:
    """The baseline drift: how similar a refer_separated speaker's final Phase 2
    description of a tangram is to their final Phase 1 description, where no
    other group was ever heard, as a mean over speaker-tangram observations
    with a game-level bootstrap interval (ceiling 1). The secondary semantic-
    change model estimates the condition differences; this is the descriptive anchor."""
    if phase_change.empty or "simPhase1Phase2" not in phase_change:
        return {}
    drift = phase_change[phase_change["condition"] == "refer_separated"]
    if drift["simPhase1Phase2"].notna().sum() < 2:
        return {}
    est = bootstrap_mean(drift, "simPhase1Phase2")
    return {
        "exploratoryDriftMean": fmt_stat(est.mean),
        "exploratoryDriftLo": fmt_stat(est.lower),
        "exploratoryDriftHi": fmt_stat(est.upper),
        "exploratoryDriftGames": fmt_count(est.n_games),
    }


def exploratory_macros(t: dict) -> dict:
    trials, utts = t["trials"], t["speaker_utts"]
    listeners = trials[(trials["role"] == "listener") & pd.to_numeric(trials["correct"], errors="coerce").notna()]
    macros = {}
    macros.update(endpoint_macros("exploratoryLength", utts, "uttLength", digits=2))
    macros.update(endpoint_macros("exploratoryAccuracy", listeners, "correct"))
    if not t["adjacent_sim"].empty and "gameBlock" in t["adjacent_sim"].columns:
        macros.update(endpoint_macros("exploratorySimAdjacent", t["adjacent_sim"], "simAdjacent"))
    for _, row in phase1_specificity_by_condition(t["gs_by_game"]).iterrows():
        cond = condition_macro_name(str(row["condition"]))
        macros[f"exploratoryPhaseOneGs{cond}N"] = int(row["games"])
        macros[f"exploratoryPhaseOneGs{cond}Mean"] = fmt_stat(row["mean"])
        macros[f"exploratoryPhaseOneGs{cond}SD"] = fmt_stat(row["sd"])
    for _, row in phase1_final_words_by_condition(utts).iterrows():
        macros[f"exploratoryPhaseOneWords{condition_macro_name(str(row['condition']))}"] = fmt_stat(row["mean_words"], 2)
    for _, row in final_words_by_condition(utts, phase=2).iterrows():
        macros[f"exploratoryPhaseTwoWords{condition_macro_name(str(row['condition']))}"] = fmt_stat(row["mean_words"], 2)
    for _, row in phase_specificity_change_by_condition(t["gs_by_game"]).iterrows():
        cond = condition_macro_name(str(row["condition"]))
        macros[f"exploratoryPhaseTwoGs{cond}N"] = int(row["games_phase2"])
        macros[f"exploratoryPhaseTwoGs{cond}Mean"] = fmt_stat(row["mean_phase2"])
        macros[f"exploratoryPhaseTwoGs{cond}SD"] = fmt_stat(row["sd_phase2"])
        macros[f"exploratoryGsChange{cond}N"] = int(row["games_paired"])
        macros[f"exploratoryGsChange{cond}Mean"] = fmt_stat(row["change"])
        macros[f"exploratoryGsChange{cond}Lo"] = fmt_stat(row["change_lower"])
        macros[f"exploratoryGsChange{cond}Hi"] = fmt_stat(row["change_upper"])
        macros[f"exploratoryGsRatio{cond}"] = fmt_stat(row["ratio"], 2)
    for _, row in listener_accuracy_by_speaker_group(t.get("ingroup_advantage", pd.DataFrame())).iterrows():
        cond = condition_macro_name(str(row["condition"]))
        kind = "InGroup" if row["listener_type"] == "in_group" else "OutGroup"
        macros[f"exploratoryAcc{kind}{cond}"] = fmt_stat(row["accuracy"])
        macros[f"exploratoryAcc{kind}{cond}Lo"] = fmt_stat(row["lower"])
        macros[f"exploratoryAcc{kind}{cond}Hi"] = fmt_stat(row["upper"])
        macros[f"exploratoryAcc{kind}{cond}Listeners"] = fmt_count(row["listeners"])
    for _, row in listener_accuracy_at_phase2_ends(trials).iterrows():
        cond = condition_macro_name(str(row["condition"]))
        kind = "InGroup" if row["listener_type"] == "in_group" else "OutGroup"
        end = "FirstBlock" if row["block"] == "first" else "LastBlock"
        macros[f"exploratoryAcc{kind}{end}{cond}"] = fmt_stat(row["accuracy"])
        macros[f"exploratoryAcc{kind}{end}{cond}Lo"] = fmt_stat(row["lower"])
        macros[f"exploratoryAcc{kind}{end}{cond}Hi"] = fmt_stat(row["upper"])
    macros.update(specificity_measure_agreement(t["gs_by_game"], t.get("gs_by_game_jaccard", pd.DataFrame())))
    macros.update(semantic_drift_macros(t["phase_change"]))
    return macros


# ── Exit survey ──────────────────────────────────────────────

# The respondents and their typed answers are the pipeline's exit_survey.csv
# (build_exit_survey() in preprocessing/preprocessing.py): one row per player
# of a real game who answered any item, with the condition, `respondent`
# (completed or removed), the Likert items and age as numbers and the closed
# questions in lower case. The survey tests in stats/ and the survey figure
# read the same table.

# The answer options of the two closed demographic questions (the option values
# of client/src/intro-exit/ExitSurvey.jsx) and the macro fragment of each.
GENDER_OPTIONS = {
    "female": "Female",
    "male": "Male",
    "non-binary": "NonBinary",
    "other": "Other",
    "prefer-not-to-say": "NotSaid",
}
EDUCATION_OPTIONS = {
    "less-than-high-school": "LessThanHighSchool",
    "high-school": "HighSchool",
    "some-college": "SomeCollege",
    "bachelors": "Bachelors",
    "advanced-degree": "AdvancedDegree",
}


def has_field(survey: pd.DataFrame, field: str) -> bool:
    return field in survey.columns and survey[field].notna().any()


def felt_human_flags(survey: pd.DataFrame, level: str = "game") -> pd.DataFrame:
    """Games (or original groups) in which any respondent said they were not playing with humans."""
    keys = ["gameId", "condition"] if level == "game" else ["gameId", "condition", "originalGroup"]
    if not has_field(survey, "feltHuman"):
        return pd.DataFrame(columns=keys + ["said_no"])
    said_no = survey.assign(said_no=(survey["feltHuman"] == "no").astype(int)).groupby(keys)["said_no"].sum().reset_index()
    return said_no[said_no["said_no"] > 0]


def survey_macros(t: dict) -> dict:
    games, players, survey = t["games"], t["players"], t["exit_survey"]
    real_players = players[players["gameId"].isin(games.loc[games["condition"].notna(), "gameId"])]
    macros = {
        "surveyNRespondents": len(survey),
        "surveyNPlayers": len(real_players),
        "surveyNRemovedRespondents": int((survey["respondent"] == "removed").sum()) if not survey.empty else 0,
    }

    def item(prefix, field):
        if not has_field(survey, field):
            return {}
        v = survey[field].dropna()
        return {f"{prefix}N": len(v), f"{prefix}Mean": fmt_stat(v.mean(), 2), f"{prefix}SD": fmt_stat(v.std(ddof=1), 2)}

    def yes(prefix, field):
        if not has_field(survey, field):
            return {}
        v = survey[field].dropna()
        return {f"{prefix}N": len(v), f"{prefix}Yes": int((v == "yes").sum()), f"{prefix}PropYes": fmt_stat((v == "yes").mean())}

    def options(prefix, field, names):
        """How many answered a closed question, and the count and percentage choosing each option."""
        if not has_field(survey, field):
            return {}
        v = survey[field].dropna()
        unknown = sorted(set(v) - set(names))
        if unknown:
            raise SystemExit(f"Error: exit-survey item '{field}' has answers outside its options: {unknown}")
        out = {f"{prefix}N": len(v)}
        for value, name in names.items():
            out[f"{prefix}{name}"] = int((v == value).sum())
            out[f"{prefix}{name}Pct"] = fmt_stat(100 * (v == value).mean(), 1)
        return out

    macros.update(item("surveyIdentification", "groupIdentification"))
    macros.update(item("surveyCloseness", "groupCloseness"))
    macros.update(yes("surveyGroupLanguage", "groupLanguage"))
    macros.update(item("surveyAge", "age"))
    if has_field(survey, "age"):
        age = survey["age"].dropna()
        macros.update({"surveyAgeMin": fmt_count(age.min()), "surveyAgeMax": fmt_count(age.max())})
    macros.update(options("surveyGender", "gender", GENDER_OPTIONS))
    macros.update(options("surveyEducation", "education", EDUCATION_OPTIONS))
    if has_field(survey, "feltHuman"):
        fh = survey[survey["feltHuman"].notna()]
        macros.update({
            "surveyFeltHumanN": len(fh),
            "surveyFeltHumanNo": int((fh["feltHuman"] == "no").sum()),
            "surveyFeltHumanPropNo": fmt_stat((fh["feltHuman"] == "no").mean()),
            "surveyNGamesFlagged": len(felt_human_flags(survey)),
            "surveyNGroupsFlagged": len(felt_human_flags(survey, level="group")),
            "surveyFeltHumanPropNoSeparated": fmt_stat((fh.loc[fh["condition"] == "refer_separated", "feltHuman"] == "no").mean()),
            "surveyFeltHumanPropNoMixed": fmt_stat((fh.loc[fh["condition"] != "refer_separated", "feltHuman"] == "no").mean()),
        })
    if has_field(survey, "understood"):
        u = survey["understood"].dropna()
        macros.update({"surveyUnderstoodN": len(u), "surveyUnderstoodYes": int((u == "yes").sum())})
    return macros


def main():
    parser = argparse.ArgumentParser(description="Write the article's descriptive macros for one dataset")
    add_dataset_argument(parser)
    add_allow_removed_argument(parser)
    args = parser.parse_args()
    dirs = dataset_dirs(args.dataset)
    if not (dirs.tables / "trials.csv").exists():
        raise SystemExit(f"Error: no processed data for dataset '{dirs.name}' at {dirs.tables}")
    if dirs.name != "full":
        print(f"Dataset '{dirs.name}' is not the full sample: the macros written to {dirs.macros} are not results.")
    t = load_tables(dirs)
    if t["gs_by_game"].empty:
        print(f"  note: {dirs.estimates / 'primary' / 'group_specificity.csv'} is missing; run "
              "`Rscript stats/primary/group_specificity.R` (or `make estimates`) first for the "
              "Phase 1 group-specificity macros")
    files = {
        stats_file("overview", dirs): overview_macros(t),
        stats_file("exploratory", dirs): exploratory_macros(t),
        stats_file("survey", dirs): survey_macros(t),
    }
    write_stats_files(files, source=SOURCE, dataset=dirs.name, allow_removed=args.allow_removed)


if __name__ == "__main__":
    main()
