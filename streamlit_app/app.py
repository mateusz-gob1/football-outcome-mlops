"""Streamlit demo: Premier League match outcome predictions vs bookmaker odds.

Ships with pre-computed out-of-fold predictions (streamlit_app/data/*.csv),
generated once by prepare_data.py from the real walk-forward validation
results - so this app has no model or feature-engineering dependency at
runtime. See the project's vault/Evaluation-Log.md for the full methodology.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data"

st.set_page_config(
    page_title="Football Outcome Predictor", page_icon="⚽", layout="wide"
)

RESULT_NAMES = {"H": "Home", "D": "Draw", "A": "Away"}


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "predictions.csv", parse_dates=["Date"])


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def pick_for(row: pd.Series, prefix: str) -> str:
    probs = {
        "Home": row[f"{prefix}_prob_H"],
        "Draw": row[f"{prefix}_prob_D"],
        "Away": row[f"{prefix}_prob_A"],
    }
    return max(probs, key=probs.get)


st.title("⚽ Premier League Match Outcome Predictor")
st.caption(
    "Random Forest trained with walk-forward validation, benchmarked against bookmaker odds. "
    "Not a live prediction service - browses real out-of-fold results from the 2015/16-2025/26 seasons."
)

tab1, tab2 = st.tabs(["Browse Matches", "Model Performance"])

predictions = load_predictions()

with tab1:
    st.subheader("Historical matches: model vs bookmaker vs actual result")

    col1, col2 = st.columns(2)
    with col1:
        seasons = sorted(predictions["season"].unique(), reverse=True)
        season = st.selectbox(
            "Season (start year)",
            seasons,
            format_func=lambda y: f"{y}/{str(y + 1)[-2:]}",
        )
    with col2:
        teams = sorted(set(predictions["HomeTeam"]) | set(predictions["AwayTeam"]))
        team_filter = st.selectbox("Filter by team (optional)", ["All teams"] + teams)

    filtered = predictions[predictions["season"] == season].copy()
    if team_filter != "All teams":
        filtered = filtered[
            (filtered["HomeTeam"] == team_filter)
            | (filtered["AwayTeam"] == team_filter)
        ]
    filtered = filtered.sort_values("Date")

    filtered["Model pick"] = filtered.apply(lambda r: pick_for(r, "model"), axis=1)
    filtered["Bookmaker pick"] = filtered.apply(
        lambda r: pick_for(r, "bookmaker"), axis=1
    )
    filtered["Actual"] = filtered["actual_result"].map(RESULT_NAMES)
    filtered["Model correct"] = (
        filtered["Model pick"].str[0] == filtered["actual_result"]
    )
    filtered["Bookmaker correct"] = (
        filtered["Bookmaker pick"].str[0] == filtered["actual_result"]
    )

    st.dataframe(
        filtered[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "Model pick",
                "Bookmaker pick",
                "Actual",
                "Model correct",
                "Bookmaker correct",
            ]
        ].rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"}),
        width="stretch",
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches shown", len(filtered))
    c2.metric("Model accuracy", f"{filtered['Model correct'].mean():.1%}")
    c3.metric("Bookmaker accuracy", f"{filtered['Bookmaker correct'].mean():.1%}")

with tab2:
    st.subheader("Log-loss & Brier score: model vs bookmaker")
    st.caption(
        "Log-loss is the primary metric (a proper scoring rule) - lower is better. "
        "Random guessing over 3 classes scores about 1.10; see the project README for why."
    )
    st.dataframe(load_csv("model_summary.csv"), width="stretch", hide_index=True)

    st.subheader("Is the difference real, or just noise?")
    st.caption(
        "Bootstrap 95% CI on (model log-loss minus bookmaker log-loss). Positive means the model is worse."
    )
    st.dataframe(
        load_csv("bootstrap_ci_vs_bookmaker.csv"), width="stretch", hide_index=True
    )
    st.markdown(
        "All three intervals sit entirely above zero - none of the trained models beat the bookmaker "
        "baseline, and the gap is statistically significant, not noise."
    )

    st.subheader("Calibration (Random Forest, home-win predictions)")
    st.caption(
        "If the model says '70% chance of a home win', that should actually happen about 70% of the time."
    )
    calibration = load_csv("random_forest_calibration_home.csv")
    st.line_chart(
        calibration.rename(
            columns={"mean_predicted": "Predicted", "mean_actual": "Actual"}
        ).set_index("bin")[["Predicted", "Actual"]]
    )

    st.subheader("What drives the model's predictions?")
    importance = load_csv("random_forest_feature_importance.csv").head(10)
    st.bar_chart(importance.set_index("feature")["importance"])
