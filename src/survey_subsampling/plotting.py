#!/usr/bin/env python
"""Contains plotting functions and runscript."""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from survey_subsampling import sorting
from survey_subsampling.core import constants
from survey_subsampling.core.learner import Learner


def single_learner_probability_distribution(learner: Learner) -> Figure:
    """Show prediction confidence scores."""
    tmpdict = {"proba": learner.proba[:, 1], "label": learner.label}
    tmpdf = pd.DataFrame.from_dict(tmpdict)
    fig = px.histogram(
        tmpdf,
        color="label",
        x="proba",
        marginal="box",
        barmode="overlay",
        opacity=0.7,
        category_orders={"label": [0, 1]},
    )
    fig.show()


def many_learner_feature_importance_stacked(
    learners: pd.DataFrame, x_ids_sorted: np.ndarray, number_of_questions: int = 20
) -> Figure:
    """Plot a stacked-bar of feature importance across diagnoses."""
    fig = px.bar(
        learners,
        x="variable",
        y="importance",
        color="Dx",
        title="Relative feature importance of survey questions across diagnoses",
        labels={
            "variable": "CBCL Question",
            "importance": "Relative Feature Importance",
            "target": "Diagnosis",
        },
        template="plotly_white",
    )

    # Update plot to reflect sorting, and target number of questions
    fig.update_xaxes(categoryorder="array", categoryarray=x_ids_sorted)
    fig.update_layout(
        width=1620, height=900, legend={"orientation": "v", "y": 0.97, "x": 0.84}
    )
    fig.add_vline(
        x=x_ids_sorted[number_of_questions + 1],
        line_width=1,
        line_dash="dash",
        line_color="gray",
    )
    fig.show()
    return fig


def many_learner_feature_importance_heatmap(
    item_relevance: np.ndarray,
    x_ids_sorted: np.ndarray,
    idx_sorted: list,
    n_diagnoses: int,
    number_of_questions: int = 20,
) -> Figure:
    """Plot a heatmap of feature importance across topN selection for each diagnosis."""
    fig = px.imshow(
        item_relevance[:, idx_sorted] * 1.0 / n_diagnoses,
        x=x_ids_sorted,
        y=np.arange(len(x_ids_sorted)) + 1,
        labels={
            "x": "CBCL Question",
            "y": "Number of Questions (N)",
            "color": "Top-N Fraction",
        },
        title="Consistency of question usefulness across diagnoses",
        template="plotly_white",
    )

    # Update plot to reflect the target number of questions
    fig.update_layout(width=1620, height=900)
    fig.add_vline(
        x=x_ids_sorted[number_of_questions + 1],
        line_width=1,
        line_dash="dash",
        line_color="gray",
    )
    fig.show()
    return fig


def run() -> None:
    """Runner for plotting utility."""
    # TODO: improve docstrings, helptext, and the like
    parser = ArgumentParser()
    parser.add_argument("outdir")
    parser.add_argument("-n", "--number_of_questions", default=20, type=int)
    parser.add_argument(
        "-e", "--extension", type=str, default="pdf", choices=["png", "pdf", "html"]
    )

    args = parser.parse_args()
    outdir = args.outdir
    NQ = args.number_of_questions
    ext = args.extension

    # Try loading the relevant data
    try:
        learners = pd.read_parquet(f"{outdir}/learners.parquet")
        # summaries_deg = pd.read_parquet(f"{outdir}/summaries_degraded.parquet")
    except FileNotFoundError as e:
        e.strerror = "Plotting requires subsample analysis to be run, generating"
        raise (e)

    # First pass: re-sort the dataframes and plot feature importance
    learners_sorted_by_aggregate, x_ids_sorted_by_aggregate = sorting.aggregate_sort(
        learners, x_ids=constants.CBCLABCL_items
    )
    relevance_sorted_by_topn, x_ids_sorted_by_topn, x_idx_topn = sorting.topn_sort(
        learners, constants.CBCLABCL_items
    )

    n_diagnoses = len(learners)
    fig = many_learner_feature_importance_stacked(
        learners_sorted_by_aggregate, x_ids_sorted_by_aggregate, number_of_questions=NQ
    )
    fig.write_image(f"{outdir}/feature_importance_agg.{ext}")

    fig = many_learner_feature_importance_heatmap(
        relevance_sorted_by_topn,
        x_ids_sorted_by_topn,
        x_idx_topn,
        n_diagnoses=n_diagnoses,
        number_of_questions=NQ,
    )
    fig.write_image(f"{outdir}/feature_importance_topn.{ext}")

    # Second pass: plot degrading performance results
    # TODO: port from notebook


if __name__ == "__main__":
    run()
