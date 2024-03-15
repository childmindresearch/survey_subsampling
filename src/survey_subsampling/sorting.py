#!/usr/bin/env python

import numpy as np
import pandas as pd


def aggregate_sort(learners_dataframe: pd.DataFrame, x_ids: list):
    # Melt the dataframe into long form
    melted_learners_df = learners_dataframe.reset_index()
    melted_learners_df = melted_learners_df.melt(
        id_vars=["Dx"], value_vars=x_ids, value_name="importance"
    )

    # Sort based on aggregate feature importance
    sort_agg = (
        melted_learners_df.groupby("variable")
        .sum()
        .reset_index()
        .sort_values("importance")["variable"]
        .values[::-1]
    )
    return melted_learners_df, sort_agg


def topn_sort(learners_dataframe: pd.DataFrame, x_ids: list):
    # Redo sorting and plotting with the top-N approach
    # Initialize an empty list of questions, to be populated iteratively.
    N = len(x_ids)
    item_relevance = np.empty((N, len(x_ids)))

    # For each threshold of "we can only include N questions..."
    for n in range(N):
        # Record the proportion of diagnoses for which a given question belongs
        item_relevance[n, :] = (
            learners_dataframe[x_ids]
            .rank(axis=1, ascending=False)
            .apply(lambda x: x <= n + 1)
            .sum(axis=0)
        )

    # Sort the prevalance table we just built, and apply it to the questions
    idx_topn = np.lexsort(item_relevance[::-1, :])[::-1]
    sort_topn = np.array(x_ids)[idx_topn]
    return item_relevance, sort_topn, idx_topn
