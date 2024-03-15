#!/usr/bin/env python

import plotly.express as px
import pandas as pd
import numpy as np

from survey_subsampling.core.learner import Learner


def single_learner_probability_distribution(learner: Learner):
    """Show prediction confidence scores"""
    tmpdict = {
        'proba' : learner.proba[:,1],            
        'label' : learner.label
    }
    tmpdf = pd.DataFrame.from_dict(tmpdict)
    fig = px.histogram(tmpdf, color='label', x='proba', marginal='box',
                       barmode="overlay", opacity=0.7, category_orders={'label':[0,1]})
    fig.show()


def many_learner_feature_importance_stacked(learners: pd.DataFrame, x_ids_sorted: list, number_of_questions: int=20):
    # Plot a stacked-bar of feature importance across diagnoses, sorted by variable importance
    fig = px.bar(learners, x="variable", y="importance", color="Dx",
                title="Relative feature importance of survey questions across diagnoses",
                labels={"variable": "CBCL Question",
                        "importance": "Relative Feature Importance",
                        "target": "Diagnosis"},
                template='plotly_white')

    # Update plot to reflect sorting, and target number of questions
    fig.update_xaxes(categoryorder='array', categoryarray=x_ids_sorted)
    fig.update_layout(width=1620, height=900, legend={'orientation': 'v','y':0.97,'x':0.84})
    fig.add_vline(x=x_ids_sorted[number_of_questions+1], line_width=1, line_dash="dash", line_color="gray")
    fig.show()
    return fig
    # fig.write_image(f'{outdir}/feature_importance_agg.png')


def many_learner_feature_importance_heatmap(item_relevance, x_ids_sorted: list, idx_sorted: list, n_diagnoses: int, number_of_questions: int=20):
    # Plot a heatmap of feature importance across top-N selection for each diagnosis
    fig = px.imshow(item_relevance[:, idx_sorted]*1.0/n_diagnoses,
                    x=x_ids_sorted, y=np.arange(len(x_ids_sorted))+1,
                    labels={'x':'CBCL Question',
                            'y':'Number of Questions (N)',
                            'color':'Top-N Fraction'},
                    title="Consistency of question usefulness across diagnoses",
                    template='plotly_white')

    # Update plot to reflect the target number of questions
    fig.update_layout(width=1620, height=900)
    fig.add_vline(x=x_ids_sorted[number_of_questions+1], line_width=1, line_dash="dash", line_color="gray")
    fig.show()
    return fig
    # fig.write_image(f'{outdir}/feature_importance_topn.png')