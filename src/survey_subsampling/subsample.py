#!/usr/bin/env python

from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser
from os import PathLike

from sklearn.metrics import accuracy_score, f1_score, classification_report, class_likelihood_ratios
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

from survey_subsampling.core import constants
from survey_subsampling.core.learner import Learner
from survey_subsampling import plotting
from survey_subsampling import sorting


def load_data(infile: PathLike, threshold: int=50, verbose: bool=True):
    """Grabs a parquet file, extracts the columns we want, removes NaNs, and returns"""
    # Grabs data file from disk, sets diagnostic labels to 0 or 1.
    df_full = pd.read_parquet(infile)
    df_full[constants.Dx_labels_all] = df_full[constants.Dx_labels_all].replace({2.0:1, 0.0:0})

    # Define column selector utility that we'll iteratively use to subsample the dataset.
    def _column_selector(df: pd.DataFrame, columns: list, drop: bool=False):        
        return df[columns].dropna(axis=0, how='any') if drop else df[columns]

    def _get_prevalance(df: pd.DataFrame, diagnoses: list):
        tmp = []
        for dx in diagnoses:
            vc = df[dx].value_counts()
            tmp += [{"Dx": dx, "HC": vc.loc[0.0], "Pt":vc.loc[1.0]}]
        return pd.DataFrame.from_dict(tmp).set_index('Dx').sort_values(by='Pt')

    # Initialize the dataset cleaning
    # Subset table based on all diagnoses, compute prevalance
    df = _column_selector(df_full, constants.CBCLABCL_items + constants.Dx_labels_all, drop=False)
    df_prev = _get_prevalance(df, diagnoses=constants.Dx_labels_all)

    # Drop low-prevalance diagnoses right away from sparse dataset
    #        full list of diagnoses  -  all diagnoses with low prevalance
    Dx_labels_subset = list(set(df_prev.index) - set(df_prev[df_prev['Pt'] < threshold].index))

    # Prepare to iteratively repeat the process, as the N of various conditions may change as
    # we prune missing data and subsequently change the included column lists
    low_N = True
    while low_N:
        # Repeat dataset table subsetting (densely this time), compute prevalance, and drop low N
        df = _column_selector(df_full, constants.CBCLABCL_items + Dx_labels_subset, drop=True)
        df_prev = _get_prevalance(df, diagnoses=Dx_labels_subset)

        # Grab a dataframe of the low-prevalance diagnoses...
        low_N_df = df_prev[df_prev['Pt'] < threshold]
        if low_N := (len(low_N_df.index) > 0):
            # ... and remove them from the set of consideration, then go again
            Dx_labels_subset = list(set(Dx_labels_subset)-set(low_N_df.index))

    # Report on prevalance table and overall dataset length
    if verbose:
        print(df_prev)
        print("Original Dataset Length:", len(df_full))
        print("Pruned Dataset Length:", len(df))

    return df, df_prev, Dx_labels_subset


def fit_models(df: pd.DataFrame, x_ids: list, y_ids: list, verbose: bool=True):
    """General purpose function for fitting callibrated classifiers for Dx from Survey data"""
 
    # Establish Models & Cross-Validation Strategy
    #   Base classifier: Random Forest. Rationale: non-parametric, has feature importance
    clf_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    #   CV: Stratified K-fold. Rationale: shuffle data, balance classes across folds
    cv = StratifiedKFold(shuffle=True, random_state=42)
    #   Top-level classifier: Calibrated CV classifier. Rationale: prioritizes maintaining class balances
    clf_calib = CalibratedClassifierCV(estimator=clf_rf, cv=cv)
    np.random.seed(42)

    # Create X (feature) matrix: grab relevant survey columns x rows from dataframe
    X = df[x_ids].values

    # Create empty list of learners
    learners = []
    summaries = []
    # For every Dx that we want to predict...
    for y_name in y_ids:
        # Create the y (target) matrix/vector: grab relevant Dx rows from dataframe
        y = df[y_name].values.astype(int)
        
        # Get Pt and HC counts from dataframe, and initialize the learner object
        _, uc = np.unique(y, return_counts=True)

        if verbose:
            print(f"Dx: {y_name} | HC: {uc[0]} | Pt: {uc[1]}")

        current_learner = Learner(dx=y_name, hc_n=uc[0], dx_n=uc[1], x_ids=x_ids)
        # Set-up a CV loop (note, we use the same CV strategy both within the Calibrated CLF and here,
        #   resulting in nested-stratified-k-fold-CV)
        for idx_train, idx_test in cv.split(X, y):
            # Split the dataset into train and test sets
            X_tr = X[idx_train, :]
            y_tr = y[idx_train]

            X_te = X[idx_test, :]
            y_te = y[idx_test]

            # Fit the callibrated classifier on the training data
            clf_calib.fit(X_tr, y_tr)

            # Extract/Generate relevant data from (all internal folds of) the classifier...
            # Make predictions on the test set
            y_pred = clf_calib.predict(X_te)
            
            # Grab training and validation performance using the integrated scoring function (accuracy)
            y_pred_tr = clf_calib.predict(X_tr)
            current_learner.acc_train.append(accuracy_score(y_tr, y_pred_tr))
            current_learner.acc_valid.append(accuracy_score(y_te, y_pred))

            # Grab feature importance scores
            fis = [_.estimator.feature_importances_ for _ in clf_calib.calibrated_classifiers_]
            current_learner.fi.append(fis)

            # Grab the prediction probabilities
            current_learner.proba.append(clf_calib.predict_proba(X_te))
            current_learner.label.append(y_te)

            # Grab the prediction labels
            current_learner.f1.append(f1_score(y_te, y_pred))

            # Grab the sensitivity and specificity (i.e. recall of each of Dx and HC classes)
            report_dict = classification_report(y_te, y_pred, output_dict=True)
            current_learner.sen.append(report_dict['1']['recall'])
            current_learner.spe.append(report_dict['0']['recall'])

            # Grab the positive/negative likelihood ratios
            lrp, lrn = class_likelihood_ratios(y_te, y_pred)
            current_learner.LRp.append(lrp)
            current_learner.LRn.append(lrn)

        # Summarize current learner performance, save it, and get ready to go again!
        summaries += [current_learner.summary()]

        tmp_learner = {'Dx': current_learner.dx}

        means = np.mean(np.vstack(current_learner.fi), axis=0)
        for assessment, importance in zip(x_ids, means):
            tmp_learner[assessment] = importance
        learners += [tmp_learner]

        del current_learner
    
    # Improve formatting of summaries and complete learners
    summaries = pd.concat(summaries).set_index('Dx')
    learners = pd.DataFrame.from_dict(learners).set_index('Dx')

    return learners, summaries


def calculate_feature_importance(learners: pd.DataFrame, x_ids: list, outdir: PathLike, number_of_questions: int=20, plot=True):
    """Visualizes feature importance and sorts values using two strategies: aggregate and top-N"""
    
    learners_sorted_by_aggregate, x_ids_sorted_by_aggregate = sorting.aggregate_sort(learners, x_ids=x_ids, number_of_questions=number_of_questions)
    relevance_sorted_by_topn, x_ids_sorted_by_topn, x_idx_topn = sorting.topn_sort()

    if plot:
        n_diagnoses = len(learners)
        plotting.many_learner_feature_importance_stacked(learners_sorted_by_aggregate, x_ids_sorted_by_aggregate)
        plotting.many_learner_feature_importance_stacked(relevance_sorted_by_topn, x_ids_sorted_by_topn, x_idx_topn, n_dx=n_diagnoses)

    # Report the sorted list using both approaches
    sorted_importance_agg = x_ids_sorted_by_aggregate[0:number_of_questions]
    print(f"The {number_of_questions} cumulatively most predictive survey questions overall are:", ", ".join(sorted_importance_agg))
    
    sorted_importance_topn = x_ids_sorted_by_topn[0:number_of_questions]
    print(f"The {number_of_questions} most commonly useful survey questions overall are:", ", ".join(sorted_importance_topn))

    # Evaluate sorting consistency
    all_qs = list(set(list(sorted_importance_agg) + list(sorted_importance_topn)))
    diff = np.abs(number_of_questions-len(all_qs))
    frac = 1.0*diff/number_of_questions*100
    print(f"The two lists differ by {diff} / {number_of_questions} items ({frac:.2f}%)")

    # Compute the average position across the two methods and produce a 3rd and final sorting
    avg_rank = np.mean(np.where(x_ids_sorted_by_aggregate[:, None] == x_ids_sorted_by_topn), axis=0)
    x_ids_sorted_average = x_ids_sorted_by_aggregate[np.argsort(avg_rank)]

    return x_ids_sorted_by_aggregate, x_ids_sorted_by_topn, x_ids_sorted_average


def degrading_fit(df: pd.DataFrame, sorted_x_ids: list, y_ids: list, threads: int=4, verbose: bool=True):
    """Routine that wraps the `fit_models` function, gradually degrading the performance by reducing the x-set"""
    # Initiatlize some storage containers
    degraded_tuple = []
    futures = []

    with ProcessPoolExecutor(max_workers=threads) as pool:
        # For each number of questions (from the length of sorted_x_ids down to 1)...
        for n_questions in range(len(sorted_x_ids))[::-1]:
            # Grab the first n_questions from the sorted list
            xi = sorted_x_ids[0:n_questions+1]

            # Add the diagnostic prediction models to the queue using this reduced x-set
            #   Equivalent command: degraded_tuple = fit_models(df, xi, y_ids, verbose=False)
            futures.append(pool.submit(fit_models, df, xi, y_ids, verbose=False))

    # Store the results as they come in
    for future in as_completed(futures):
        degraded_tuple.append(future.result())
    
    # Separate the learners and respective summaries
    degraded_learners = [dt[0] for dt in degraded_tuple]
    degraded_summaries = [dt[1] for dt in degraded_tuple]
    
    # Concatenate and return the learners and summaries
    degraded_summaries = pd.concat(degraded_summaries)
    degraded_learners = pd.concat(degraded_learners)
    return  degraded_learners, degraded_summaries


def run():
    parser = ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outdir")
    parser.add_argument("-n", "--number_of_questions", default=20, type=int)
    parser.add_argument("--random_state", default=42, type=int)
    parser.add_argument("-t", "--dx_threshold", default=150, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-w", "--warnings", action="store_true")
    parser.add_argument("--n_threads", default=4, type=int)
    args = parser.parse_args()

    # Grab input values
    # infile = '../../data/converted/harmonized_allwaves.parquet'
    # outdir = '../../data/figures/'
    infile = args.infile
    outdir = args.outdir
    threshold = args.dx_threshold
    verbose = args.verbose
    NQ = args.number_of_questions
    nt = args.n_threads

    np.random.seed(args.random_state)

    # Suppress the many warnings that come up when training degrading learners by default
    if not args.warnings:
        import warnings
        warnings.filterwarnings("ignore")

    # Load the dataset and remove incomplete subjects and under-represented diagnoses
    df, df_prev, Dx_labels_subset = load_data(infile, threshold=threshold, verbose=verbose)

    # Establish baseline prediction, and further remove diagnostic labels for which the models
    #  fail to make reasonable predictions (read as: make predictions to both classes;
    #  identifiable as diagnoses with a NaN for LR+)
    learners, summaries = fit_models(df, constants.CBCLABCL_items, Dx_labels_subset, verbose=verbose)
    Dx_labels_subset = list(set(Dx_labels_subset) - set(summaries[summaries['LR+'].isna()].index))
    learners = learners.loc[Dx_labels_subset]
    learners.to_parquet(f'{outdir}/learners.parquet')

    summaries = summaries.loc[Dx_labels_subset]
    summaries.to_parquet(f'{outdir}/summaries.parquet')
    print(summaries)

    # Compute and plot feature importance, and then save results in a CSV file
    sorted_agg, sorted_topn, sorted_avg = calculate_feature_importance(learners, constants.CBCLABCL_items, outdir, number_of_questions=NQ)
    importance = pd.DataFrame({'Aggregate': sorted_agg, 'Top-N': sorted_topn, 'Average': sorted_avg})
    importance.to_parquet(f'{outdir}/feature_importance.parquet')

    # Finally, redo the learning process with a degrading set of data, iteratively removing questions
    learners_deg, summaries_deg = degrading_fit(df, sorted_avg, Dx_labels_subset, threads=nt, verbose=verbose)
    summaries_deg.to_parquet(f'{outdir}/summaries_degraded.parquet')
    learners_deg.to_parquet(f'{outdir}/learners_degraded.parquet')


if __name__ == "__main__":
    run()
