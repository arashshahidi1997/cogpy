import numpy as np
import pandas as pd

def normalized_sum(x):
    return np.sum(x) / np.sqrt((np.abs(x)**2).sum())

def data_agg(x_list):
    return np.concatenate([x.data for x in x_list])

def dim_agg(x_list, dim):
    return np.concatenate([x.coords[dim] for x in x_list])

def classify_detections(detection_df: pd.DataFrame, ground_truth_df: pd.DataFrame, ordered_cols: list):
    """
    Classify detections into True Positives (TP), False Positives (FP), and False Negatives (FN).

    Parameters:
        detection_df (pd.DataFrame): DataFrame with detected items
        ground_truth_df (pd.DataFrame): DataFrame with ground truth items
        ordered_cols (list): List of column names to match on

    Returns:
        dict: Dictionary with DataFrames for TP, FP, FN
    """

    # True Positives: detection matches ground truth
    true_positives = pd.merge(detection_df, ground_truth_df, on=ordered_cols, how='inner')

    # False Positives: detection not in ground truth
    false_positives = detection_df.merge(ground_truth_df, on=ordered_cols, how='left', indicator=True)
    false_positives = false_positives[false_positives['_merge'] == 'left_only'].drop(columns=['_merge'])

    # False Negatives: ground truth not detected
    false_negatives = ground_truth_df.merge(detection_df, on=ordered_cols, how='left', indicator=True)
    false_negatives = false_negatives[false_negatives['_merge'] == 'left_only'].drop(columns=['_merge'])

    results = {
        "TruePositives": true_positives,
        "FalsePositives": false_positives,
        "FalseNegatives": false_negatives
    }
    rate_fn = lambda x, total: x / total if total > 0 else 0
    tp = len(results['TruePositives'])
    fp = len(results['FalsePositives'])
    fn = len(results['FalseNegatives'])

    # precision and recall
    miss = rate_fn(fn, (tp + fn))
    precision = rate_fn(tp, (tp + fp))

    performance_table = pd.DataFrame({
        'Category': ['miss', 'precision'],
        'Performance': [miss, precision],
    })
    performance_table = performance_table.set_index('Category')
    results['Performance'] = performance_table
    return results
