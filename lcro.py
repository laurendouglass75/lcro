import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings('ignore')

def custom_qcut(df, pred_colname, actual_colname, n_deciles):
    """"
    custom implementation of qcut to not throw boundary tie errors

    adds columns pred_colname_bin and actual_colname_bin to df with the prediction and actual bin numbers.

    ranks data, then performs qcut from the rank. data earlier in the df will get a lower rank.

    :param df: DataFrame to qcut on
    :param pred_colname: name of prediction column (str)
    :param actual_colname: name of actuals column (str)
    :param n_deciles: Number of bins

    :return: copy of df with pred_colname_bin and actual_colname_bin added
    """
    df.loc[:, 'pred_rank'] = df.loc[:, pred_colname].rank(method = 'first')
    df.loc[:, 'act_rank'] = df.loc[:, actual_colname].rank(method = 'first')
    df.loc[:, pred_colname + '_bin'] = pd.qcut(df.loc[: ,'pred_rank'], n_deciles, labels=False) + 1
    df.loc[:, actual_colname + '_bin'] = pd.qcut(df.loc[: ,'act_rank'], n_deciles, labels=False) + 1
    df = df.drop(['pred_rank', 'act_rank'], axis = 1)
    return df

def custom_cut(df, pred_colname, actual_colname, tiercuts):
    """
    Performs pd.cut on df[pred_colname] using tiercuts, then finds the number of rows in each tier,
    and cuts the actuals into tiers by sorting them, then making tiers that contain the same number
    of rows as the prediction tiers.

    :param df: dataframe to cut with predictions and actuals
    :param pred_colname: name of prediction column (str)
    :param actual_colname: name of actuals column (str)
    :param tiercuts: tier edges to use to cut predictions (list of scalars or list of tuples)
    :return: df with columns 'pred_bin' and 'actual_bin' designating each row's tier number
        based on predictions and actuals
    """
    if isinstance(tiercuts, list) and all(isinstance(i, tuple) for i in tiercuts):
        tiercuts = pd.IntervalIndex.from_tuples(tiercuts)

    # TODO: validate that the labels work regardless of the type of tiercuts
    df['pred_bin'] = pd.cut(df[pred_colname], tiercuts, labels = range(1, len(tiercuts)))
    tier_counts = df['pred_bin'].value_counts().sort_index().reset_index()

    # Sort the actual values
    sorted_actuals = df.sort_values(by=actual_colname).reset_index(drop=True)

    # Generate actual bins by cumulative record counts
    actual_bins = []
    for i, count in enumerate(tier_counts['count']):
        actual_bins += [tier_counts['pred_bin'][i]] * count

    # Assign actual bins back to the DataFrame
    sorted_actuals['actual_bin'] = actual_bins

    return sorted_actuals

def generate_lcro_report(df, pred_agg, act_agg, slope, monotonicity, monotonicity_param, combined_metric, slope_std, n_bins, plot_actuals):
    plot_max = act_agg['actual'].iloc[-1]*1.05
    plot_min = act_agg['actual'].iloc[0]*0.9


    # barplot of prediction lift chart

    if plot_actuals:
        plt.figure(figsize = (10, 4))
        plt.subplot(1, 2, 1)
    sns.barplot(x = pred_agg['pred_bin'], y = pred_agg['actual'], errorbar = None)
    if plot_actuals:
        plt.ylim((plot_min, plot_max))
    plt.xlabel('Prediction Tier')
    plt.ylabel('Average Actual')
    plt.title('Lift Chart of Predictions, {} Tiers'.format(n_bins))

    # barplot of actuals
    if plot_actuals:
        plt.subplot(1, 2, 2)
        sns.barplot(x = act_agg['actual_bin'], y = act_agg['actual'], errorbar = None)
        plt.ylim((plot_min, plot_max))
        plt.xlabel('Actual Tier')
        plt.ylabel('Average Actual')
        plt.title('Lift Chart of Actuals, {} Tiers (Perfect Model)'.format(n_bins))

        plt.tight_layout()
    plt.show()

    # calculate other metrics
    # average standard deviation within tiers

    if plot_actuals:
        plt.figure(figsize = (10, 4))
        plt.subplot(1, 2, 1)
    std_agg = df.groupby('pred_bin', observed = False)['actual'].std().reset_index()
    plot_max = max(std_agg['actual'])*1.05
    sns.barplot(x = std_agg['pred_bin'], y = std_agg['actual'], ci = None)
    plt.title('Standard Deviation of Actuals Within Prediction Tier')
    plt.xlabel('Prediction Tier')
    plt.ylabel('Standard Deviation of Actuals')
    plt.ylim((0, plot_max))

    if plot_actuals:
        plt.subplot(1, 2, 2)
        std_agg = df.groupby('actual_bin', observed = False)['actual'].std().reset_index()
        sns.barplot(x = std_agg['actual_bin'], y = std_agg['actual'], ci = None)
        plt.title('Standard Deviation of Actuals Within Actual Tier')
        plt.xlabel('Actual Tier')
        plt.ylabel('Standard Deviation of Actuals')
        plt.ylim((0, plot_max))

        plt.tight_layout()
    plt.show()

    # display metrics
    print('\n')
    print('-----------------------------------------------------------------------------')
    print('LIFT CHART RANK ORDER REPORT')
    print('Monotonicity: {:.4f} -- Penalized Monotonicity with Param {}: {:.4f}'.format(monotonicity, monotonicity_param, (monotonicity - monotonicity_param) / (1 - monotonicity_param)))
    print('Slope Score: {:.4f}'.format(slope))
    print('Combined Metric: {:.4f}'.format(combined_metric))
    print('\n')
    print('Standard deviation of tier-to-tier deltas: {:.4f}'.format(slope_std))
    print('-----------------------------------------------------------------------------')


def rank_order(actuals, predictions, n_bins = 13, tiercuts = None, monotonicity_param = 0.9, generate_report = False, plot_actuals = False):
    """
    Given two arrays of predictions and actuals, calculate custom rank order score("LCRO").
    This is performed in the following way:

    1. Cut predictions into n_bins tiers, and find the average actual of each tier.
    2. Calculate a monotonicity score: the extent to which each tier's average is greater
        than the previous tier's (measured as spearman correlation between tier number
        and tier average).
    2.5. Penalize monotonicity according to monotonicity_param (default 0.9) -
        assuming that our models will generally have monotonicity > 0.9, convert monotonicity
        m to (m - 0.9) / 0.1. This converts a score of 0.95 to 0.5, 0.99 to 0.9, 0.91 to 0.1, etc.
        If monotonicity < 0.9, we will get a negative LCRO.
    3. Calculate a slope score to indicate what percentage of possible tier separation is achieved:
        i. Take the difference of the average actual for the highest prediction tier and the lowest
            prediction tier
        ii. Cut the actuals into the same number of tiers, and take the difference of the average actual
            for the highest actual tier and the average actual for the lowest actual tier. This represents
            what the slope would be of a model that rank ordered perfectly.
        iii. Divide item i) by item ii) to get a final slope score: percentage of possible tier
            separation achieved.
    4. Return monotonicity * slope as the rank order score - the score is primarily based on
        slope, but decreases for imperfect monotonicity.

    :param actuals: Array of actuals.
    :param predictions: Array of predictions.
    :param generate_report: Bool, whether to print a detailed report
    :param n_bins: How many tiers to cut predictions into for calculation. Default = 30
    :param monotonicity_param: What Spearman correlation should set the penalized monotonicity score to 0, or the
        minimum monotonicity we might expect. Default = 0.9
    :param tiercuts: List of scalars or list of tuples, explicit tiers to cut predictions into. Default: None
    :param plot_actuals: Bool, whether to plot the charts for actuals in report.

    :return: Combined LCRO score
    """
    df = pd.DataFrame({'pred': predictions, 'actual': actuals})
    if tiercuts is not None:
        df = custom_cut(df, 'pred', 'actual', tiercuts)
    else:
        df = custom_qcut(df, 'pred', 'actual', 13)
    pred_agg = df.groupby('pred_bin', observed = False)['actual'].mean().reset_index()
    act_agg = df.groupby('actual_bin', observed = False)['actual'].mean().reset_index()


    deltas = [pred_agg['actual'].iloc[i + 1] - pred_agg['actual'].iloc[i] for i in range(n_bins - 1)]
    slope_std = np.std(deltas)

    # spearman correlation of pred tier number and average actual
    monotonicity = spearmanr(pred_agg['pred_bin'], pred_agg['actual'])[0]

    # calculate slope score
    pred_slope = pred_agg['actual'].iloc[-1] - pred_agg['actual'].iloc[0]
    perfect_slope = act_agg['actual'].iloc[-1] - act_agg['actual'].iloc[0]
    slope = pred_slope / perfect_slope

    # apply scaling to monotonicity and return the score
    combined_metric = (monotonicity - monotonicity_param) / (1 - monotonicity_param) * slope

    if generate_report:
        generate_lcro_report(df, pred_agg, act_agg, slope, monotonicity, monotonicity_param, combined_metric, slope_std, n_bins, plot_actuals)


    return combined_metric