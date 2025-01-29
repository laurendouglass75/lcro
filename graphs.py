"""
File with custom graph functions that you might find useful!
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

def simple_shap_plot(model, dataset, feature_list, check_additivity = True, **kwargs):
    """
    Creates a simple feature importance plot using shap values.

    The graph plots the "average shap value when this feature is >= 1" - so 
    when a one-hot feature is a "yes" or when a standard scaled numeric feature is
    at least one standard deviation above the mean.

    Customization is necessary for this to make sense for numerical features that are not standard scaled.

    :param model: model you have trained
    :param dataset: dataset you want to evaluate feature importance on
    :param feature_list: list of input features to the model (i.e. ['make', 'size'])
    :param check_additivity: set this to false if you get errors [default: True]
    :kwargs: any additional params you want to pass into the graphing function
    """
    explainer = shap.Explainer(model, dataset[feature_list])
    shap_values = explainer(dataset[feature_list], check_additivity = check_additivity)
    shap_vals = np.array(shap_values.values) # actual shap value for each feature for each row in dset
    shap_data = np.array(shap_values.data) # original data point for that feature & that row
    shap_values = np.stack((shap_vals, shap_data), axis = -1) # combine into one array
    positive_mask = shap_values[:, :, 1] >= 1 # mask where data point is >= 1
    positive_values = np.where(positive_mask, shap_values[:, :, 0], np.nan) # see shap values for data points that are >= 1
    mean_shap = np.nanmean(positive_values, axis = 0) # get avg shap for each feature when data point >= 1
    shap_df = pd.DataFrame({'Feature': feature_list, 'Importance': mean_shap})
    shap_df = shap_df.sort_values(by = 'Importance', ascending = False)
    plt.barh(shap_df['Feature'], shap_df['Importance'], **kwargs)

    plt.xticks([])
    plt.axvline(x = 0, color = 'black')
    plt.tight_layout()
    plt.show()

def performance_by_segment(dataset, numerical_feature_names, categorical_feature_names, prediction_col, target_col, n_quantiles_model = 10,
                           n_quantiles_numerical = 10, categorical_pct_threshold = 0.02, **kwargs):
    """
    Function to create charts like Daniel's that show model performance across different features.
    y-axis is the average actual value of the target divided by the average expectation for each observation's prediction tier.
    x-axis is the quantile for numerical features or category for categorical features.
    Categorical features are ordered from lowest counts to highest counts and only include levels that are at least
    categorical_pct_threshold of the data.

    :param dataset: Dataset to evaluate on
    :param numerical_feature_names: List of names of numerical features to chart.
    :param categorical_feature_names: List of names of categorical features to chart.
    :param prediction_col: Name of prediction column
    :param target_col: Name of target column
    :param n_quantiles_model: How many quantiles to cut model predictions into, i.e. AppScore Decile (Default: 10)
    :param n_quantiles_numerical: How many quantiles to cut numerical features into for visualization, i.e. VBC Decile (Default: 10)
    :param categorical_pct_threshold: Minimum proportion of the data a category should be to be included in the visualization. (Default: 0.02)
    :kwargs: any additional params you want to pass into the graphing function
    """
    dataset = dataset.copy()
    # create table with expectation by tier/quantile
    dataset['pred_quantile'] = pd.qcut(dataset[prediction_col], q = n_quantiles_model, labels = range(1, n_quantiles_model + 1))
    expectations = dataset.groupby('pred_quantile')[target_col].mean().reset_index().rename(columns = {target_col: 'target_expectation'})
    dataset = pd.merge(dataset, expectations, how = 'left', on = 'pred_quantile')
    dataset['perf_vs_exp'] = dataset[target_col] / dataset['target_expectation']

    # for each numerical feature
    for num_feat in numerical_feature_names:
        colname = num_feat + '_quantile'
        # cut feature into quantiles + aggregate
        dataset[colname] = pd.qcut(dataset[num_feat], q = n_quantiles_numerical, labels = range(1, n_quantiles_numerical + 1))
        feat_avg = dataset.groupby(colname)['perf_vs_exp'].mean().reset_index()

        # getting info to set graph y axis bounds
        feat_avg['diff'] = np.abs(1 - feat_avg['perf_vs_exp'])
        max_diff = np.max(feat_avg['diff'])

        # creating graph
        ax = sns.lineplot(feat_avg, x = colname, y = 'perf_vs_exp', label = 'Actual', **kwargs)
        plt.xlabel(num_feat + ' quantile')
        plt.ylabel('Performance vs. Expectation')
        plt.ylim(0.9 - max_diff, 1.1 + max_diff)
        plt.title(f'Performance vs. Expectation: {num_feat}')
        plt.axhline(1, label = 'Perfect', color = 'gray', linestyle = 'dashed')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
        plt.legend()
        plt.show()

    for cat_feat in categorical_feature_names:

        # filtering out categories with low counts
        counts = dataset[cat_feat].value_counts()
        threshold = len(dataset) * categorical_pct_threshold
        valid_categories = counts[counts >= threshold].index

        # aggregate average performance vs. expectation
        result = (
            dataset[dataset[cat_feat].isin(valid_categories)]
            .groupby(cat_feat)
            .agg(
                avg_perf_vs_exp = ('perf_vs_exp', 'mean'),
                count = (cat_feat, 'count')
            )
            .sort_values('count', ascending = True)
            .reset_index()
        )
        # getting info to set graph y axis bounds
        result['diff'] = np.abs(1 - result['avg_perf_vs_exp'])
        max_diff = np.max(result['diff'])

        # creating graph
        ax = sns.lineplot(result, x = cat_feat, y = 'avg_perf_vs_exp', label = 'Actual', **kwargs)
        plt.xlabel(cat_feat)
        plt.xticks(rotation = 90)
        plt.ylabel('Performance vs. Expectation')
        plt.ylim(0.9 - max_diff, 1.1 + max_diff)
        plt.title(f'Performance vs. Expectation: {cat_feat}')
        plt.axhline(1, label = 'Perfect', color = 'gray', linestyle = 'dashed')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
        plt.legend()
        plt.tight_layout()
        plt.show()