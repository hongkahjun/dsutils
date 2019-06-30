import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pygam import GAM
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import plot_partial_dependence


def import_csv(file):
    df = pd.read_csv(file)
    print(df.head())
    print(df.shape)
    print("")
    print("Percentage of nans:")
    print(df.isna().mean().round(4) * 100)
    return df


def import_csv_txt(file):
    df = np.genfromtxt(file, delimiter=',', dtype='float64')
    df = pd.DataFrame(df)
    print(df.head())
    print(df.shape)
    print("")
    print("Percentage of nans:")
    print(df.isna().mean().round(4) * 100)
    return df


def pca_elbow_plot(x):
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA()
    pca.fit(x_scaled)
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Explained Variance')
    plt.show()
    return pca


def pca_biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show()


def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')


def diagnostic_plots(x, y, model_fit=None):
    if not model_fit:
        model_fit = sm.OLS(y, sm.add_constant(x)).fit()

    dataframe = pd.concat([x, y], axis=1)
    model_fitted_y = model_fit.fittedvalues
    model_residuals = model_fit.resid
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    model_abs_resid = np.abs(model_residuals)
    model_leverage = model_fit.get_influence().hat_matrix_diag
    model_cooks = model_fit.get_influence().cooks_distance[0]

    plot_lm_1 = plt.figure()
    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, dataframe.columns[-1], data=dataframe, lowess=True,
                                      scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals')
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    for i in abs_resid_top_3.index:
        plot_lm_1.axes[0].annotate(i, xy=(model_fitted_y[i], model_residuals[i]))

    QQ = ProbPlot(model_norm_residuals)
    plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
    plot_lm_2.axes[0].set_title('Normal Q-Q')
    plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_2.axes[0].set_ylabel('Standardized Residuals')
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for r, i in enumerate(abs_norm_resid_top_3):
        plot_lm_2.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], model_norm_residuals[i]))

    plot_lm_3 = plt.figure()
    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, scatter=False, ci=False, lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_3.axes[0].set_title('Scale-Location')
    plot_lm_3.axes[0].set_xlabel('Fitted values')
    plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$')

    for i in abs_norm_resid_top_3:
        plot_lm_3.axes[0].annotate(i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]))

    plot_lm_4 = plt.figure()
    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals, scatter=False, ci=False, lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_4.axes[0].set_xlim(0, max(model_leverage) + 0.01)
    plot_lm_4.axes[0].set_ylim(-3, 5)
    plot_lm_4.axes[0].set_title('Residuals vs Leverage')
    plot_lm_4.axes[0].set_xlabel('Leverage')
    plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
    for i in leverage_top_3:
        plot_lm_4.axes[0].annotate(i, xy=(model_leverage[i], model_norm_residuals[i]))

    p = len(model_fit.params)  # number of model parameters
    graph(lambda a: np.sqrt((0.5 * p * (1 - a)) / a), np.linspace(0.001, max(model_leverage), 50), 'Cook\'s distance')
    graph(lambda a: np.sqrt((1 * p * (1 - a)) / a), np.linspace(0.001, max(model_leverage), 50))
    plot_lm_4.legend(loc='upper right')
    plt.show()


def rf_feature_ranking(x, y, labels=None):
    parameters = {
        'n_estimators': [256, 512],
        'max_depth': [1, 5, 7],
        'max_features': (1, 0.2, 0.4, 0.6, 0.8, None)
    }
    rf = RandomForestRegressor(criterion='mse', oob_score=True)
    rf_cv = GridSearchCV(rf, parameters, cv=5, iid=False, return_train_score=True, refit=True,
                         scoring='neg_mean_squared_error')
    rf_cv.fit(x, y)
    best_rf = rf_cv.best_estimator_
    if labels is None:
        labels = np.arange(x.shape[0])
    order = np.arange(x.shape[0])
    ranked_features = [n for _, n, _ in sorted(zip(best_rf.feature_importances_, labels, order), reverse=True)]
    ordered_indices = [n for _, _, n in sorted(zip(best_rf.feature_importances_, labels, order), reverse=True)]
    return ranked_features, ordered_indices, best_rf


def pdp(est, x, feature, feature_names, no):
    fig = plt.figure(figsize=(24, 18))
    if no == -1:
        plot_partial_dependence(est, x, feature, feature_names, fig=fig)
    else:
        plot_partial_dependence(est, x, feature[:no], feature_names, fig=fig)
    fig = plt.gcf()
    fig.suptitle('Partial dependence', fontsize=30)
    plt.subplots_adjust(top=0.95)
    plt.show()


def GamCV(x, y):
    lams = np.random.rand(10, x.shape[1])
    lams = np.exp(lams)
    linear_gam = GAM(n_splines=10, max_iter=1000)
    parameters = {
        'lam': [x for x in lams]
    }
    gam_cv = GridSearchCV(linear_gam, parameters, cv=5, iid=False, return_train_score=True, refit=True,
                          scoring='neg_mean_squared_error')
    gam_cv.fit(x, y)
    cv_results_df = pd.DataFrame(gam_cv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    return gam_cv, cv_results_df


def gam(x, y):
    lams = np.random.rand(100, x.shape[1])
    lams = np.exp(lams)
    linear_gam = GAM(n_splines=10, max_iter=1000)
    cv_results = linear_gam.gridsearch(x, y, return_scores=True, lam=lams, progress=False)
    cv_results_df = pd.DataFrame(cv_results, index=['score']).T.sort_values(by='score', ascending=False)
    return linear_gam, cv_results_df


def en(x, y):
    parameters = {
        'alpha': [2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1, 2, 2 ** 2, 2 ** 3, 2 ** 4,
                  2 ** 5, 2 ** 6, 2 ** 7],
        'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1]
    }
    en_pipeline = Pipeline([('scale', StandardScaler()), ('en', ElasticNet())])
    en_cv = GridSearchCV(en_pipeline, parameters, cv=5, iid=False, return_train_score=True, refit=True,
                         scoring='neg_mean_squared_error')
    en_cv.fit(x, y)
    cv_results_df = pd.DataFrame(en_cv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    return en_cv, cv_results_df


def rf(x, y):
    parameters = {
        'n_estimators': [256, 512],
        'max_depth': [1, 5, 7],
        'max_features': (1, 0.2, 0.4, 0.6, 0.8, None)
    }
    _rf = RandomForestRegressor(criterion='mse', oob_score=True)
    rf_cv = GridSearchCV(_rf, parameters, cv=5, iid=False, return_train_score=True, refit=True,
                         scoring='neg_mean_squared_error')
    rf_cv.fit(x, y)
    cv_results_df = pd.DataFrame(rf_cv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    return rf_cv, cv_results_df


def gam_pdp(est, feature, feature_names, no):
    if no == -1:
        plt.figure()
        fig, axs = plt.subplots(1, len(feature_names), figsize=(40, 8))
        for i, ax in enumerate(axs):
            XX = est.generate_X_grid(term=i)
            ax.plot(XX[:, i], est.partial_dependence(term=i, X=XX))
            ax.plot(XX[:, i], est.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
            if i == 0:
                ax.set_ylim(-30, 30)
            ax.set_title(feature_names[feature.index(i)])
    else:
        plt.figure()
        fig, axs = plt.subplots(1, no, figsize=(40, 8))
        for i, ax in enumerate(axs):
            f_no = feature[i]
            XX = est.generate_X_grid(term=f_no)
            ax.plot(XX[:, f_no], est.partial_dependence(term=f_no, X=XX))
            ax.plot(XX[:, f_no], est.partial_dependence(term=f_no, X=XX, width=.95)[1], c='r', ls='--')
            if f_no == 0:
                ax.set_ylim(-30, 30)
            ax.set_title(feature_names[i])
    plt.show()


def cal_test_results(_x_train, _x_test, _y_train, _y_test):
    gam_cv, gam_results_df = gam(_x_train, _y_train)
    en_cv, en_results_df = en(_x_train, _y_train)
    rf_cv, rf_results_df = rf(_x_train, _y_train)
    result_index = ['GAM', 'EN', 'RF']
    result_column = ['MSE', 'R2']
    test_results = np.zeros((3, 2))
    test_results[0, 0] = mean_squared_error(_y_test, gam_cv.predict(_x_test))
    test_results[0, 1] = r2_score(_y_test, gam_cv.predict(_x_test))
    test_results[1, 0] = mean_squared_error(_y_test, en_cv.predict(_x_test))
    test_results[1, 1] = r2_score(_y_test, en_cv.predict(_x_test))
    test_results[2, 0] = mean_squared_error(_y_test, rf_cv.predict(_x_test))
    test_results[2, 1] = r2_score(_y_test, rf_cv.predict(_x_test))
    results_df = pd.DataFrame(test_results, result_index, result_column)
    return [gam_cv, en_cv, rf_cv], [gam_results_df, en_results_df, rf_results_df], results_df


np.random.seed(42)
csv_file = 'winequality-red.csv'
dataset = import_csv(csv_file)
# dataset = import_csv_txt(csv_file)

# if dataset has invalid values e.g. negative values for positive-only features
dataset.loc[[1, 2, 3, 4]] = dataset.loc[[1, 2, 3, 4]].replace(0, np.nan)
dataset[dataset < 0] = np.nan

# drop nan
dataset = dataset.dropna(subset=['columns'])

# impute instead
dataset = dataset.interpolate()

# assume targets are last column of DataFame
dataset_values = dataset.values
x_all = dataset_values[:, :-1]
y_all = dataset_values[:, -1]
features = list(dataset)[:-1]

# pca elbow plot
pca_model = pca_elbow_plot(x_all)
x_scaled = StandardScaler().fit_transform(x_all)
x_all_pca = pca_model.transform(x_scaled)

# PCA biplot
pca_biplot(x_all_pca[:, 0:2], np.transpose(pca_model.components_[0:2, :]), labels=features)

# diagnostic plots
diagnostic_plots(dataset.iloc[:, :-1], dataset.iloc[:, -1])

# feature ranking
feature_ranked, indices_ranked, ranked_model = rf_feature_ranking(x_all, y_all, features)
pdp(ranked_model, x_all, indices_ranked, feature_ranked, 7)
# gam_rank, cv_results = gam(x_all, y_all)
# gam_pdp(gam_rank, indices_ranked, feature_ranked, 3)

# classification models
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2)
models, cv_result_list, test_results_df = cal_test_results(x_train, x_test, y_train, y_test)
print(test_results_df)
