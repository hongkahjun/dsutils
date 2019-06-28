import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.inspection import plot_partial_dependence
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support


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


def pca_biplot(score, coeff, y, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c=y)
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


def check_class_imba(y):
    plt.hist(y)
    plt.show()


def rf_feature_ranking(x, y, labels=None, imbalanced=False):
    parameters = {
        'n_estimators': [256, 512],
        'max_depth': [1, 5, 7],
        'max_features': (1, 0.2, 0.4, 0.6, 0.8, None)
    }
    if imbalanced:
        rf = BalancedRandomForestClassifier(criterion='entropy', oob_score=True, replacement=True)
    else:
        rf = RandomForestClassifier(criterion='entropy', oob_score=True)
    rf_cv = GridSearchCV(rf, parameters, cv=5, iid=False, return_train_score=True, refit=True)
    rf_cv.fit(x, y)
    best_rf = rf_cv.best_estimator_
    if labels is None:
        labels = np.arange(x.shape[0])
    order = np.arange(x.shape[0])
    ranked_features = [n for _, n, _ in sorted(zip(best_rf.feature_importances_, labels, order), reverse=True)]
    ordered_indices = [n for _, _, n in sorted(zip(best_rf.feature_importances_, labels, order), reverse=True)]
    return ranked_features, ordered_indices, best_rf


def pdp(est, x, feature, feature_names, no, target):
    if no == -1:
        plot_partial_dependence(est, x, feature, feature_names, target=target)
    else:
        plot_partial_dependence(est, x, feature[:no], feature_names, target=target)
    fig = plt.gcf()
    fig.suptitle('Partial dependence')
    plt.subplots_adjust(top=0.9)
    plt.show()


def pca_lr(x, y, imbalanced=False):
    lr = LogisticRegression(solver='lbfgs', multi_class='auto')
    pca = PCA()
    if imbalanced:
        pca_lr_parameters = {
            'bagged_lr__n_estimators': [512, 1024],
            'pca__n_components': [None, 0.95, 0.9, 0.85, 0.8],
            'bagged_lr__base_estimator__C': [2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1,
                                             2, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7]
        }
        bagged_lr = BalancedBaggingClassifier(base_estimator=lr, oob_score=True, replacement=True)
        pca_lr_pipeline = Pipeline([('scale', StandardScaler()), ('pca', pca), ('bagged_lr', bagged_lr)])
    else:
        pca_lr_parameters = {
            'pca__n_components': [None, 0.95, 0.9, 0.85, 0.8],
            'lr__C': [2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1,
                      2, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7]
        }
        pca_lr_pipeline = Pipeline([('scale', StandardScaler()), ('pca', pca), ('lr', lr)])
    pca_lr_cv = GridSearchCV(pca_lr_pipeline, pca_lr_parameters, cv=5, iid=True, return_train_score=True)
    pca_lr_cv.fit(x, y)
    cv_results_df = pd.DataFrame(pca_lr_cv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    return pca_lr_cv, cv_results_df


def pca_rf(x, y, imbalanced=False):
    parameters = {
        'pca__n_components': [None, 0.95, 0.9, 0.85, 0.8],
        'rf__n_estimators': [512, 1024],
        'rf__max_depth': [1, 5, 7],
        'rf__max_features': (1, 0.2, 0.4, 0.6, 0.8, None)
    }
    pca = PCA()
    if imbalanced:
        rf = BalancedRandomForestClassifier(criterion='entropy', oob_score=True, replacement=True)
    else:
        rf = RandomForestClassifier(criterion='entropy', oob_score=True)
    pca_rf_pipeline = Pipeline([('scale', StandardScaler()), ('pca', pca), ('rf', rf)])
    rf_cv = GridSearchCV(pca_rf_pipeline, parameters, cv=5, iid=False, return_train_score=True, refit=True)
    rf_cv.fit(x, y)
    cv_results_df = pd.DataFrame(rf_cv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    return rf_cv, cv_results_df


def pca_rbf_svm(x, y, imbalanced=False):
    svm = SVC()
    pca = PCA()
    if imbalanced:
        pca_svm_parameters = {
            'bagged_svm__n_estimators': [512, 1024],
            'pca__n_components': [None, 0.95, 0.9, 0.85, 0.8],
            'bagged_svm__base_estimator__C': [2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1, 2,
                                              2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7],
            'bagged_svm__base_estimator__gamma': [2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1, 2,
                                                  2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7]
        }
        bagged_svm = BalancedBaggingClassifier(base_estimator=svm, oob_score=True, replacement=True)
        pca_svm_pipeline = Pipeline([('scale', StandardScaler()), ('pca', pca), ('bagged_svm', bagged_svm)])
    else:
        pca_svm_parameters = {
            'pca__n_components': [None, 0.95, 0.9, 0.85, 0.8],
            'svm__C': [2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1, 2, 2 ** 2, 2 ** 3, 2 ** 4,
                       2 ** 5, 2 ** 6, 2 ** 7],
            'svm__gamma': [2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1, 2, 2 ** 2, 2 ** 3, 2 ** 4,
                           2 ** 5, 2 ** 6, 2 ** 7]
        }
        pca_svm_pipeline = Pipeline([('scale', StandardScaler()), ('pca', pca), ('svm', svm)])
    pca_svm_cv = GridSearchCV(pca_svm_pipeline, pca_svm_parameters, cv=5, iid=True, return_train_score=True)
    pca_svm_cv.fit(x, y)
    cv_results_df = pd.DataFrame(pca_svm_cv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    return pca_svm_cv, cv_results_df


def cal_test_results(_x_train, _x_test, _y_train, _y_test, imbalanced=False):
    pca_lr_grcv, pca_lr_results = pca_lr(_x_train, _y_train, imbalanced=imbalanced)
    pca_rbf_svm_grcv, pca_rbf_svm_results = pca_rbf_svm(_x_train, _y_train, imbalanced=imbalanced)
    pca_rf_grcv, pca_rf_results = pca_rf(_x_train, _y_train, imbalanced=imbalanced)
    result_index = ['LR', 'RBF-SVM', 'RF']
    result_column = ['Precision', 'Recall', 'F1']
    test_results = np.zeros((3, 3))
    test_results[0, :] = precision_recall_fscore_support(_y_test, pca_lr_grcv.predict(_x_test), average='weighted')[:3]
    test_results[1, :] = precision_recall_fscore_support(_y_test, pca_rbf_svm_grcv.predict(_x_test),
                                                         average='weighted')[:3]
    test_results[2, :] = precision_recall_fscore_support(_y_test, pca_rf_grcv.predict(_x_test), average='weighted')[:3]
    results_df = pd.DataFrame(test_results, result_index, result_column)
    return [pca_lr_grcv, pca_rbf_svm_grcv, pca_rf_grcv], [pca_lr_results, pca_rbf_svm_results,
                                                          pca_rf_results], results_df


np.random.seed(42)
csv_file = 'iris_df.csv'
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
pca_biplot(x_all_pca[:, 0:2], np.transpose(pca_model.components_[0:2, :]), y_all, labels=features)

# check class imbalance
check_class_imba(y_all)
IMBALANCED = False
# feature ranking and pdp
feature_ranked, indices_ranked, ranked_model = rf_feature_ranking(x_all, y_all, features, imbalanced=IMBALANCED)
pdp(ranked_model, x_all, indices_ranked, feature_ranked, 3, target=1)

# classification models
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2)
models, cv_result_list, test_results_df = cal_test_results(x_train, x_test, y_train, y_test, IMBALANCED)
print(test_results_df)
