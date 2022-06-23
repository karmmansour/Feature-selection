# Feature-selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import (load_digits, load_breast_cancer,
                              load_diabetes, load_boston)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import (SelectKBest, MutualInfoSelector,
                                       f_classif, f_regression)
from sklearn.svm import LinearSVC


def compare_methods(clf, X, y, discrete_features, discrete_target,
                    k_all=None, cv=5):
    if k_all is None:
        k_all = np.arange(1, X.shape[1] + 1)

    if discrete_target:
        f_test = SelectKBest(score_func=f_classif)
    else:
        f_test = SelectKBest(score_func=f_regression)

    max_rel = MutualInfoSelector(use_redundancy=False,
                                 n_features_to_select=np.max(k_all),
                                 discrete_features=discrete_features,
                                 discrete_target=discrete_target)

    mrmr = MutualInfoSelector(n_features_to_select=np.max(k_all),
                              discrete_features=discrete_features,
                              discrete_target=discrete_target)

    f_test.fit(X, y)
    max_rel.fit(X, y)
    mrmr.fit(X, y)

    f_test_scores = []
    max_rel_scores = []
    mrmr_scores = []

    for k in k_all:
        f_test.set_params(k=k)
        max_rel.set_params(n_features_to_select=k)
        mrmr.set_params(n_features_to_select=k)

        X_f_test = X[:, f_test.get_support()]
        X_max_rel = X[:, max_rel.get_support()]
        X_mrmr = X[:, mrmr.get_support()]

        f_test_scores.append(np.mean(cross_val_score(clf, X_f_test, y, cv=cv)))
        max_rel_scores.append(
            np.mean(cross_val_score(clf, X_max_rel, y, cv=cv)))
        mrmr_scores.append(np.mean(cross_val_score(clf, X_mrmr, y, cv=cv)))

    scores = np.vstack((f_test_scores, max_rel_scores, mrmr_scores))

    return k_all, scores


digits = load_digits()
X = digits.data
y = digits.target
k_digits, scores_digits = compare_methods(LinearSVC(), X, y, True, True,
                                          k_all=np.arange(1, 16))

cancer = load_breast_cancer()
X = minmax_scale(cancer.data)
y = cancer.target
k_cancer, scores_cancer = compare_methods(LinearSVC(), X, y, False, True,
                                          k_all=np.arange(1, 16))

diabetis = load_diabetes()
X = diabetis.data
y = diabetis.target
k_diabetis, scores_diabetis = compare_methods(RidgeCV(normalize=True), X, y,
                                              [1], False)

boston = load_boston()
X = boston.data
y = boston.target
k_boston, scores_boston = compare_methods(RidgeCV(normalize=True),
                                          X, y, [3], False)


plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.plot(k_digits, scores_digits[0], 'x-', label='F-test')
plt.plot(k_digits, scores_digits[1], 'x-', label='MaxRel')
plt.plot(k_digits, scores_digits[2], 'x-', label='mRMR')
plt.title("LinearSVC on digits dataset")
plt.xlabel('Number of kept features')
plt.ylabel('5-fold CV average score')
plt.legend(loc='lower right')

plt.subplot(222)
plt.plot(k_cancer, scores_cancer[0], 'x-', label='F-test')
plt.plot(k_cancer, scores_cancer[1], 'x-', label='MaxRel')
plt.plot(k_cancer, scores_cancer[2], 'x-', label='mRMR')
plt.title("LinearSVC on breast cancer dataset")
plt.xlabel('Number of kept features')
plt.ylabel('5-fold CV average score')
plt.legend(loc='lower right')

plt.subplot(223)
plt.plot(k_diabetis, scores_diabetis[0], 'x-', label='F-test')
plt.plot(k_diabetis, scores_diabetis[1], 'x-', label='MaxRel')
plt.plot(k_diabetis, scores_diabetis[2], 'x-', label='mRMR')
plt.title("RidgeCV on diabetes dataset")
plt.xlabel('Number of kept features')
plt.ylabel('5-fold CV average score')
plt.legend(loc='lower right')

plt.subplot(224)
plt.plot(k_boston, scores_boston[0], 'x-', label='F-test')
plt.plot(k_boston, scores_boston[1], 'x-', label='MaxRel')
plt.plot(k_boston, scores_boston[2], 'x-', label='mRMR')
plt.title("RidgeCV on Boston dataset")
plt.xlabel('Number of kept features')
plt.ylabel('5-fold CV average score')
plt.legend(loc='lower right')

plt.suptitle("Algorithm scores using different feature selection methods",
             fontsize=16)
plt.show()
