"""Classical ML models + one ANN (MLP). No deep learning / CNNs anywhere."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Prep:
    """StandardScaler -> PCA -> one global scale (keeps relative component variances)."""

    def __init__(self, n_components=512):
        self.n_components = n_components

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        n = min(self.n_components, Z.shape[0] - 1, Z.shape[1])
        self.pca = PCA(n_components=n, svd_solver="randomized", random_state=0).fit(Z)
        self.scale = float(self.pca.transform(Z).std()) or 1.0
        return self

    def transform(self, X):
        return self.pca.transform(self.sc.transform(X)) / self.scale


class AttributeStack(BaseEstimator, ClassifierMixin):
    """Two-stage model inspired by the report's 312-attribute idea.

    Stage 1: ridge regressors predict the 312 bird attributes (colours, bill
             shape, wing shape...) from image features. Out-of-fold predictions
             are used for training stage 2 to avoid leakage.
    Stage 2: logistic regression maps predicted attributes -> species.
    At test time NO human-annotated attributes are needed.
    """

    def __init__(self, class_attrs=None, alpha=10.0, C=10.0):
        self.class_attrs, self.alpha, self.C = class_attrs, alpha, C

    def fit(self, X, y):
        T = self.class_attrs[y]
        oof = np.zeros_like(T, dtype=np.float64)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            oof[b] = Ridge(alpha=self.alpha).fit(X[a], T[a]).predict(X[b])
        self.ridge_ = Ridge(alpha=self.alpha).fit(X, T)
        self.lr_ = LogisticRegression(C=self.C, max_iter=500).fit(np.clip(oof, 0, 1), y)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(np.clip(self.ridge_.predict(X), 0, 1))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def zoo(class_attrs, skip=()):
    m = {
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(5, weights="distance", metric="cosine"),
        "RandomForest": RandomForestClassifier(300, n_jobs=-1, random_state=0),
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "LogReg": LogisticRegression(C=0.5, max_iter=500),
        "SVM-RBF": SVC(C=10, kernel="rbf", gamma="scale", probability=True, random_state=0),
        "ANN-MLP": MLPClassifier((1024, 512), alpha=1e-2, early_stopping=True, n_iter_no_change=15,
                                 max_iter=300, random_state=0),
        "AttrStack": AttributeStack(class_attrs),
    }
    return {k: v for k, v in m.items() if k.lower() not in {s.lower() for s in skip}}


def full_proba(model, Z, n_classes):
    """predict_proba aligned to all n_classes columns."""
    P = np.full((len(Z), n_classes), 1e-9)
    P[:, model.classes_] = model.predict_proba(Z)
    return P / P.sum(1, keepdims=True)


def greedy_ensemble(probs, y, n_classes, rounds=15):
    """Caruana greedy ensemble selection (with replacement) on validation log-loss."""
    labels, chosen, total, best = list(range(n_classes)), [], 0.0, np.inf
    for _ in range(rounds):
        cand = {n: log_loss(y, (total + P) / (len(chosen) + 1), labels=labels) for n, P in probs.items()}
        n = min(cand, key=cand.get)
        if cand[n] >= best - 1e-4 and chosen:
            break
        best, total, chosen = cand[n], total + probs[n], chosen + [n]
    return {n: chosen.count(n) for n in set(chosen)}


def blend(probs, weights):
    return sum(w * probs[n] for n, w in weights.items()) / sum(weights.values())
