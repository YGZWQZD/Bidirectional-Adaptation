from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
import numpy as np


class BDA(InductiveEstimator, ClassifierMixin):
    def __init__(self, UnlabeledDomainAdaption, LabeledDomainAdaption, pred_path=None, score_path=None,
                 load_from_path=False, T=0.2, threshold=0.7):
        self.LabeledDomainAdaption = LabeledDomainAdaption
        self.UnlabeledDomainAdaption = UnlabeledDomainAdaption
        self.pred_path = pred_path
        self.score_path = score_path
        self.load_from_path = load_from_path
        self.T = T
        self.threshold = threshold

    def fit(self, X, y, unlabeled_X, valid_X=None, valid_y=None,unlabeled_y=None):
        if self.load_from_path:
            pred = np.load(self.pred_path, mmap_mode=None, allow_pickle=False, fix_imports=True,
                           encoding='ASCII')
            est = np.load(self.score_path, mmap_mode=None, allow_pickle=False, fix_imports=True,
                          encoding='ASCII')
        else:
            self.UnlabeledDomainAdaption.fit(X, y, unlabeled_X)
            pred = self.UnlabeledDomainAdaption.predict(unlabeled_X)
            est = self.UnlabeledDomainAdaption.y_est.detach().to('cpu').numpy()
            if self.pred_path is not None:
                np.save(self.pred_path, pred, allow_pickle=True, fix_imports=True)
            if self.score_path is not None:
                np.save(self.score_path, est, allow_pickle=True, fix_imports=True)
        unlabeled_y = pred
        self.LabeledDomainAdaption.fit(X=X, y=y, unlabeled_X=unlabeled_X, unlabeled_y=unlabeled_y, valid_X=valid_X,
                                       valid_y=valid_y,est=est)
        return self

    def predict(self, X):
        y = self.LabeledDomainAdaption.predict(X)
        return y
