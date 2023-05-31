from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from LAMDA_SSL.utils import to_device
from ..Config.Default_Config import config
from LAMDA_SSL.utils import Bn_Controller
import copy
import math
from torch.utils.data.dataset import Dataset
from LAMDA_SSL.Dataset.TrainDataset import TrainDataset
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

class LabeledAdaption(DeepModelMixin, InductiveEstimator, ClassifierMixin):
    def __init__(self,
                 lambda_s=1.0,
                 T=0.5,
                 weight=True,
                 threshold=0.75,
                 warmup=None,
                 mu=1,
                 ema_decay=None,
                 weight_decay=5e-4,
                 epoch=1,
                 num_it_epoch=1000,
                 num_it_total=1000,
                 num_classes=31,
                 eval_epoch=None,
                 eval_it=200,
                 alpha=0.5,
                 soft=False,
                 device='cuda:0',
                 train_dataset=config.train_dataset,
                 labeled_dataset=config.labeled_dataset,
                 unlabeled_dataset=config.unlabeled_dataset,
                 valid_dataset=config.valid_dataset,
                 test_dataset=config.test_dataset,
                 train_dataloader=config.train_dataloader,
                 labeled_dataloader=config.labeled_dataloader,
                 unlabeled_dataloader=config.unlabeled_dataloader,
                 valid_dataloader=config.valid_dataloader,
                 test_dataloader=config.test_dataloader,
                 train_sampler=config.train_sampler,
                 train_batch_sampler=config.train_batch_sampler,
                 valid_sampler=config.valid_sampler,
                 valid_batch_sampler=config.valid_batch_sampler,
                 test_sampler=config.test_sampler,
                 test_batch_sampler=config.test_batch_sampler,
                 labeled_sampler=config.labeled_sampler,
                 unlabeled_sampler=config.unlabeled_sampler,
                 labeled_batch_sampler=config.labeled_batch_sampler,
                 unlabeled_batch_sampler=config.unlabeled_batch_sampler,
                 augmentation=config.augmentation,
                 network=config.network,
                 optimizer=config.optimizer,
                 scheduler=config.scheduler,
                 evaluation=config.evaluation,
                 parallel=config.parallel,
                 file=config.file,
                 verbose=config.verbose
                 ):
        DeepModelMixin.__init__(self, train_dataset=train_dataset,
                                valid_dataset=valid_dataset,
                                test_dataset=test_dataset,
                                train_dataloader=train_dataloader,
                                valid_dataloader=valid_dataloader,
                                test_dataloader=test_dataloader,
                                augmentation=augmentation,
                                network=network,
                                train_sampler=train_sampler,
                                train_batch_sampler=train_batch_sampler,
                                valid_sampler=valid_sampler,
                                valid_batch_sampler=valid_batch_sampler,
                                test_sampler=test_sampler,
                                test_batch_sampler=test_batch_sampler,
                                labeled_dataset=labeled_dataset,
                                unlabeled_dataset=unlabeled_dataset,
                                labeled_dataloader=labeled_dataloader,
                                unlabeled_dataloader=unlabeled_dataloader,
                                labeled_sampler=labeled_sampler,
                                unlabeled_sampler=unlabeled_sampler,
                                labeled_batch_sampler=labeled_batch_sampler,
                                unlabeled_batch_sampler=unlabeled_batch_sampler,
                                epoch=epoch,
                                num_it_epoch=num_it_epoch,
                                num_it_total=num_it_total,
                                eval_epoch=eval_epoch,
                                eval_it=eval_it,
                                mu=mu,
                                weight_decay=weight_decay,
                                ema_decay=ema_decay,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                device=device,
                                evaluation=evaluation,
                                parallel=parallel,
                                file=file,
                                verbose=verbose
                                )
        self.lambda_s = lambda_s
        self.T = T
        self.threshold = threshold
        self.num_classes = num_classes
        self.weight = weight
        self.warmup = warmup
        self.soft=soft
        self.bn_controller = Bn_Controller()
        self.alpha = alpha
        self._estimator_type = ClassifierMixin._estimator_type

    def start_fit(self, *args, **kwargs):
        self.init_epoch()
        self._network.zero_grad()
        self._network.train()
        self.labeled_mean = 0
        self.labeled_num = 0

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation, dim=1, x=0, y=0)
        self._train_dataset.add_transform(self.weak_augmentation, dim=1, x=0, y=0)
        for _ in range(self.num_classes):
            if self.class_dataset_labeled[_] is not None:
                self.class_dataset_labeled[_].add_transform(self.weak_augmentation, dim=1, x=0, y=0)
            if self.class_dataset_unlabeled[_] is not None:
                self.class_dataset_unlabeled[_].add_transform(self.weak_augmentation, dim=1, x=0, y=0)

    def fit(self, X=None, y=None, unlabeled_X=None, valid_X=None, valid_y=None, unlabeled_y=None,est=None):
        self.unlabeled_X = unlabeled_X
        self.unlabeled_y = unlabeled_y
        self.est=torch.Tensor(est).to(self.device)
        self.init_train_dataset(X, y, unlabeled_X, unlabeled_y)
        self.init_train_dataloader()
        if self.network is not None:
            self.init_model()
            self.init_ema()
            self.init_optimizer()
            self.init_scheduler()
        self.init_augmentation()
        self.init_transform()
        self.start_fit()
        self.fit_epoch_loop(valid_X, valid_y)
        self.end_fit()
        return self

    def init_train_dataset(self, X=None, y=None, unlabeled_X=None, unlabeled_y=None, *args, **kwargs):
        self.class_dataset_labeled = []
        self.class_dataset_unlabeled = []
        self.class_labeled_X = []
        self.class_unlabeled_X = []
        self.per_class_num=np.zeros(self.num_classes)
        self.avg_per_class_confidence=np.zeros(self.num_classes)
        self.avg_confidence=0
        self.num=0
        for _ in range(self.num_classes):
            self.class_labeled_X.append([])
            self.class_unlabeled_X.append([])
        for _ in range(len(X)):
            self.class_labeled_X[y[_]].append(X[_])
        for _ in range(len(unlabeled_X)):
            self.class_unlabeled_X[unlabeled_y[_]].append(unlabeled_X[_])
        self.num_unlabeled_samples = []
        self.num_labeled_samples = []
        for _ in range(self.num_classes):
            self.class_dataset_labeled.append(
                copy.deepcopy(self.labeled_dataset).init_dataset(self.class_labeled_X[_], y[y == _]))
            self.num_labeled_samples.append(len(self.class_labeled_X[_]))
            if unlabeled_y[unlabeled_y == _].shape[0] != 0:
                self.class_dataset_unlabeled.append(
                    copy.deepcopy(self.labeled_dataset).init_dataset(self.class_unlabeled_X[_],
                                                                     unlabeled_y[unlabeled_y == _]))
                self.num_unlabeled_samples.append(len(self.class_unlabeled_X[_]))
            else:
                self.class_dataset_unlabeled.append(None)
                self.num_unlabeled_samples.append(0)
        self.num_labeled_samples = np.array(self.num_labeled_samples)
        self.num_unlabeled_samples = np.array(self.num_unlabeled_samples)
        self.num_unlabeled_samples[self.num_unlabeled_samples == 0] = 1
        self.weight_class = self.num_labeled_samples *len(unlabeled_X)/len(X)/ self.num_unlabeled_samples
        self._train_dataset = copy.deepcopy(self.train_dataset)
        if isinstance(X, TrainDataset):
            self._train_dataset = X
        elif isinstance(X, Dataset) and y is None:
            self._train_dataset.init_dataset(labeled_dataset=X, unlabeled_dataset=unlabeled_X, unlabeled_y=unlabeled_y)
        else:
            self._train_dataset.init_dataset(labeled_X=X, labeled_y=y, unlabeled_X=unlabeled_X, unlabeled_y=unlabeled_y)

    def fit_batch_loop(self, valid_X=None, valid_y=None):
        for (lb_idx, lb_X, lb_y), (ulb_idx, ulb_X, ulb_y) in zip(self._labeled_dataloader, self._unlabeled_dataloader):
            if self.it_epoch >= self.num_it_epoch or self.it_total >= self.num_it_total:
                break
            self.start_fit_batch()
            lb_idx = to_device(lb_idx, self.device)
            lb_X = to_device(lb_X, self.device)
            lb_y = to_device(lb_y, self.device)
            ulb_idx = to_device(ulb_idx, self.device)
            ulb_X = to_device(ulb_X, self.device)
            ulb_y = to_device(ulb_y, self.device)
            train_result = self.train(lb_X=lb_X, lb_y=lb_y, ulb_X=ulb_X, ulb_y=ulb_y, lb_idx=lb_idx, ulb_idx=ulb_idx)
            self.end_fit_batch(train_result)
            self.it_total += 1
            self.it_epoch += 1
            if valid_X is not None and self.eval_it is not None and self.it_total % self.eval_it == 0:
                self.evaluate(X=valid_X, y=valid_y, valid=True)
                self.valid_performance.update(
                    {"epoch_" + str(self._epoch) + "_it_" + str(self.it_epoch): self.performance})

    def train(self, lb_X, lb_y, ulb_X, ulb_y=None, lb_idx=None, ulb_idx=None, *args, **kwargs):
        lb_X = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        ulb_X = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X
        ulb_y = ulb_y[0] if isinstance(ulb_y, (tuple, list)) else ulb_y
        est=self.est[ulb_idx]

        target_features, target_logits = self._network(lb_X)
        self.bn_controller.freeze_bn(self._network)
        source_features, source_logits = self._network(ulb_X)
        self.bn_controller.unfreeze_bn(self._network)
        return target_logits, lb_y, source_logits, source_features, target_features, ulb_y,est

    def get_loss(self, train_result, *args, **kwargs):
        target_logits, lb_y, source_logits, source_features, target_features, ulb_y ,est= train_result
        target_clf_loss = CrossEntropyLoss()(target_logits, lb_y)

        if self.T is not None:
            est = est / self.T
        if self.weight:
            if self.T is not None:
                pseudo_label = torch.softmax((source_logits/self.T).detach() , dim=-1)
            else:
                pseudo_label = torch.softmax((source_logits).detach(), dim=-1)
            weight=np.zeros(ulb_y.shape[0])
            _ulb_y=ulb_y.detach().cpu().numpy()
            pseudo_label=pseudo_label.detach().cpu().numpy()
            for _ in range(_ulb_y.shape[0]):
                weight[_] = pseudo_label[_][_ulb_y[_]]
                self.avg_per_class_confidence[_ulb_y[_]] = (self.avg_per_class_confidence[_ulb_y[_]] * self.per_class_num[_ulb_y[_]]+weight[_])/(self.per_class_num[_ulb_y[_]]+1)
                self.per_class_num[_ulb_y[_]] += 1
                self.avg_confidence=(self.avg_confidence*self.num+weight[_])/(self.num+1)
                self.num+=1
            for _ in range(_ulb_y.shape[0]):
                weight[_] = weight[_]*self.avg_confidence/self.avg_per_class_confidence[_ulb_y[_]]
            weight=torch.Tensor(weight).to(self.device)
            norm = torch.mean(weight)
            weight=weight/norm
            if self.soft:
                source_loss = torch.mean(CrossEntropyLoss(reduction='none')(source_logits, (est).softmax(1).detach()) * weight.detach()*torch.Tensor(self.weight_class[ulb_y.cpu()]).to(self.device))
            else:
                source_loss = torch.mean(CrossEntropyLoss(reduction='none')(source_logits, ulb_y.detach()) * weight.detach() * torch.Tensor(self.weight_class[ulb_y.cpu()]).to(self.device))
        else:
            if self.soft:
                source_loss = (CrossEntropyLoss(reduction='none')(source_logits, (est).softmax(1).detach())).mean()
            else:
                source_loss = (CrossEntropyLoss(reduction='none')(source_logits, ulb_y.detach())).mean()
        if self.warmup is not None:
            coef = 1. * math.exp(-5 * (1 - min(self.it_total / (self.warmup * self.num_it_total), 1)) ** 2)
        else:
            coef= 1
        loss = target_clf_loss + self.lambda_s*coef * source_loss
        return loss

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        _, outputs = self._network(X)
        return outputs

    def predict(self, X=None, valid=None):
        return DeepModelMixin.predict(self, X=X, valid=valid)
