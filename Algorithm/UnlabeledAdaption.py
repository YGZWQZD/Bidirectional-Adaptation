from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from ..Config.Default_Config import config
from LAMDA_SSL.utils import Bn_Controller
import torch
from LAMDA_SSL.utils import to_device

class UnlabeledAdaption(DeepModelMixin,InductiveEstimator,ClassifierMixin):
    def __init__(self,lambda_t=1.0,
                 mu=1,
                 ema_decay=None,
                 weight_decay=5e-4,
                 epoch=1,
                 num_it_epoch=1000,
                 num_it_total=1000,
                 eval_epoch=None,
                 eval_it=None,
                 device='cpu',
                 num_classes=31,
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
        DeepModelMixin.__init__(self,train_dataset=train_dataset,
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
        self.lambda_t=lambda_t
        self.num_classes=num_classes
        self.weight_decay=weight_decay
        self._estimator_type = ClassifierMixin._estimator_type
        self.bn_controller=Bn_Controller()

    def init_transform(self):
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)

    def fit(self,X=None,y=None,unlabeled_X=None,unlabeled_y=None,valid_X=None,valid_y=None):
        self.unlabeled_X=unlabeled_X
        self.unlabeled_y=unlabeled_y
        self.init_train_dataset(X,y,unlabeled_X)
        self.init_train_dataloader()
        if self.network is not None:
            self.init_model()
            self.init_ema()
            self.init_optimizer()
            self.init_scheduler()
        self.init_augmentation()
        self.init_transform()
        self.start_fit()
        self.fit_epoch_loop(valid_X,valid_y)
        self.end_fit()
        return self

    def fit_batch_loop(self,valid_X=None,valid_y=None):
        for (lb_idx, lb_X, lb_y), (ulb_idx, ulb_X, ulb_y) in zip(self._labeled_dataloader, self._unlabeled_dataloader):
            if self.it_epoch >= self.num_it_epoch or self.it_total >= self.num_it_total:
                break
            self.start_fit_batch()
            lb_idx = to_device(lb_idx,self.device)
            lb_X = to_device(lb_X,self.device)
            lb_y = to_device(lb_y,self.device)
            ulb_idx = to_device(ulb_idx,self.device)
            ulb_X  = to_device(ulb_X,self.device)
            train_result = self.train(lb_X=lb_X, lb_y=lb_y, ulb_X=ulb_X, lb_idx=lb_idx, ulb_idx=ulb_idx)
            self.end_fit_batch(train_result)
            self.it_total += 1
            self.it_epoch += 1
            if valid_X is not None and self.eval_it is not None and self.it_total % self.eval_it == 0:
                self.evaluate(X=valid_X, y=valid_y,valid=True)

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X = lb_X[0] if isinstance(lb_X, (tuple, list)) else lb_X
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        ulb_X = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X
        clf_loss, transfer_loss = self._network(lb_X, ulb_X, lb_y)
        return clf_loss,transfer_loss

    def get_loss(self,train_result,*args,**kwargs):
        clf_loss, transfer_loss=train_result
        loss = clf_loss + self.lambda_t * transfer_loss
        return loss

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        outputs=self._network.predict(X)
        return outputs

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)
