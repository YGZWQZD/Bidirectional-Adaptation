from LAMDA_SSL.Augmentation.Vision.RandomHorizontalFlip import RandomHorizontalFlip
from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from LAMDA_SSL.Network.WideResNet import WideResNet
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classifier.Top_k_Accuracy import Top_k_Accurary
from LAMDA_SSL.Evaluation.Classifier.Precision import Precision
from LAMDA_SSL.Evaluation.Classifier.Recall import Recall
from LAMDA_SSL.Evaluation.Classifier.F1 import F1
from LAMDA_SSL.Evaluation.Classifier.AUC import AUC
from LAMDA_SSL.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Transform.ToImage import ToImage

class default_config:
    def __init__(self):
        self.pre_transform = ToImage()
        self.transforms = None
        self.target_transform = None
        self.transform = ToTensor(dtype='float',image=True)
        self.unlabeled_transform = ToTensor(dtype='float',image=True)
        self.test_transform = ToTensor(dtype='float',image=True)
        self.valid_transform = ToTensor(dtype='float',image=True)

        self.train_dataset=None
        self.labeled_dataset=LabeledDataset(pre_transform=self.pre_transform,transforms=self.transforms,
                                       transform=self.transform,target_transform=self.target_transform)

        self.unlabeled_dataset=UnlabeledDataset(pre_transform=self.pre_transform,transform=self.unlabeled_transform)

        self.valid_dataset=UnlabeledDataset(pre_transform=self.pre_transform,transform=self.valid_transform)

        self.test_dataset=UnlabeledDataset(pre_transform=self.pre_transform,transform=self.test_transform)

        # Batch sampler
        self.train_batch_sampler=None
        self.labeled_batch_sampler=None
        self.unlabeled_batch_sampler=None
        self.valid_batch_sampler=None
        self.test_batch_sampler=None

        # sampler
        self.train_sampler=None
        self.labeled_sampler=RandomSampler(replacement=True,num_samples=64*(2**20))
        self.unlabeled_sampler=RandomSampler(replacement=True)
        self.valid_sampler=SequentialSampler()
        self.test_sampler=SequentialSampler()

        #dataloader
        self.train_dataloader=None
        self.labeled_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=True)
        self.unlabeled_dataloader=UnlabeledDataLoader(num_workers=0,drop_last=True)
        self.valid_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)
        self.test_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)

        # network
        self.network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)

        # optimizer
        self.optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)

        # scheduler
        self.scheduler=CosineAnnealingLR(eta_min=0,T_max=1000)

        # augmentation
        self.augmentation=RandomHorizontalFlip()

        # evalutation
        self.evaluation={
            'accuracy':Accuracy(),
            'top_5_accuracy':Top_k_Accurary(k=5),
            'precision':Precision(average='macro'),
            'Recall':Recall(average='macro'),
            'F1':F1(average='macro'),
            'AUC':AUC(multi_class='ovo'),
            'Confusion_matrix':Confusion_Matrix(normalize='true')
        }

        self.parallel=None
        self.file=None
        self.verbose=False

config=default_config()