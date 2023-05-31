import torch
import numpy as np
import random
from Dataset.Office31 import Office31
from Dataset.VisDA import VisDA
from Dataset.ImageCLEF import ImageCLEF
from LAMDA_SSL.Split.DataSplit import DataSplit
import copy
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Transform.Vision.Normalization import Normalization
from sklearn.pipeline import Pipeline
import torchvision.transforms as transforms
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Augmentation.Vision.RandomHorizontalFlip import RandomHorizontalFlip
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from Network.ResNet50Fc import ResNet50Fc
from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Scheduler.CosineWarmup import CosineWarmup
from Scheduler.DAScheduler import DAScheduler
from Network.TransferNet import TransferNet
from Algorithm.BDA import BDA
from Algorithm.UnlabeledAdaption import UnlabeledAdaption
from Algorithm.LabeledAdaption import LabeledAdaption
from LAMDA_SSL.utils import class_status
from sklearn.metrics import accuracy_score
import argparse
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='Office-31')
parser.add_argument('--root', type=str, default='/data/jialh/Office-31')
parser.add_argument('--source', type=str, default='dslr')
parser.add_argument('--target', type=str, default='amazon')
parser.add_argument('--lt', type=bool, default=False)
parser.add_argument('--op', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--iteration', type=int, default=2000)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--weight', type=bool, default=True)
parser.add_argument('--lambda_t', type=float, default=0.5)
parser.add_argument('--soft', type=bool, default=True)
parser.add_argument('--lambda_s', type=float, default=0.1)
parser.add_argument('--T', type=float, default=None)
parser.add_argument('--warmup', type=float, default=None)
parser.add_argument('--threshold', type=float, default=None)
parser.add_argument('--labels', type=int, default=100)
parser.add_argument('--unlabels', type=int, default=None)
parser.add_argument('--load_from_path', type=bool, default=False)
parser.add_argument('--lr',type=float,default=5e-4)
parser.add_argument('--verbose', type=bool, default=True)

args = parser.parse_args()
root = args.root
source = args.source
target = args.target
batch_size = args.batch_size
iteration= args.iteration
device=args.device
soft=args.soft
weight=args.weight
load_from_path=args.load_from_path
lambda_t=args.lambda_t
lambda_s=args.lambda_s
warmup=args.warmup
T=args.T
labels=args.labels
unlabels=args.unlabels
threshold=args.threshold
transfer_loss=args.transfer_loss
lr=args.lr
dataset=args.dataset
verbose=args.verbose
lt=args.lt
op=args.op

if dataset=='Office-31':
    if op:
        source_dataset = Office31(root=root, domain=source,
            classnames=['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle',
            'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer',
            'file_cabinet', 'headphones', 'keyboard', 'laptop_computer',
            'letter_tray', 'mobile_phone'])
        num_classes = 15
    else:
        source_dataset=Office31(root=root,domain=source)
        num_classes = 31
    target_dataset=Office31(root=root,domain=target)

elif dataset=='image-CLEF':
    if op:
        source_dataset = ImageCLEF(root=root, domain=source,classnames=['0', '1', '2',
                                                                     '3', '4', '5'])
        num_classes = 6
    else:
        source_dataset=ImageCLEF(root=root,domain=source)
        num_classes = 12
    target_dataset=ImageCLEF(root=root,domain=target)

else:
    if op:
        source_dataset = VisDA(root=root, domain=source,classnames=['aeroplane',
                                                                    'bicycle',
                                                                    'bus',
                                                                    'car',
                                                                    'horse',
                                                                    'knife'])
        num_classes = 6
    else:
        source_dataset=VisDA(root=root,domain=source)
        num_classes = 12
    target_dataset = VisDA(root=root,domain=target)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_mena_std(imgs):
    tmp_imgs = []
    for _ in range(len(imgs)):
        tmp_imgs.append(np.array(imgs[_]).transpose(2,0,1).reshape(3,-1))
    tmp_imgs=np.hstack(tmp_imgs)
    mean = np.mean(tmp_imgs / 255, axis=1)
    std = np.std(tmp_imgs / 255, axis=1)
    return mean,std



performance_list=[]
file_f = open('./Result/BDA_'+source+'_'+target+'_'+str(labels)+'_op_'+str(op)+'_lt_'+str(lt)+'_final.txt' ,"w")
for _ in range(0,5):
    seed = _
    def worker_init(worked_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    S_X, S_y = source_dataset.imgs, source_dataset.data_labels
    T_X, T_y = target_dataset.imgs, target_dataset.data_labels


    set_seed(seed)
    S_y=np.array(S_y)
    T_y=np.array(T_y)
    if lt:
        img_max=len(T_X)/num_classes
        class_st=class_status(T_y)
        num_per_cls=class_st.class_counts
        imb_factor=0.1
        for cls_idx in range(num_classes):
            num = img_max * (imb_factor**(cls_idx / (num_classes - 1.0)))
            num_per_cls[cls_idx]=min(int(num),num_per_cls[cls_idx])
        y_indices=class_st.y_indices

        index_list=[]
        for _ in range(num_classes):
            y_ind=np.argwhere(y_indices==_).flatten().tolist()
            y_ind=random.sample(y_ind,num_per_cls[_])
            index_list=index_list+y_ind
        _T_X=copy.copy(T_X)
        T_X=[_T_X[_] for _ in index_list]
        T_y=T_y[index_list]
        unlabels=None

    labeled_X, labeled_y, test_X, test_y = DataSplit(stratified=True, shuffle=True, X=S_X, y=S_y, size_split=labels,
                                                     random_state=seed)
    if unlabels is not None:
        unlabeled_X, unlabeled_y,_,_ = DataSplit(stratified=True, shuffle=True, X=T_X, y=T_y, size_split=unlabels,
                                                     random_state=seed)
    else:
        unlabeled_X, unlabeled_y = T_X, T_y

    mean,std=get_mena_std(labeled_X)

    train_pre_transform = transforms.Compose([transforms.Resize([256, 256]), transforms.RandomCrop(224)])
    valid_pre_transform = transforms.Compose([transforms.Resize([256, 256]), transforms.CenterCrop(224)])
    test_pre_transform = transforms.Compose([transforms.Resize([256, 256]), transforms.CenterCrop(224)])
    transform = Pipeline([('ToTensor', ToTensor(dtype='float', image=True)), ('Normalization', Normalization(mean=mean, std=std))])

    weak_augmentation = RandomHorizontalFlip()

    labeled_dataset = LabeledDataset(pre_transform=train_pre_transform, transform=transform)

    unlabeled_dataset = UnlabeledDataset(pre_transform=train_pre_transform, transform=transform)

    valid_dataset = UnlabeledDataset(pre_transform=valid_pre_transform, transform=transform)

    test_dataset = UnlabeledDataset(pre_transform=test_pre_transform, transform=transform)

    labeled_sampler = RandomSampler(replacement=True, num_samples=batch_size * iteration)
    unlabeled_sampler = RandomSampler(replacement=True)
    valid_sampler = SequentialSampler()
    test_sampler = SequentialSampler()

    labeled_dataloader = LabeledDataLoader(batch_size=batch_size, num_workers=0, drop_last=True,worker_init_fn=worker_init)
    unlabeled_dataloader = UnlabeledDataLoader(num_workers=0, drop_last=True,worker_init_fn=worker_init)
    valid_dataloader = UnlabeledDataLoader(batch_size=batch_size, num_workers=0, drop_last=False,worker_init_fn=worker_init)
    test_dataloader = UnlabeledDataLoader(batch_size=batch_size, num_workers=0, drop_last=False,worker_init_fn=worker_init)

    network_unlabeled=TransferNet(num_class=num_classes,transfer_loss=transfer_loss)
    network_labeled = ResNet50Fc(num_classes=num_classes,output_feature=True)
    network_meta = ResNet50Fc(num_classes=num_classes, output_feature=True)

    optimizer_unlabeled=SGD(lr=lr,momentum=0.9)
    optimizer_labeled=SGD(lr=lr,momentum=0.9)

    scheduler_unlabeled=DAScheduler()
    scheduler_labeled=CosineWarmup(num_cycles=7./16,num_training_steps=iteration)

    file_u = open('./Result/BDA_' + dataset+'_'+ source +'_'+target+ '_' + str(seed) +'_'+str(labels)+'_op_'+str(op)+'_'+str(iteration)+'_'+str(weight)+'_lt_'+str(lt)+'_unlabeledadaption'+str(verbose)+'.txt', "w")
    file_l = open('./Result/BDA_'+str(soft)+'_'+ dataset+'_'+ source +'_'+target+ '_' + str(seed) +'_'+str(labels)+'_op_'+str(op)+'_'+str(iteration)+'_'+str(weight)+'_lt_'+str(lt)+'_labeledadaption'+str(verbose)+str(T)+'.txt', "w")
    UDA= UnlabeledAdaption(
        lambda_t=lambda_t,mu=1,
        weight_decay=5e-4,epoch=1,
        num_it_epoch=2000, num_it_total=iteration,
        device=device, num_classes=num_classes,
        eval_it=None,
        labeled_dataset=copy.deepcopy(labeled_dataset),
        unlabeled_dataset=copy.deepcopy(unlabeled_dataset),
        valid_dataset=copy.deepcopy(valid_dataset),
        test_dataset=copy.deepcopy(test_dataset),
        labeled_sampler=copy.deepcopy(labeled_sampler),
        unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
        valid_sampler=copy.deepcopy(valid_sampler),
        test_sampler=copy.deepcopy(test_sampler),
        labeled_dataloader=copy.deepcopy(labeled_dataloader),
        unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
        valid_dataloader=copy.deepcopy(valid_dataloader),
        test_dataloader=copy.deepcopy(test_dataloader),
        augmentation=weak_augmentation,
        network=copy.deepcopy(network_unlabeled),
        optimizer=copy.deepcopy(optimizer_unlabeled),
        scheduler=copy.deepcopy(scheduler_unlabeled),
        file=file_u,
        verbose=verbose
    )

    LDA= LabeledAdaption(
        lambda_s=lambda_s,
        mu=1, weight_decay=5e-4,weight=weight,warmup=warmup,
        T=T,num_classes=num_classes,
        threshold=threshold,
        eval_it=None,epoch=1,
        num_it_epoch=iteration, num_it_total=iteration,
        device=device,soft=soft,
        labeled_dataset=copy.deepcopy(labeled_dataset),
        unlabeled_dataset=copy.deepcopy(unlabeled_dataset),
        valid_dataset=copy.deepcopy(valid_dataset),
        test_dataset=copy.deepcopy(test_dataset),
        labeled_sampler=copy.deepcopy(labeled_sampler),
        unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
        valid_sampler=copy.deepcopy(valid_sampler),
        test_sampler=copy.deepcopy(test_sampler),
        labeled_dataloader=copy.deepcopy(labeled_dataloader),
        unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
        valid_dataloader=copy.deepcopy(valid_dataloader),
        test_dataloader=copy.deepcopy(test_dataloader),
        augmentation=weak_augmentation,
        network=network_labeled,
        optimizer=optimizer_labeled,
        scheduler=None,
        file=file_l,
        verbose=verbose
    )

    model = BDA(UnlabeledDomainAdaption=UDA,LabeledDomainAdaption=LDA,
                      pred_path='./Pseudo_Label/'+source +'_'+target+'_'+str(seed)+'_'+str(labels)+'_op_'+str(op)+'_'+str(iteration)+'_'+transfer_loss+'_'+str(lambda_t)+'_lt_'+str(lt)+'_pred.npy',
                      score_path='./Pseudo_Label/'+source +'_'+target+'_'+str(seed)+'_'+str(labels)+'_op_'+str(op)+'_'+str(iteration)+'_'+transfer_loss+'_'+str(lambda_t)+'_lt_'+str(lt)+'_score.npy',
                      T=T,threshold=threshold,load_from_path=load_from_path)

    model.fit(X=labeled_X, y=labeled_y,unlabeled_X=unlabeled_X,unlabeled_y=unlabeled_y)

    pred=model.predict(test_X)

    performance=accuracy_score(test_y,pred)

    performance_list.append(performance)

    print(performance, file=file_l)

performance_list = np.array(performance_list)
mean = performance_list.mean()
std = performance_list.std()

print(performance_list,file=file_f)
print(mean,file=file_f)
print(std,file=file_f)
