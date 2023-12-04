import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from data_sampler import *
from data_utils import *
import numpy as np

class BaseDataset(Dataset):
    
    def __init__(self):
        super(Dataset, self).__init__()
    def __len__(self):
        return self.sampler.record.shape[0]
    def __getitem__(self, index):
        raise NotImplementedError


class Pointwise_Dataset(BaseDataset):
    def __init__(self, dataset_file, flags_obj, is_train):
        super().__init__()
        self.sampler = Point_Sampler(dataset_file, flags_obj, is_train=is_train)


class DIN_Dataset(Pointwise_Dataset):
    def __init__(self, *args):
        super().__init__(*args)

    def __getitem__(self, index):
        user, item, rec_his, feedback = self.sampler.sample(index)
        feedback = [float(x) for x in feedback]
        return user, item, rec_his, feedback

    def get_user_batch_final(self, user):
        batch = self.sampler.get_user_batch(user)
        return batch

    @staticmethod
    def collate_func(batch):
        B = len(batch)
        user_feats = [batch[i][0] for i in range(B)]
        user_feats = torch.tensor(np.array(user_feats))
        item_feats = [batch[i][1] for i in range(B)]
        item_feats = torch.tensor(np.array(item_feats))
        rec_his_feats = [batch[i][2] for i in range(B)]
        rec_his_feats = torch.tensor(np.array(rec_his_feats))
        feedbacks = [batch[i][3] for i in range(B)]
        feedbacks = torch.tensor(np.array(feedbacks))
        return user_feats, item_feats, rec_his_feats, feedbacks


GLOBAL_SEED = 542674
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def get_dataloader(data_set, bs, shuffle=False, **kwargs):
    return DataLoader(  data_set, batch_size = bs,
                        shuffle=shuffle, pin_memory = True, 
                        worker_init_fn=worker_init_fn, **kwargs
                    )

