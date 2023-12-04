import os
import torch
import config.const as const_util
from trainer import Trainer
from recommender import Recommender


class ContextManager(object):

    def __init__(self, flags_obj):
        self.workspace = flags_obj.workspace
        self.set_workspace()

    def set_workspace(self):
        if not os.path.exists(self.workspace):
            os.mkdir(self.workspace)
        
    @staticmethod
    def set_trainer(flags_obj, cm,  dm, nc=None):
        if flags_obj.model == 'DIN':
            return Trainer(flags_obj, cm, dm, nc)
        else:
            raise NameError('trainer model name error!')

    @staticmethod
    def set_recommender(flags_obj, workspace, dm, new_config):
        return Recommender(flags_obj, workspace, dm, new_config)

    @staticmethod
    def set_device(flags_obj):
        if not flags_obj.use_gpu:
            return torch.device('cpu')
        else:
            return torch.device('cuda:{}'.format(flags_obj.gpu_id))


class DatasetManager(object):

    def __init__(self, flags_obj):

        self.dataset_name = flags_obj.dataset_name
        self.batch_size = flags_obj.batch_size
        self.set_dataset()

    def set_dataset(self):
        setattr(self, 'load_path', getattr(const_util, 'load_path'))

    def show(self):
        print(self.__dict__)