import torch
import torch.nn as nn
import torch.optim as optim
from model.DIN import DIN
import utils.data as data
import config.const as const_util
import os
import yaml
from model.MLP import Labeler


class Recommender(object):
    def __init__(self, flags_obj, workspace, dm, nc=None):
        self.dm = dm # dataset manager
        self.model_name = flags_obj.model
        self.flags_obj = flags_obj
        self.load_model_config()
        self.set_model()
        self.set_labeler()
        self.workspace = workspace

    def load_model_config(self):
        path = './config/{}_{}.yaml'.format(self.model_name, self.dm.dataset_name)
        f = open(path)
        self.model_config = yaml.load(f, Loader=yaml.FullLoader)

    def set_model(self):
        self.model = DIN(config=self.model_config)

    def set_labeler(self):
        self.labeler = Labeler(feedback_num=self.model_config['feedback_num'], dim_num=self.model_config['dim_num'])

    def transfer_model(self, device):
        self.model = self.model.to(device)
        self.labeler = self.labeler.to(device)

    def get_dataset(self, *args):
        return getattr(data, f'DIN_Dataset')(*args)