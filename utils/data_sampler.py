from data_utils import NpyLoader, CsvLoader, JsonLoader, PickleLoader
import random
from config import const
import torch
import numpy as np


class Sampler(object):
    def __init__(self, dataset_file, load_path):
        self.load_path = load_path
        self.csv_loader = CsvLoader(load_path)
        self.json_loader = JsonLoader(load_path)
        self.pickle_loader = PickleLoader(load_path)
        self.record = self.pickle_loader.load(filename=dataset_file)
        self.cache = dict()
    
    def sample(self, index, **kwargs):
        raise NotImplementedError


class Point_Sampler(Sampler):
    def __init__(self, dataset_file, args, is_train):
        super().__init__(dataset_file, args.load_path)
        self.is_train = is_train
        self.dataset_name = args.dataset_name
        self.build(args)

    def build(self, args):
        self.max_rec_his = const.max_rec_his_len
        self.parse_line = self.custom_parse_line 

    def custom_parse_line(self, index):
        line = self.record.iloc[index]
        user = [line['user_id']]
        item = [line['item_id'], line['item_duration'], line['item_tag']]
        rec_his = line['rec_his']
        play_time = float(line['play_time'])
        duration = float(line['duration'])
        lfc = float(line['lfc'])
        
        feedback = [play_time, duration, lfc]

        return user, item, rec_his, feedback

    def sample(self, index):
        user, item, rec_his, feedback = self.parse_line(index)
        rec_his = rec_his[-self.max_rec_his:]
        if len(rec_his) < self.max_rec_his:
            rec_his = [[0,0,0]] * (self.max_rec_his - len(rec_his)) + rec_his
        return user, item, rec_his, feedback
    
    def get_user_batch(self, user):
        base = self.record.query('user_id=='+str(user))
        user_feats = torch.tensor(np.array(base['user_id'])).unsqueeze(1)
        item_feats = torch.stack((torch.tensor(np.array(base['item_id'])), torch.tensor(np.array(base['item_duration'])), torch.tensor(np.array(base['item_tag'])))).T
        rec_his_feats = [[[0,0,0]]*(self.max_rec_his - len(r)) + r for r in base['rec_his']]
        rec_his_feats = torch.tensor(np.array(rec_his_feats))
        temp = (torch.tensor(np.array(base['play_time'])),
                torch.tensor(np.array(base['duration'])), 
                torch.tensor(np.array(base['lfc'])), 
                )
        feedbacks = torch.stack(temp).T
        feedbacks = torch.tensor(np.array(feedbacks))
        return user_feats, item_feats, rec_his_feats, feedbacks
