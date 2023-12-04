import os
from utils.data import get_dataloader
import config.const as const_util
import utils.Context.ContextManager as CM
import torch
import torch.optim as optim
import copy
import pickle
import numpy as np
import random
from time import time
import torch.nn as nn
from utils.topk import TopK_custom, set_device_for_topk


class Trainer(object):

    def __init__(self, flags_obj, cm,  dm, new_config=None):
        self.cm = cm #context manager
        self.dm = dm #dataset manager
        self.flags_obj = flags_obj

        self.recommender = CM.set_recommender(flags_obj, cm.workspace, dm, new_config)
        self.device = CM.set_device(self.flags_obj)
        self.recommender.transfer_model(self.device)

        self.lr_rec = self.flags_obj.lr_rec
        self.lr_label = self.flags_obj.lr_label
        
        self.set_train_dataloader()
        self.set_valid_dataloader()
        self.set_test_dataloader()

        set_device_for_topk(self.device)
        self.topk_model = TopK_custom(k=10, epsilon=1e-4).to(self.device)

        if self.flags_obj.dataset_name == 'kuaishou':
            if self.flags_obj.normalization == 1:

                def normalize_play_time(tensor):
                    mask = tensor <= 50.0
                    y = torch.zeros_like(tensor)
                    y[mask] = tensor[mask] / 50.0 * 0.2
                    if len(mask) < len(tensor):
                        y[~mask] = torch.min( (tensor[~mask]-50.0) / (800.0-50.0) * 0.8 + 0.2, torch.tensor([1.0]).to(self.device))[0]
                    return y

                def normalize_duration(tensor):
                    mask = tensor <= 235.0
                    y = torch.zeros_like(tensor)
                    y[mask] = tensor[mask] / 235.0 * 0.2
                    if len(mask) < len(tensor):
                        y[~mask] = torch.min( (tensor[~mask]-235.0) / (3600.0-235.0) * 0.8 + 0.2, torch.tensor([1.0]).to(self.device))[0]
                    return y
            else:
                def normalize_play_time(tensor):
                    return tensor
                def normalize_duration(tensor):
                    return tensor        

        elif self.flags_obj.dataset_name == 'wechat':
            if self.flags_obj.normalization == 1:
                def normalize_play_time(tensor):
                    mask = tensor <= 51.3
                    y = torch.zeros_like(tensor)
                    y[mask] = tensor[mask] / 51.3 * 0.2
                    if len(mask) < len(tensor):
                        y[~mask] = torch.min( (tensor[~mask]-51.3) / (8514.0-51.3) * 0.8 + 0.2, torch.tensor([1.0]).to(self.device))[0]
                    return y

                def normalize_duration(tensor):
                    mask = tensor <= 59.0
                    y = torch.zeros_like(tensor)
                    y[mask] = tensor[mask] / 59.0 * 0.2
                    if len(mask) < len(tensor):
                        y[~mask] = torch.min( (tensor[~mask]-59.0) / (10275.0-59.0) * 0.8 + 0.2, torch.tensor([1.0]).to(self.device))[0]
                    return y
            else:
                def normalize_play_time(tensor):
                    return tensor
                def normalize_duration(tensor):
                    return tensor
        self.normalize_duration = normalize_duration
        self.normalize_play_time = normalize_play_time

    def set_train_dataloader(self):
        self.train_dataset = self.recommender.get_dataset(const_util.train_file, self.dm, True)
        self.train_dataloader = get_dataloader(
            data_set = self.train_dataset,
            bs = self.dm.batch_size,
            collate_fn =  self.train_dataset.collate_func,
            shuffle = True
        )

    def set_valid_dataloader(self, k=10):
        self.valid_dataset = self.recommender.get_dataset(const_util.valid_file, self.dm, False)
        self.valid_user_counts = self.valid_dataset.sampler.record.groupby('user_id').size().to_dict()
        self.valid_user_counts= {key: value for key, value in self.valid_user_counts.items() if value > k}
        self.valid_dataloader = get_dataloader(
            data_set = self.valid_dataset,
            bs = self.dm.batch_size,
            collate_fn =  self.valid_dataset.collate_func,
            shuffle = False
        )

    def set_test_dataloader(self, k=10):
        self.test_dataset = self.recommender.get_dataset(const_util.test_file, self.dm, False)
        self.test_user_counts = self.test_dataset.sampler.record.groupby('user_id').size().to_dict()
        self.test_user_counts= {key: value for key, value in self.test_user_counts.items() if value > k}
        self.test_dataloader = get_dataloader(
            data_set = self.test_dataset,
            bs = self.dm.batch_size,
            collate_fn =  self.test_dataset.collate_func,
            shuffle = False
        )

    def test(self, k=10, user_counts=None, dataset=None):
        with torch.no_grad():
            model = self.recommender.model
            result_test_k = dict()
            test_user_num = len(user_counts)
            test_user_weight = list(user_counts.values())
            temp = sum(test_user_weight)
            test_user_weight = [x / temp for x in test_user_weight]
            for u in list(user_counts.keys()):
                items_info, feedbacks_info, scores = self.get_score_test(u, model, dataset=dataset)

                play_time = feedbacks_info[:, 0].float()
                duration = feedbacks_info[:, 1].float()
                lfc = feedbacks_info[:, 2].float()

                if len(play_time) > k:
                    _, top_indices = torch.topk(scores, k=k)
                    pos_weight = torch.log2(torch.arange(2, k+2)).to(self.device)

                    duration_result = torch.std(duration[top_indices]).cpu().numpy()

                    top_play_time_pos = torch.dot(pos_weight, torch.topk(play_time, k=k)[0])
                    top_play_time_pos_by_score = torch.dot(pos_weight, play_time[top_indices])
                    play_time_pos_result = (top_play_time_pos_by_score / top_play_time_pos).cpu().numpy() if top_play_time_pos > 0 else np.array(1.0)

                    top_lfc_pos = torch.dot(pos_weight, torch.topk(lfc, k=k)[0])
                    top_lfc_pos_by_score = torch.dot(pos_weight, lfc[top_indices])
                    lfc_pos_result = (top_lfc_pos_by_score / top_lfc_pos).cpu().numpy() if top_lfc_pos > 0 else np.array(1.0)

                    result_test_k[u] = [duration_result, 
                                        play_time_pos_result, 
                                        lfc_pos_result]
                
            temp = [v[0] for v in list(result_test_k.values())]
            duration_average_result = np.sum(temp) /  test_user_num
            duration_weighted_result =  sum([test_user_weight[i] * temp[i] for i in range(len(test_user_weight))])
            print("top@{} {} duration std:{}".format(k, 'average', duration_average_result))
            print("top@{} {} duration std:{}".format(k, 'weighted', duration_weighted_result))

            temp = [v[1] for v in list(result_test_k.values())]
            play_time_pos_average_result = np.sum(temp) /  test_user_num
            play_time_pos_weighted_result =  sum([test_user_weight[i] * temp[i] for i in range(len(test_user_weight))])
            print("top@{} {} play time ratio:{}".format(k, 'average', play_time_pos_average_result))
            print("top@{} {} play time ratio:{}".format(k, 'weighted', play_time_pos_weighted_result))

            temp = [v[2] for v in list(result_test_k.values())]
            lfc_pos_average_result = np.sum(temp) /  test_user_num
            lfc_pos_weighted_result =  sum([test_user_weight[i] * temp[i] for i in range(len(test_user_weight))])
            print("top@{} {} lfc ratio:{}".format(k, 'average', lfc_pos_average_result))
            print("top@{} {} lfc ratio:{}".format(k, 'weighted', lfc_pos_weighted_result))

            result = [duration_average_result, duration_weighted_result, 
                      play_time_pos_average_result, play_time_pos_weighted_result,
                      lfc_pos_average_result, lfc_pos_weighted_result]
        return result
    
    def get_score_test(self, user, model, params_updated=None, dataset=None):
      
        sample = dataset.get_user_batch_final(user)
        sample = [[k.to(self.device) for k in i] if type(i) == list else i.to(self.device) for i in sample]
        input_data, feedbacks = sample[:-1], sample[3]
        scores = model.forward(input_data=input_data, params_updated=params_updated)  # shape: [B]

        items_info = sample[1]
        feedbacks_info = sample[3]
        return items_info, feedbacks_info, scores
        

    def train_and_test(self):
        train_loss = [] #store every training loss
        valid_metric = [] #store every validation metric

        not_better = 0
        best_metric = 0

        save_name = ''
        for name_str, name_val in self.recommender.model_config.items():
            save_name += name_str + '_' + str(name_val) + '_'

        for epoch in range(self.flags_obj.epochs):
            loss = self.train_one_epoch(epoch, self.train_dataloader)
            print("TRAIN")
            print("train loss:{}".format(loss))
            train_loss.append(loss)

            if epoch % 1 == 0:
                print("VALIDATE")
                valid_result = self.test(k=10, user_counts=self.valid_user_counts, dataset=self.valid_dataset)
                valid_loss = self.get_loss(self.valid_dataloader)
                print("valid loss:", valid_loss)
                valid_metric.append(valid_result[3])

                print("TEST")
                test_result = self.test(k=10, user_counts=self.test_user_counts, dataset=self.test_dataset)
                test_loss = self.get_loss(self.test_dataloader)
                print("test loss:", test_loss)

                if valid_result[3] > best_metric + 1e-4:
                    not_better = 0
                    best_metric = valid_result[3]
                    torch.save(self.recommender.model.state_dict(), './workspace/ckpt/rec_' + self.flags_obj.extra_name + '_' + save_name + '.pth')
                    torch.save(self.recommender.labeler.state_dict(), './workspace/ckpt/labeler_' + self.flags_obj.extra_name + '_' + save_name + '.pth')
                else:
                    not_better += 1
                    if not_better >= 5:
                        print("end at epoch {}".format(epoch))
                        print("train loss", train_loss)
                        print("valid metric", valid_metric)
                        break
        
        print('best result:')
        self.recommender.model.load_state_dict(torch.load('./workspace/ckpt/rec_' + self.flags_obj.extra_name + '_' + save_name + '.pth'))
        self.recommender.labeler.load_state_dict(torch.load('./workspace/ckpt/labeler_' + self.flags_obj.extra_name + '_' + save_name + '.pth'))
        valid_result = self.test(k=10, user_counts=self.valid_user_counts, dataset=self.valid_dataset)
        test_result = self.test(k=10, user_counts=self.test_user_counts, dataset=self.test_dataset)
    
    def train_one_epoch(self, epoch, dataloader):
        self.recommender.model.train()
        self.recommender.labeler.train()
    
        loss_train = 0
        for step, sample in enumerate(dataloader):
            #t0 = time()
            sample = [[k.to(self.device) for k in i] if type(i) == list else i.to(self.device) for i in sample]
            #t1 = time()
            #print("Time: sample to device ", t1-t0)

            model_assume = copy.deepcopy(self.recommender.model).to(self.device)
            for param in model_assume.parameters():
                param.grad = None
            input_data, feedbacks = sample[:-1], sample[-1]
            labels = self.recommender.labeler(feedbacks).squeeze(-1)
            loss_assume = model_assume.forward(input_data, labels)
            grads = torch.autograd.grad(loss_assume, model_assume.parameters(), create_graph=True, retain_graph=True)
            
            params_assume_updated = list()
            for i, param in enumerate(model_assume.parameters()):
                param_updated = param - self.lr_rec * grads[i]
                params_assume_updated.append(param_updated)
    
            #t2 = time()
            #print("Time: substep1 ", t2-t1)

            valid_user_num = len(self.valid_user_counts)
            sampled_users = random.sample(list(self.valid_user_counts.keys()), int(valid_user_num * self.flags_obj.sample_ratio))
            result = dict()
            for u in sampled_users:
                result[u] = self.get_score_test(u, model_assume, params_assume_updated, self.valid_dataset)
              
            total_play_time, total_duration, total_lfc= self.get_metric(result=result, k=10)
            total_play_time = torch.stack(total_play_time)
            total_duration = torch.stack(total_duration)
            total_lfc = torch.stack(total_lfc)

            metric_play_time, metric_duration, metric_lfc = sum(total_play_time), sum(total_duration), sum(total_lfc)

            grads_alpha_play_time = torch.autograd.grad(metric_play_time, self.recommender.labeler.parameters(), retain_graph=True)
            grads_alpha_duration = torch.autograd.grad(metric_duration, self.recommender.labeler.parameters(), retain_graph=True)
            grads_alpha_lfc = torch.autograd.grad(metric_lfc, self.recommender.labeler.parameters(), retain_graph=True)

            grads_alpha_play_time_fc = self.flatten_and_concatenate(grads_alpha_play_time)
            grads_alpha_duration_fc = self.flatten_and_concatenate(grads_alpha_duration)
            grads_alpha_lfc_fc = self.flatten_and_concatenate(grads_alpha_lfc)

            if self.flags_obj.adapt == 0:
                weight = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)
            elif self.flags_obj.disable > 0:
                weight = torch.ones(3, dtype=torch.float32)
                weight[self.flags_obj.disable - 1] = 0
            else:
                weight = torch.stack([metric_play_time, metric_duration, metric_lfc])
                while torch.min(weight) < 1:
                    weight = weight * 10
                weight = torch.exp(-self.flags_obj.adapt * weight)
                weight = weight / torch.sum(weight)

            total_metric = metric_play_time * weight[0] +  metric_duration * weight[1] + metric_lfc * weight[2]

            for i, param in enumerate(self.recommender.labeler.parameters()):
                param.data = param.data + self.lr_label * (grads_alpha_play_time[i] * weight[0] +  grads_alpha_duration[i] * weight[1] + grads_alpha_lfc[i] * weight[2])
            #t3 = time()
            #print("Time: substep2 ", t3-t2)

            #self.rec_optimizer.zero_grad()
            with torch.no_grad():
                labels = self.recommender.labeler(feedbacks).squeeze(-1)
            loss = self.recommender.model.forward(input_data, labels)
            loss_train += loss.detach().cpu().numpy()
            grads_final = torch.autograd.grad(loss, self.recommender.model.parameters())
            for i, param in enumerate(self.recommender.model.parameters()):
                param.data = param.data - self.lr_rec * grads_final[i]
            #t4 = time()
            #print("Time: substep3 ", t4-t3)
            
            print("epoch:{}, step:{}, loss_assume:{}".format(epoch, step, loss_assume.data))
            print("metric_play_time:{}, metric_duration:{}, metric_lfc:{}".format(metric_play_time, metric_duration, metric_lfc))
            print("weight:{}, total_metric:{}".format(weight.data.cpu().numpy(), total_metric))
            print("loss:{}".format(loss))
            print("\n")
            
        return loss_train / step

    def get_loss(self, dataloader):
        with torch.no_grad():
            model = self.recommender.model
            labeler = self.recommender.labeler
            result = 0
            for step, sample in enumerate(dataloader):
                sample = [[k.to(self.device) for k in i] if type(i) == list else i.to(self.device) for i in sample]
                input_data, feedbacks = sample[:-1], sample[-1]
                labels = labeler(feedbacks).squeeze(-1)
                loss = model.forward(input_data, labels)
                result += loss.detach().cpu().numpy()
        return result / step    

    def flatten_and_concatenate(self, tensor_list):
        flat_tensor_list = []
        for tensor in tensor_list:
            flat_tensor = torch.flatten(tensor)
            flat_tensor_list.append(flat_tensor)
        concatenated_tensor = torch.cat(flat_tensor_list, dim=0)
        return concatenated_tensor
    
    def get_metric(self, result, k=10):
        total_play_time = []
        total_duration = []
        total_lfc = []
        for user, info in result.items():
            items_info = info[0]  # id,duration,tag
            feedbacks_info = info[1]  # play_time, duration, like follow comment
            
            play_time = feedbacks_info[:, 0].float()
            duration = feedbacks_info[:, 1].float()
            lfc = feedbacks_info[:, 2].float()

            scores = info[-1]
            N = items_info.shape[0]
            if N < k:
                continue
            
            is_topk = self.topk_model(scores.unsqueeze(dim=0))[0]
            is_topk = torch.sum(torch.sum(is_topk, dim=0), dim=1)
            
            play_time = self.normalize_play_time(play_time)
            duration = self.normalize_duration(duration)

            total_play_time.append(torch.dot(is_topk, play_time)/k)
            total_duration.append(torch.std(is_topk * duration))
            total_lfc.append(torch.dot(is_topk, lfc)/k)
            
        return total_play_time, total_duration, total_lfc