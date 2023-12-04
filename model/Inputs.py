import torch.nn as nn
import torch
from config import const

def init_data_attribute_ls(dataset_name):
    global user_attr_ls, item_attr_ls
    if dataset_name in ['kuaishou', 'wechat']:
        user_attr_ls = ['id']
        item_attr_ls = ['id', 'duration', 'tag']
    else:
        return ValueError


class user_feat(nn.Module):
    def __init__(self):
        super().__init__()

        global user_attr_ls
        self.attr_ls = user_attr_ls

        self.size = 0
        for attr in self.attr_ls:
            setattr(
                self, f'user_{attr}_emb', 
                nn.Embedding(
                    num_embeddings = getattr(const, f'user_{attr}_num'),
                    embedding_dim = getattr(const, f'user_{attr}_dim')
                )
            )
            self.size += getattr(const, f'user_{attr}_dim')

    def get_emb(self, sample):
        feats_ls = []
        for i in range(sample.shape[-1]):
            attr = sample[:, i]
            feats_ls.append(
                getattr(self, f'user_{self.attr_ls[i]}_emb')(attr)
            )  # [ [B, embed_1], [B, embed_2], ... , [B, embed_F] ]
        return torch.cat(feats_ls, dim=-1)



class item_feat(nn.Module):
    def __init__(self):
        super().__init__()

        global item_attr_ls
        self.attr_ls = item_attr_ls

        self.size = 0
        for attr in self.attr_ls:
            setattr(
                self, f'item_{attr}_emb', 
                nn.Embedding(
                    num_embeddings = getattr(const, f'item_{attr}_num'),
                    embedding_dim = getattr(const, f'item_{attr}_dim'),
                    padding_idx = 0 if attr in ['id'] else None
                )
            )
            self.size += getattr(const, f'item_{attr}_dim')
        
    def get_emb(self, sample):
        feats_ls = []
        if len(sample.shape) == 2:
            for i in range(sample.shape[-1]):
                attr = sample[:, i]
                feats_ls.append(getattr(self, f'item_{self.attr_ls[i]}_emb')(attr))
        else:
             for i in range(sample.shape[-1]):
                attr = sample[:, :, i].reshape(-1)
                feats_ls.append(getattr(self, f'item_{self.attr_ls[i]}_emb')(attr).reshape(sample.shape[0], sample.shape[1], -1))

        return torch.cat(feats_ls, dim=-1)
 