import torch.nn as nn
import torch
from module import FullyConnectedLayer
from Inputs import user_feat, item_feat


class DIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.user_feat = user_feat()  # user embed
        self.item_feat = item_feat()  # item embed
        self.attn = AttentionSequencePoolingLayer(embedding_dim=self.item_feat.size, hidden_unit=[36])
        self.fc_layer = FullyConnectedLayer(input_size=2*self.item_feat.size+self.user_feat.size,
                                            hidden_unit=config['hid_units'],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='dice',
                                            dropout=0,
                                            dice_dim=2)
        self.loss_func = nn.MSELoss()
        self._init_weights()
        

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m)


    def input_from_feature_tables(self, input_data):
        # user: [B, F_u]
        user, item, rec_his = input_data
        user_emb = self.user_feat.get_emb(user)  # [B, user_embed]
        item_emb = self.item_feat.get_emb(item)  # [B, item_embed]
        # rec_his: [B, T, F_i]
        rec_his_emb = self.item_feat.get_emb(rec_his)  # [B, T, item_embed]
        rec_his_item = rec_his[: , :, 0].squeeze(-1)  # [B, T]
        rec_his_mask = torch.where(rec_his_item==0, 1, 0).bool()
        return user_emb, item_emb, rec_his_emb, rec_his_mask

    def forward(self, input_data, labels=None, params_updated=None):
        if params_updated == None:
            user_emb, item_emb, rec_his_emb, rec_his_mask = self.input_from_feature_tables(input_data=input_data)
            browse_atten = self.attn(item_emb.unsqueeze(dim=1), rec_his_emb, rec_his_mask)
            concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb], dim=-1)
            output = self.fc_layer(concat_feature).squeeze(dim=-1) #[B] 

        elif len(params_updated) == 20:  # manual forward with updated parameters
            user, item, rec_his = input_data
            user_emb = params_updated[0][user].squeeze(1)
            item_emb = torch.cat((params_updated[1][item[:,0]], params_updated[2][item[:,1]], params_updated[3][item[:,2]]), dim=1)
            rec_his_emb = torch.cat((params_updated[1][rec_his[:,:,0]], params_updated[2][rec_his[:,:,1]], params_updated[3][rec_his[:,:,2]]), dim=2)
            rec_his_item = rec_his[: , :, 0].squeeze(-1)
            rec_his_mask = torch.where(rec_his_item==0, 1, 0).bool()

            query = item_emb.unsqueeze(dim=1)
            user_behavior = rec_his_emb
            user_behavior_len = user_behavior.size(1)
            queries = query.expand(-1, user_behavior_len, -1)  
            attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior], dim=-1) 
            attention_output = torch.matmul(attention_input, params_updated[4].t()) + params_updated[5]
            attention_score = torch.matmul(attention_output, params_updated[6].t()) + params_updated[7]
            attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
            attention_score = attention_score.masked_fill(rec_his_mask.unsqueeze(1), torch.tensor(0)) 
            browse_atten = torch.matmul(attention_score, user_behavior)  # B * 1 * embed
            concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb], dim=-1)  
            x = torch.matmul(concat_feature, params_updated[8].t()) + params_updated[9]
            mean = torch.mean(x, dim=0, keepdim=True)
            var = torch.var(x, dim=0, keepdim=True)
            x = (x - mean) / torch.sqrt(var + 1e-8)
            x = x * params_updated[11].view(1, -1) + params_updated[12].view(1, -1)
            x_p = nn.Sigmoid()(x)  # B * embed
            x = params_updated[10] * (1 - x_p) * x + x_p * x 

            x = torch.matmul(x, params_updated[13].t()) + params_updated[14]
            mean = torch.mean(x, dim=0, keepdim=True)
            var = torch.var(x, dim=0, keepdim=True)
            x = (x - mean) / torch.sqrt(var + 1e-8)
            x = x * params_updated[16].view(1, -1) + params_updated[17].view(1, -1)
            x_p = nn.Sigmoid()(x)  # B * embed
            x = params_updated[15] * (1 - x_p) * x + x_p * x

            x = torch.matmul(x, params_updated[18].t()) + params_updated[19]
            x = nn.Sigmoid()(x)
            return x.squeeze(dim=-1) #[B] 
            
        else:
            return NotImplementedError


        if labels is not None:
            labels = labels.float()
            return self.loss_func(output, labels).float()
        else:
            return output.float()


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_unit):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.local_att = LocalActivationUnit(hidden_unit=hidden_unit, embedding_dim=embedding_dim, batch_norm=False)

    
    def forward(self, query_ad, user_behavior, mask=None):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # mask                : size -> batch_size * time_seq_len
        # output              : size -> batch_size * 1 * embedding_size
        attention_score = self.local_att(query_ad, user_behavior)  # B * T * 1
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        if mask is not None:
            attention_score = attention_score.masked_fill(mask.unsqueeze(1), torch.tensor(0))
        output = torch.matmul(attention_score, user_behavior)  # B * 1 * embed
        return output
        

class LocalActivationUnit(nn.Module):

    def __init__(self, hidden_unit, embedding_dim, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim, 
                                       hidden_unit=hidden_unit,
                                       batch_norm=batch_norm,
                                       sigmoid=False,
                                       activation='dice',
                                       dice_dim=3) 
        self.fc2 = nn.Linear(hidden_unit[-1], 1) 

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        user_behavior_len = user_behavior.size(1)
        queries = query.expand(-1, user_behavior_len, -1)
        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior], dim=-1) 
        attention_output = self.fc1(attention_input)
        attention_score = self.fc2(attention_output)
        return attention_score
