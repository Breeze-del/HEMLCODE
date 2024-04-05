

import enum
import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, HGNN, LinearAttention


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.matmul(z1, z2.permute(0,2,1))

class MBHT(SequentialRecommender):

    def __init__(self, config, dataset):
        super(MBHT, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.mask_ratio = config['mask_ratio']

        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        self.hglen = config['hyper_len']
        self.enable_hg = config['enable_hg']
        self.enable_ms = config['enable_ms']
        self.dataset = config['dataset']

        self.buy_type = dataset.field2token_id["item_type_list"]['0']

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.type_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
        if self.enable_ms:
            self.trm_encoder = TransformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                multiscale=True,
                scales=config["scales"]
            )
        else:
            self.trm_encoder = TransformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                multiscale=False
            )
        self.hgnn_layer = HGNN(self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.hg_type_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        self.metric_w1 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.metric_w2 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.gating_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.gating_bias = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.attn_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)
        # nn.init.normal_(self.gating_bias, std=0.02)
        nn.init.normal_(self.gating_weight, std=0.02)
        nn.init.normal_(self.metric_w1, std=0.02)
        nn.init.normal_(self.metric_w2, std=0.02)

        # create capsule
        self.interest_num = 4
        self.capsule_1 = CapsuleNetwork(self.hidden_size, int(self.max_seq_length / config["scales"][1]), bilinear_type=2, num_interest=self.interest_num,
                                         hard_readout=True, relu_layer=False)
        self.capsule_2 = CapsuleNetwork(self.hidden_size, int(self.max_seq_length / config["scales"][2]), bilinear_type=2, num_interest=self.interest_num,
                                         hard_readout=True, relu_layer=False)
        # co-guide fusion
        self.w_pi_1 = nn.Linear(self.hidden_size , self.hidden_size , bias=True)
        self.w_pi_2 = nn.Linear(self.hidden_size , self.hidden_size, bias=True)
        self.w_c_z = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.u_j_z = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.w_c_r = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.u_j_r = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.w_p = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.u_p = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.w_i = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.u_i = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.mlp_co_guide = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.mlp_fusion = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)

        # interest conv
        # number of horizontal filters
        self.n_h = 2
        # number of vertical filters
        self.n_v = 2
        self.fc1_dim_v = self.n_v * self.hidden_size
        self.fc1_dim_h = self.n_h * self.interest_num
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.hidden_size)
        activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}
        self.conv_v = nn.Conv2d(1, self.n_v, (self.interest_num, 1))
        lengths = [i + 1 for i in range(self.interest_num)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, self.hidden_size)) for i in lengths])
        self.dropout = nn.Dropout(0.5, True)
        self.ac_conv = activation_getter['relu']
        self.ac_fc = activation_getter['relu']

        # self.frnet = FRNet(self.max_seq_length, self.hidden_size)


        # self.attention1 = LinearAttention(self.n_heads, self.hidden_size, self.hidden_dropout_prob, self.attn_dropout_prob,
        #                                   self.layer_norm_eps)



        if self.dataset == "retail_beh":
            self.sw_before = 10
            self.sw_follow = 6
        elif self.dataset == "ijcai_beh":
            self.sw_before = 30
            self.sw_follow = 18
        elif self.dataset == "tmall_beh":
            self.sw_before = 20
            self.sw_follow = 12

        self.hypergraphs = dict()
        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ['BPR', 'CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-scale attention."""
        if self.enable_ms:
            attention_mask = (item_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1)
            return extended_attention_mask
        else:
            """Generate bidirectional attention mask for multi-head attention."""
            attention_mask = (item_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
            # bidirectional mask
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            return extended_attention_mask

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def reconstruct_train_data(self, item_seq, type_seq, last_buy):
        """
        Mask item sequence for training.
        """
        last_buy = last_buy.tolist()
        device = item_seq.device
        batch_size = item_seq.size(0)

        zero_padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)
        item_seq = torch.cat((item_seq, zero_padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        type_seq = torch.cat((type_seq, zero_padding.unsqueeze(-1)), dim=-1)
        n_objs = (torch.count_nonzero(item_seq, dim=1)+1).tolist()
        for batch_id in range(batch_size):
            n_obj = n_objs[batch_id]
            item_seq[batch_id][n_obj-1] = last_buy[batch_id]
            type_seq[batch_id][n_obj-1] = self.buy_type

        sequence_instances = item_seq.cpu().numpy().tolist()
        type_instances = type_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        masked_index = []

        for instance_idx, instance in enumerate(sequence_instances):
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if index_id == n_objs[instance_idx]-1:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)

            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        type_instances = torch.tensor(type_instances, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence, pos_items, masked_index, type_instances

    def reconstruct_test_data(self, item_seq, item_seq_len, item_type):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        item_type = torch.cat((item_type, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        return item_seq, item_type

    def co_guide_fusion(self, fine_interest, coarse_interest):
        # fine_interest = fine_interest.view([fine_interest.size(0), -1])
        # coarse_interest = coarse_interest.view([coarse_interest.size(0), -1])
        # Co-guided Learning
        m_c = torch.tanh(self.w_pi_1(fine_interest * coarse_interest))
        m_j = torch.tanh(self.w_pi_2(fine_interest + coarse_interest))

        r_i = torch.sigmoid(self.w_c_z(m_c) + self.u_j_z(m_j))
        r_p = torch.sigmoid(self.w_c_r(m_c) + self.u_j_r(m_j))

        m_p = torch.tanh(self.w_p(fine_interest * r_p) + self.u_p((1 - r_p) * coarse_interest))
        m_i = torch.tanh(self.w_i(coarse_interest * r_i) + self.u_i((1 - r_i) * fine_interest))

        # enriching the semantics of price and interest preferences
        p_pre = (fine_interest + m_i) * m_p
        i_pre = (coarse_interest + m_p) * m_i

        # output = torch.stack([p_pre, i_pre], dim=1).view([fine_interest.size(0), -1])
        # output = torch.tanh(self.mlp_co_guide(output))
        output = torch.cat([p_pre, i_pre],dim=2)
        return output

    def feature_conv(self, interest_emb):
        interest_emb = torch.unsqueeze(interest_emb, dim= 1)
        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(interest_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(interest_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        return z

    def select_interest(self, interest1, interest2, input):
        interest = torch.stack((interest1, interest2), dim=1)
        score = torch.einsum("abc,adc->adb", interest, input)
        mask = (score == score.max(dim=2, keepdim=True)[0]).to(dtype=torch.float)
        res = torch.einsum("abc,acd->abd", mask, interest)
        return res


    def forward(self, item_seq, type_seq, mask_positions_nums=None, session_id=None, is_inference = False):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        type_embedding = self.type_embedding(type_seq)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding + type_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        # mask code is here
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output, trm_1, trm_2 = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        # 得到最终两个尺度的兴趣表示
        final_trm_1 = trm_1[-1]
        interest_u_1 = self.capsule_1(final_trm_1)
        final_trm_2 = trm_2[-1]
        interest_u_2 = self.capsule_2(final_trm_2)
        interest_u_1 = self.feature_conv(interest_u_1)
        interest_u_2 = self.feature_conv(interest_u_2)

        # interest_select
        interest_u = self.select_interest(interest_u_1, interest_u_2, output)

        # interest_u = self.co_guide_fusion(interest_u_1, interest_u_2)

        if self.enable_hg:
            x_raw = item_emb
            x_raw = x_raw * torch.sigmoid(x_raw.matmul(self.gating_weight) + self.gating_bias)
            # b, l, l
            x_m = torch.stack((self.metric_w1 * x_raw, self.metric_w2 * x_raw)).mean(0)
            item_sim = sim(x_m, x_m)
            item_sim[item_sim < 0] = 0.01

            Gs = self.build_Gs_unique(item_seq, item_sim, self.hglen)
            # Gs = self.build_Gs_light(item_seq, item_sim, self.hglen)

            batch_size = item_seq.shape[0]
            seq_len = item_seq.shape[1]
            n_objs = torch.count_nonzero(item_seq, dim=1)
            indexed_embs = list()
            for batch_idx in range(batch_size):
                n_obj = n_objs[batch_idx]
                # l', dim
                indexed_embs.append(x_raw[batch_idx][:n_obj])
            indexed_embs = torch.cat(indexed_embs, dim=0)
            hgnn_embs = self.hgnn_layer(indexed_embs, Gs)
            hgnn_take_start = 0
            hgnn_embs_padded = []
            for batch_idx in range(batch_size):
                n_obj = n_objs[batch_idx]
                embs = hgnn_embs[hgnn_take_start:hgnn_take_start + n_obj]
                hgnn_take_start += n_obj
                # l', dim || padding emb -> l, dim
                padding = torch.zeros((seq_len - n_obj, embs.shape[-1])).to(item_seq.device)
                embs = torch.cat((embs, padding), dim=0)
                if mask_positions_nums is not None:
                    mask_len = mask_positions_nums[1][batch_idx]
                    poss = mask_positions_nums[0][batch_idx][-mask_len:].tolist()
                    for pos in poss:
                        if pos == 0:
                            continue
                        # if pos<n_obj-1:
                        #     readout = torch.mean(torch.cat((embs[:pos], embs[pos+1:]), dim=0), dim=0)
                        # else:
                        sliding_window_start = pos - self.sw_before if pos - self.sw_before > -1 else 0
                        sliding_window_end = pos + self.sw_follow if pos + self.sw_follow < n_obj else n_obj - 1
                        readout = torch.mean(
                            torch.cat((embs[sliding_window_start:pos], embs[pos + 1:sliding_window_end]), dim=0), dim=0)
                        embs[pos] = readout
                else:
                    pos = (item_seq[batch_idx] == self.mask_token).nonzero(as_tuple=True)[0][0]
                    sliding_window_start = pos - self.sw_before if pos - self.sw_before > -1 else 0
                    embs[pos] = torch.mean(embs[sliding_window_start:pos], dim=0)
                hgnn_embs_padded.append(embs)
            # b, l, dim
            hgnn_embs = torch.stack(hgnn_embs_padded, dim=0)
            # x = x_raw
            # 2, b, l, dim

            mixed_x= self.co_guide_fusion(interest_u, hgnn_embs)
            mixed_x = torch.cat((output, mixed_x), dim=2)
            mixed_x = torch.tanh(self.mlp_fusion(mixed_x))

            assert not torch.isnan(mixed_x).any()
            return mixed_x
        return output  # [B L H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        # begin here
        item_seq = interaction[self.ITEM_SEQ]
        session_id = interaction['session_id']
        item_type = interaction["item_type_list"]
        last_buy = interaction["item_id"]
        masked_item_seq, pos_items, masked_index, item_type_seq = self.reconstruct_train_data(item_seq, item_type,
                                                                                              last_buy)

        mask_nums = torch.count_nonzero(pos_items, dim=1)
        # seq_output, diff_loss = self.forward(masked_item_seq, item_type_seq, mask_positions_nums=(masked_index, mask_nums), session_id=session_id)
        seq_output = self.forward(masked_item_seq, item_type_seq,
                                  mask_positions_nums=(masked_index, mask_nums), session_id=session_id)
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        test_item_emb = self.item_embedding.weight  # [item_num H]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num]
        targets = (masked_index > 0).float().view(-1)  # [B*mask_len]

        loss = torch.sum(loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) \
               / torch.sum(targets)
        loss = loss
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction['item_id_list']
        type_seq = interaction['item_type_list']
        item_seq_len = torch.count_nonzero(item_seq, 1)
        item_seq, type_seq = self.reconstruct_test_data(item_seq, item_seq_len, type_seq)
        seq_output = self.forward(item_seq, type_seq, is_inference=True)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores

    def customized_sort_predict(self, interaction):
        item_seq = interaction['item_id_list']
        type_seq = interaction['item_type_list']
        truth = interaction['item_id']
        if self.dataset == "ijcai_beh":
            raw_candidates = [73, 3050, 22557, 5950, 4391, 6845, 1800, 2261, 13801, 2953, 4164, 32090, 3333, 44733, 7380, 790, 1845, 2886, 2366, 21161, 6512, 1689, 337, 3963, 3108, 715, 169, 2558, 6623, 888, 6708, 3585, 501, 308, 9884, 1405, 5494, 6609, 7433, 25101, 3580, 145, 3462, 5340, 1131, 6681, 7776, 8678, 52852, 19229, 4160, 33753, 4356, 920, 15312, 43106, 16669, 1850, 2855, 43807, 15, 8719, 89, 3220, 36, 2442, 9299, 8189, 701, 300, 526, 4564, 516, 1184, 178, 2834, 16455, 9392, 22037, 344, 15879, 3374, 2984, 3581, 11479, 6927, 779, 5298, 10195, 39739, 663, 9137, 24722, 7004, 7412, 89534, 2670, 100, 6112, 1355]
        elif self.dataset == "retail_beh":
            raw_candidates = [101, 11, 14, 493, 163, 593, 1464, 12, 297, 123, 754, 790, 243, 250, 508, 673, 1161, 523, 41, 561, 2126, 196, 1499, 1093, 1138, 1197, 745, 1431, 682, 1567, 440, 1604, 145, 1109, 2146, 209, 2360, 426, 1756, 46, 1906, 520, 3956, 447, 1593, 1119, 894, 2561, 381, 939, 213, 1343, 733, 554, 2389, 1191, 1330, 1264, 2466, 2072, 1024, 2015, 739, 144, 1004, 314, 1868, 3276, 1184, 866, 1020, 2940, 5966, 3805, 221, 11333, 5081, 685, 87, 2458, 415, 669, 1336, 3419, 2758, 2300, 1681, 2876, 2612, 2405, 585, 702, 3876, 1416, 466, 7628, 572, 3385, 220, 772]
        elif self.dataset == "tmall_beh":
            raw_candidates = [2544, 7010, 4193, 32270, 22086, 7768, 647, 7968, 26512, 4575, 63971, 2121, 7857, 5134, 416, 1858, 34198, 2146, 778, 12583, 13899, 7652, 4552, 14410, 1272, 21417, 2985, 5358, 36621, 10337, 13065, 1235, 3410, 14180, 5083, 5089, 4240, 10863, 3397, 4818, 58422, 8353, 14315, 14465, 30129, 4752, 5853, 1312, 3890, 6409, 7664, 1025, 16740, 14185, 4535, 670, 17071, 12579, 1469, 853, 775, 12039, 3853, 4307, 5729, 271, 13319, 1548, 449, 2771, 4727, 903, 594, 28184, 126, 27306, 20603, 40630, 907, 5118, 3472, 7012, 10055, 1363, 9086, 5806, 8204, 41711, 10174, 12900, 4435, 35877, 8679, 10369, 2865, 14830, 175, 4434, 11444, 701]
        customized_candidates = list()
        for batch_idx in range(item_seq.shape[0]):
            seen = item_seq[batch_idx].cpu().tolist()
            cands = raw_candidates.copy()
            for i in range(len(cands)):
                if cands[i] in seen:
                    new_cand = random.randint(1, self.n_items)
                    while new_cand in seen:
                        new_cand = random.randint(1, self.n_items)
                    cands[i] = new_cand
            cands.insert(0, truth[batch_idx].item())
            customized_candidates.append(cands)
        candidates = torch.LongTensor(customized_candidates).to(item_seq.device)
        item_seq_len = torch.count_nonzero(item_seq, 1)
        item_seq, type_seq = self.reconstruct_test_data(item_seq, item_seq_len, type_seq)
        seq_output = self.forward(item_seq, type_seq, is_inference=True)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_items_emb = self.item_embedding(candidates)  # delete masked token
        scores = torch.bmm(test_items_emb, seq_output.unsqueeze(-1)).squeeze()  # [B, item_num]
        return scores

    def build_Gs_unique(self, seqs, item_sim, group_len):
        Gs = []
        n_objs = torch.count_nonzero(seqs, dim=1).tolist()
        for batch_idx in range(seqs.shape[0]):
            seq = seqs[batch_idx]
            n_obj = n_objs[batch_idx]
            seq = seq[:n_obj].cuda()
            seq_list = seq.tolist()
            unique = torch.unique(seq)
            unique = unique.tolist()
            n_unique = len(unique)

            multibeh_group = seq.tolist()
            for x in unique:
                multibeh_group.remove(x)
            multibeh_group = list(set(multibeh_group))
            try:
                multibeh_group.remove(self.mask_token)
            except:
                pass

            # l', l'
            seq_item_sim = item_sim[batch_idx][:n_obj, :][:, :n_obj]
            # l', group_len
            if group_len > n_obj:
                metrics, sim_items = torch.topk(seq_item_sim, n_obj, sorted=False)
            else:
                metrics, sim_items = torch.topk(seq_item_sim, group_len, sorted=False)
            # map indices to item tokens
            sim_items = seq[sim_items]
            row_idx, masked_pos = torch.nonzero(sim_items == self.mask_token, as_tuple=True)
            sim_items[row_idx, masked_pos] = seq[row_idx]
            metrics[row_idx, masked_pos] = 1.0
            # print(sim_items.detach().cpu().tolist())
            multibeh_group = seq.tolist()
            for x in unique:
                multibeh_group.remove(x)
            multibeh_group = list(set(multibeh_group))
            try:
                multibeh_group.remove(self.mask_token)
            except:
                pass
            n_edge = n_unique + len(multibeh_group)
            # hyper graph: n_obj, n_edge
            H = torch.zeros((n_obj, n_edge), device=metrics.device)
            normal_item_indexes = torch.nonzero((seq != self.mask_token), as_tuple=True)[0]
            for idx in normal_item_indexes:
                sim_items_i = sim_items[idx].tolist()
                map_f = lambda x: unique.index(x)
                unique_idx = list(map(map_f, sim_items_i))
                H[idx, unique_idx] = metrics[idx]

            for i, item in enumerate(seq_list):
                ego_idx = unique.index(item)
                H[i, ego_idx] = 1.0
                # multi-behavior hyperedge
                if item in multibeh_group:
                    H[i, n_unique + multibeh_group.index(item)] = 1.0
            # print(H.detach().cpu().tolist())
            # W = torch.ones(n_edge, device=H.device)
            # W = torch.diag(W)
            DV = torch.sum(H, dim=1)
            DE = torch.sum(H, dim=0)
            invDE = torch.diag(torch.pow(DE, -1))
            invDV = torch.diag(torch.pow(DV, -1))
            # DV2 = torch.diag(torch.pow(DV, -0.5))
            HT = H.t()
            G = invDV.mm(H).mm(invDE).mm(HT)
            # G = DV2.mm(H).mm(invDE).mm(HT).mm(DV2)
            assert not torch.isnan(G).any()
            Gs.append(G.to(seqs.device))
        Gs_block_diag = torch.block_diag(*Gs)
        return Gs_block_diag


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class GaussianDiffusion(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max, \
                 steps, device, history_num_per_term=10, beta_fixed=True):

        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.history_num_per_term = history_num_per_term
        self.Lt_history = torch.zeros(steps, history_num_per_term, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

        if noise_scale != 0.:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

            self.calculate_for_diffusion()

        super(GaussianDiffusion, self).__init__()

    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
                self.steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(
            self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(
            self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t

    def training_losses(self, model, x_start, reweight=False):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = torch.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / (
                            (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts]))
                weight = torch.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output)) ** 2 / 2.0)
                loss = torch.where((ts == 0), likelihood, mse)
        else:
            weight = torch.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss

        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

        terms["loss"] /= pt
        return terms, model_output

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')

            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt

        elif method == 'uniform':  # uniform sampling
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CapsuleNetwork(torch.nn.Module):
    def __init__(self, dim,
                 seq_len,
                 bilinear_type=2,
                 num_interest=4,
                 hard_readout=True,
                 relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.weights = nn.Parameter(torch.Tensor(1, self.seq_len, self.num_interest * self.dim, self.dim))
        nn.init.normal_(self.weights, std=0.02)
        # self.dense_net = torch.nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU())
        # self.mlp = nn.Linear(self.dim * self.num_interest, self.dim)

        # parameters initialization
        self._init_weights(self)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, seq_out):
        # [N, T, 1, C]
        u = torch.unsqueeze(seq_out, dim=2)
        # [N, T, num_caps * dim_caps]
        item_emb_hat = torch.sum(self.weights[:, :self.seq_len, :, :] * u, dim=3)
        item_emb_hat = torch.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = item_emb_hat.permute([0, 2, 1, 3])

        if self.stop_grad:
            # Returns a new Tensor, requires_grad=False.
            item_emb_hat_iter = item_emb_hat.detach()
        else:
            item_emb_hat_iter = item_emb_hat

        capsule_weight = torch.Tensor(seq_out.size(0), self.num_interest, self.seq_len).cuda()
        capsule_weight = self.truncated_normal_(capsule_weight, std=1.0).requires_grad_(requires_grad=False)

        for i in range(3):
            capsule_softmax_weight = torch.softmax(capsule_weight, dim=1)
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, dim=2)

            if i < 2:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(item_emb_hat_iter, interest_capsule.permute([0, 1, 3, 2]))
                delta_weight = torch.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight

                if i >= 0:
                    item_emb_hat_iter = item_emb_hat.detach()
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.squeeze(interest_capsule, dim=2)
        # interest_capsule = interest_capsule.permute([-1, self.num_interest, self.dim])
        if self.relu_layer:
            interest_capsule = self.dense_net(interest_capsule)

        # interest_capsule = interest_capsule.view([interest_capsule.size(0), -1])
        # interest_capsule = torch.tanh(self.mlp(interest_capsule))

        return interest_capsule

    # def get_readout(self, interest_capsule, item_eb):
    #     atten = torch.matmul(interest_capsule, torch.unsqueeze(item_embd, dim=2))
    #     atten = torch.pow(torch.squeeze(atten, dim=2), 1)
    #     atten = torch.softmax(atten, dim = 1)
    #
    #     self.hard_readout = False
    #     # 第一个问题 hardeadout 怎么算的
    #     if self.hard_readout:
    #         index = (torch.argmax(atten, dim=1) + torch.arange(interest_capsule.size(0)) * self.num_interest).int()
    #         # index = torch.tensor(index, dtype=torch.int64)
    #         input = interest_capsule.view([-1, self.dim])
    #         readout = torch.index_select(input, dim=0, index=index)
    #     else:
    #         readout = torch.matmul(atten.view([interest_capsule.size(0), 1, self.num_interest]), interest_capsule)
    #         readout = readout.view([interest_capsule.size(0), self.dim])
    #     return readout

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor


class FRNet(nn.Module):
    """
    Feature refinement network：
    (1) IEU
    (2) CSGate
    """
    def __init__(self, field_length, embed_dim, weight_type="bit", num_layers=1, att_size=10, mlp_layer=256):
        """
        :param field_length: field_length
        :param embed_dim: embedding dimension
        type: bit or vector
        """
        super(FRNet, self).__init__()
        # IEU_G computes complementary features.
        self.IEU_G = IEU(field_length, embed_dim, weight_type="bit",
                         bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

        # IEU_W computes bit-level or vector-level weights.
        self.IEU_W = IEU(field_length, embed_dim, weight_type=weight_type,
                         bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

    def forward(self, x_embed):
        com_feature = self.IEU_G(x_embed)
        wegiht_matrix = torch.sigmoid(self.IEU_W(x_embed))
        # CSGate
        # x_out = x_embed * wegiht_matrix
        x_out = x_embed * wegiht_matrix + com_feature * (torch.tensor(1.0) - wegiht_matrix)
        return x_out


class IEU(nn.Module):
    """
    Information extraction Unit (IEU) for FRNet
    (1) Self-attention
    (2) DNN
    """
    def __init__(self, field_length, embed_dim, weight_type="bit",
                 bit_layers=1, att_size=10, mlp_layer=256):
        """
        :param field_length:
        :param embed_dim:
        :param type: vector or bit
        :param bit_layers:
        :param att_size:
        :param mlp_layer:
        """
        super(IEU,self).__init__()
        self.input_dim = field_length * embed_dim
        self.weight_type = weight_type

        # Self-attention unit, which is used to capture cross-feature relationships.
        self.vector_info = SelfAttentionIEU(embed_dim=embed_dim, att_size=att_size)

        #  contextual information extractor(CIE), we adopt MLP to encode contextual information.
        mlp_layers = [mlp_layer for _ in range(bit_layers)]
        self.mlps = MultiLayerPerceptronPrelu(self.input_dim, embed_dims=mlp_layers,
                                              output_layer=False)
        self.bit_projection = nn.Linear(mlp_layer, embed_dim)
        self.activation = nn.ReLU()
        # self.activation = nn.PReLU()


    def forward(self,x_emb):
        """
        :param x_emb: B,F,E
        :return: B,F,E (bit-level weights or complementary fetures)
                 or B,F,1 (vector-level weights)
        """

        # （1）self-attetnion unit
        x_vector = self.vector_info(x_emb)  # B,F,E

        # (2) CIE unit
        x_bit = self.mlps(x_emb.view(-1, self.input_dim))
        x_bit = self.bit_projection(x_bit).unsqueeze(1) # B,1,e
        x_bit = self.activation(x_bit)

        # （3）integration unit
        x_out = x_bit * x_vector

        if self.weight_type == "vector":
            # To compute vector-level importance in IEU_W
            x_out = torch.sum(x_out,dim=2,keepdim=True)
            # B,F,1
            return x_out

        return x_out


class SelfAttentionIEU(nn.Module):
    def __init__(self, embed_dim, att_size=20):
        """
        :param embed_dim:
        :param att_size:
        """
        super(SelfAttentionIEU, self).__init__()
        self.embed_dim = embed_dim
        self.trans_Q = nn.Linear(embed_dim,att_size)
        self.trans_K = nn.Linear(embed_dim,att_size)
        self.trans_V = nn.Linear(embed_dim,att_size)
        self.projection = nn.Linear(att_size,embed_dim)
        # self.scale = 1.0/ torch.LongTensor(embed_dim)
        # self.scale = torch.sqrt(1.0 / torch.tensor(embed_dim).float())
        # self.dropout = nn.Dropout(0.5)
        # self.layer_norm = nn.LayerNorm(embed_dim)


    def forward(self,x, scale=None):
        """
        :param x: B,F,E
        :return: B,F,E
        """
        Q = self.trans_Q(x)
        K = self.trans_K(x)
        V = self.trans_V(x)
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # B,F,F
        attention_score = F.softmax(attention, dim=-1)
        context = torch.matmul(attention_score, V)
        # Projection
        context = self.projection(context)
        # context = self.layer_norm(context)
        return context


class MultiLayerPerceptronPrelu(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.PReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        :param x: [B,F*E]
        """
        return self.mlp(x)








