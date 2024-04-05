import torch
import torch.nn as nn

def test_capsule(embedd, mask, item_ebd):
    capsule_network = CapsuleNetwork(16, 100, bilinear_type=2, num_interest=4,
                                         hard_readout=True, relu_layer=False)
    user_eb_list = capsule_network(embedd, mask)
    inters_user = capsule_network.get_readout(user_eb_list, item_ebd)

    print(user_eb_list)


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
        self.dense_net = torch.nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU())

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

    def forward(self, seq_out, mask):
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

        capsule_weight = torch.Tensor(seq_out.size(0), self.num_interest, self.seq_len)
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

        return interest_capsule

    def get_readout(self, interest_capsule, item_eb):
        atten = torch.matmul(interest_capsule, torch.unsqueeze(item_embd, dim=2))
        atten = torch.pow(torch.squeeze(atten, dim=2), 1)
        atten = torch.softmax(atten, dim = 1)

        self.hard_readout = False
        # 第一个问题 hardeadout 怎么算的
        if self.hard_readout:
            index = (torch.argmax(atten, dim=1) + torch.arange(interest_capsule.size(0)) * self.num_interest).int()
            # index = torch.tensor(index, dtype=torch.int64)
            input = interest_capsule.view([-1, self.dim])
            readout = torch.index_select(input, dim=0, index=index)
        else:
            readout = torch.matmul(atten.view([interest_capsule.size(0), 1, self.num_interest]), interest_capsule)
            readout = readout.view([interest_capsule.size(0), self.dim])
        return readout

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

def readout_select(input1, input2, itemembd):
    input = torch.stack([input1, input2], dim=1)
    atten = torch.einsum("abc, ac -> ab", input, itemembd)
    print(atten)
    index = (torch.argmax(atten, dim=1) + torch.arange(0, input1.size(0)) * 2).int()
    # index = torch.tensor(index, dtype=torch.int64)
    input = input.view([-1, 16])
    readout = torch.index_select(input, dim=0, index=index)
    return readout




# if __name__ == '__main__':
#      embedd = torch.randn([256, 100, 16])
#      mask = torch.randn([256, 100])
#      item_embd = torch.randn([256, 16])
#      test_capsule(embedd, mask, item_embd)

#      # read1 = torch.randn([5, 16])
#      # print(read1)
#      # read2 = torch.randn([5, 16])
#      # item = torch.randn([5, 16])
#      # res = readout_select(read1, read2, item)
#      # print(res)

#      # a = torch.range(0 ,5)
#      # b = torch.arange(5)
#      # print(a)
#      # print(b)



# multi-relation transformer
# num_type = 4
# hidden_size = 64
# sc = 8
# W_relation = {}
# M_relation = {}
# a = torch.randn([10, hidden_size, hidden_size])
# nn.init.normal_(a, std=0.02)
# W_relation["00"] = a[0]
# W_relation["01"] = a[1]
# W_relation["10"] = a[1]
# W_relation["02"] = a[2]
# W_relation["20"] = a[2]
# W_relation["03"] = a[3]
# W_relation["30"] = a[3]
# W_relation["11"] = a[4]
# W_relation["12"] = a[5]
# W_relation["21"] = a[5]
# W_relation["13"] = a[6]
# W_relation["31"] = a[6]
# W_relation["22"] = a[7]
# W_relation["23"] = a[8]
# W_relation["32"] = a[8]
# W_relation["33"] = a[9]
#
# b = torch.randn([10, hidden_size, hidden_size])
# nn.init.normal_(b, std=0.02)
# M_relation["00"] = b[0]
# M_relation["01"] = b[1]
# M_relation["10"] = b[1]
# M_relation["02"] = b[2]
# M_relation["20"] = b[2]
# M_relation["03"] = b[3]
# M_relation["30"] = b[3]
# M_relation["11"] = b[4]
# M_relation["12"] = b[5]
# M_relation["21"] = b[5]
# M_relation["13"] = b[6]
# M_relation["31"] = b[6]
# M_relation["22"] = b[7]
# M_relation["23"] = b[8]
# M_relation["32"] = b[8]
# M_relation["33"] = b[9]
#
# W_Q = torch.randn([hidden_size , hidden_size])
# W_K = torch.randn([hidden_size , hidden_size])
# W_V = torch.randn([hidden_size , hidden_size])
# item = torch.randn([3, hidden_size])
# type = torch.tensor([0, 1, 2])
#
# query = torch.matmul(item, W_Q)
# key = torch.matmul(item, W_K)
# value = torch.matmul(item, W_V)
#
# item_list = []
# for uid in range(item.size(0)):
#     value_u = []
#     attn = []
#     for tid in range(item.size(0)):
#         index_u = "%d%d"%(type[uid],type[tid])
#         query_u = query[uid]
#         query_u = torch.matmul(query_u, W_relation[index_u])
#         key_u = key[tid]
#         attn_u = torch.matmul(query_u, key_u) / sc
#         attn.append(attn_u)
#         value_tp = torch.matmul(M_relation[index_u], value[tid])
#         value_u.append(value_tp)
#     attn = torch.softmax(torch.tensor(attn), dim=0)
#     item_list.append(torch.matmul(torch.stack(value_u, dim=1), attn))
# item_raw = torch.stack(item_list, dim=0)
# print(item.size())

# selecet_interest_fun

# x = torch.randn((3,4,2))
# # print(x)
# # mask = (x == x.max(dim=2, keepdim=True)[0]).to(dtype=torch.int32)
# # print(mask)
# # result = torch.mul(mask, x)
# # print(result)

x = torch.randn((2,2,2))
print(x)
y = torch.randn((2, 1)).unsqueeze(dim = 1)
print(y)
print(y.size())
y = y.repeat(1,2,1)
print(y)
print(y.size())
res = torch.cat([x,y], dim=2)
print(res)


