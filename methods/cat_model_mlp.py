import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CATModel(torch.nn.Module):

    def __init__(self, classes_per_task, n_head, size,
                 nlayers=2, nhid=2000, pdrop1=0.2, pdrop2=0.5, smax=400):
        super().__init__()

        ncha = 3
        num_tasks = len(classes_per_task)
        self.classes_per_task = classes_per_task

        self.nlayers = nlayers
        self.relu = torch.nn.ReLU()

        self.drop1 = torch.nn.Dropout(pdrop1)
        self.drop2 = torch.nn.Dropout(pdrop2)
        self.nhid = nhid
        self.gate = torch.nn.Sigmoid()

        self.mcl = MainContinualLearning(nhid, ncha, size, num_tasks, classes_per_task)
        self.transfer = TransferLayer(classes_per_task, nhid, ncha, size)
        self.kt = KnowledgeTransfer(nhid, ncha, size, classes_per_task, n_head)
        self.smax = smax

    # progressive style
    def forward(self, t, x, s=1, phase=None,
                pre_mask=None, pre_task=None,
                similarity=None, history_mask_pre=None, check_federated=None):

        if 'mcl' in phase:
            max_masks = self.mask(t, s=s)
            gfc1, gfc2 = max_masks

            # Gated
            h = self.drop1(x.view(x.size(0), -1))

            h = self.drop2(self.relu(self.mcl.fc1(h)))
            h = h*gfc1.expand_as(h)

            h = self.drop2(self.relu(self.mcl.fc2(h)))
            h = h*gfc2.expand_as(h)

            y = []
            for t in range(len(self.classes_per_task)):
                y.append(self.mcl.mask_last[t](h))
            return y, max_masks, None

        elif 'mcl' in phase and 'multi-loss-joint-Tsim' in self.args.loss_type:
            max_masks = self.mask(t, s=s)
            gfc1, gfc2 = max_masks

            # Gated
            h = self.drop1(x.view(x.size(0), -1))

            h = self.drop2(self.relu(self.mcl.fc1(h)))
            h = h*gfc1.expand_as(h)

            h = self.drop2(self.relu(self.mcl.fc2(h)))
            h = h*gfc2.expand_as(h)

            pre_models = []

            pre_ts = []
            for pre_t in range(t):
                if self.training == True and similarity[pre_t]:
                    continue
                elif self.training == False and check_federated.check_t(pre_t) == False:
                    # Cautions: testing needs to be careful
                    continue

                pre_gfc1, pre_gfc2 = self.mask(torch.autograd.Variable(torch.LongTensor([pre_t]).cuda(), volatile=False), s=self.smax)
                pre_gfc1 = pre_gfc1.data.clone()
                pre_gfc2 = pre_gfc2.data.clone()

                pre_h = self.drop1(x.view(x.size(0), -1))

                pre_h = self.drop2(self.relu(self.mcl.fc1(pre_h)))  # fc1 is changing
                pre_h = pre_h*pre_gfc1.expand_as(pre_h)

                pre_h = self.drop2(self.relu(self.mcl.fc2(pre_h)))
                pre_h = pre_h*pre_gfc2.expand_as(pre_h)

                pre_models.append(pre_h.clone())
                pre_ts.append(pre_t)
            # Tsim: model for each Tsim

            if len(pre_models) > 1:
                task_models = torch.stack(pre_models)
                task_models = task_models.permute(1, 0, 2)

                query = torch.unsqueeze(self.relu(self.kt.q1(t)).expand(task_models.size(0), -1), 1)  # hard to train

                h_attn, _ = self.kt.encoder(task_models, query)

                y_attn = []
                y = []

                for t in range(len(self.classes_per_task)):
                    y_attn.append(self.mcl.att_last[t](h_attn))
                    y.append(self.mcl.mask_last[t](h))

                return y, max_masks, y_attn

            else:
                if 'no-isolate' in self.args.loss_type:

                    y = []
                    for t in range(len(self.classes_per_task)):
                        y.append(self.mcl.mask_last[t](h))

                    return y, max_masks, None

                else:
                    # only care about myself, even in forward pass
                    gfc1, gfc2 = self.Tsim_mask(t, history_mask_pre=history_mask_pre, similarity=similarity)

                    h_attn = self.drop1(x.view(x.size(0), -1))
                    h_attn = self.drop2(self.relu(self.mcl.fc1(h_attn)))
                    h_attn = h_attn*gfc1.expand_as(h_attn)

                    h_attn = self.drop2(self.relu(self.mcl.fc2(h_attn)))
                    h_attn = h_attn*gfc2.expand_as(h_attn)

                    y_attn = []
                    y = []
                    for t in range(len(self.classes_per_task)):
                        y_attn.append(self.mcl.att_last[t](h_attn))
                        y.append(self.mcl.mask_last[t](h))

                    return y, max_masks, y_attn

        elif phase == 'transfer':
            gfc1, gfc2 = pre_mask

            # source domain data
            h = self.drop1(x.view(x.size(0), -1))
            h = self.drop2(self.relu(self.mcl.fc1(h)))
            h = h*gfc1.expand_as(h)
            h = self.drop2(self.relu(self.mcl.fc2(h)))
            h = h*gfc2.expand_as(h)

            y = []
            for t in range(len(self.classes_per_task)):
                y.append(self.transfer.transfer[pre_task][t](self.mcl.mask_last[pre_task](h)))
            return y

        elif phase == 'reference':
            gfc1, gfc2 = pre_mask

            # no source domain
            h = self.drop1(x.view(x.size(0), -1))
            h = self.drop2(self.relu(self.transfer.fc1(h)))
            h = h*gfc1.expand_as(h)
            h = self.drop2(self.relu(self.transfer.fc2(h)))
            h = h*gfc2.expand_as(h)

            y = []
            for t in range(len(self.classes_per_task)):
                y.append(self.transfer.transfer[pre_task][t](self.transfer.last[pre_task](h)))
            return y

    def mask(self, t, s=1, phase=None):
        # used by training

        gfc1 = self.gate(s*self.mcl.efc1(t))
        gfc2 = self.gate(s*self.mcl.efc2(t))
        return [gfc1, gfc2]

    def Tsim_mask(self, t, history_mask_pre=None, similarity=None, phase=None):
        # find the distinct mask, used by block the backward pass

        # want aggregate Tsim
        if phase is None:
           # Tsim mask
            Tsim_gfc1 = torch.ones_like(self.gate(0*self.mcl.efc1(t)))
            Tsim_gfc2 = torch.ones_like(self.gate(0*self.mcl.efc2(t)))

        for history_t in range(t):
            if history_t == 0:
                Tsim_gfc1_index = history_mask_pre[history_t][0].round().nonzero()
                Tsim_gfc2_index = history_mask_pre[history_t][1].round().nonzero()
            else:
                Tsim_gfc1_index = (history_mask_pre[history_t][0] - history_mask_pre[history_t-1][0]).round().nonzero()
                Tsim_gfc2_index = (history_mask_pre[history_t][1] - history_mask_pre[history_t-1][1]).round().nonzero()
            if similarity[history_t] == 0:
                Tsim_gfc1[Tsim_gfc1_index[:, 0], Tsim_gfc1_index[:, 1]] = 0
                Tsim_gfc2[Tsim_gfc2_index[:, 0], Tsim_gfc2_index[:, 1]] = 0

        return [Tsim_gfc1, Tsim_gfc2]

    def get_view_for(self, n, masks):
        gfc1, gfc2 = masks

        if n == 'mcl.fc1.weight':
            return gfc1.data.view(-1, 1).expand_as(self.mcl.fc1.weight)
        elif n == 'mcl.fc1.bias':
            return gfc1.data.view(-1)
        elif n == 'mcl.fc2.weight':
            post = gfc2.data.view(-1, 1).expand_as(self.mcl.fc2.weight)
            pre = gfc1.data.view(1, -1).expand_as(self.mcl.fc2.weight)
            return torch.min(post, pre)
        elif n == 'mcl.fc2.bias':
            return gfc2.data.view(-1)
        return None

    def pre_model_generator(self, t, similarity, x):
        pre_models = []
        for pre_t in range(t):
            if similarity[pre_t] == 0:
                continue
            pre_gfc1, pre_gfc2 = self.mask(torch.autograd.Variable(torch.LongTensor([pre_t]).cuda(), volatile=False), s=self.smax)

            pre_h = self.drop1(x.view(x.size(0), -1))

            pre_h = self.drop2(self.relu(self.mcl.fc1(pre_h)))  # fc1 is changing
            pre_h = pre_h*pre_gfc1.expand_as(pre_h)

            pre_h = self.drop2(self.relu(self.mcl.fc2(pre_h)))
            pre_h = pre_h*pre_gfc2.expand_as(pre_h)

            pre_models.append(pre_h.clone())
        return pre_models


class MainContinualLearning(torch.nn.Module):

    def __init__(self, nhid, ncha, size, num_tasks, classes_per_task):

        super(MainContinualLearning, self).__init__()

        self.efc1 = torch.nn.Embedding(num_tasks, nhid)
        self.efc2 = torch.nn.Embedding(num_tasks, nhid)

        self.fc1 = torch.nn.Linear(ncha*size*size, nhid)
        self.fc2 = torch.nn.Linear(nhid, nhid)

        self.mask_last = torch.nn.ModuleList()
        self.att_last = torch.nn.ModuleList()

        for num_classes in classes_per_task:
            self.mask_last.append(torch.nn.Linear(nhid, num_classes))
            self.att_last.append(torch.nn.Linear(nhid, num_classes))


class TransferLayer(torch.nn.Module):

    def __init__(self, classes_per_task, nhid, ncha, size):

        super(TransferLayer, self).__init__()

        self.fc1 = torch.nn.Linear(ncha*size*size, nhid)
        self.fc2 = torch.nn.Linear(nhid, nhid)

        self.fusion = torch.nn.Linear(nhid*2, nhid)

        self.last = torch.nn.ModuleList()
        self.last_fusion = torch.nn.ModuleList()

        for num_classes in classes_per_task:
            self.last.append(torch.nn.Linear(nhid, num_classes))
            self.last_fusion.append(torch.nn.Linear(nhid*2, num_classes))

        self.transfer = torch.nn.ModuleList()
        for from_n in classes_per_task:
            self.transfer_to_n = torch.nn.ModuleList()
            for to_n in classes_per_task:
                self.transfer_to_n.append(torch.nn.Linear(from_n, to_n))
            self.transfer.append(self.transfer_to_n)


class KnowledgeTransfer(torch.nn.Module):

    def __init__(self, nhid, ncha, size, classes_per_task, n_head):

        super(KnowledgeTransfer, self).__init__()

        self.last = torch.nn.ModuleList()
        for num_classes in classes_per_task:
            self.last.append(torch.nn.Linear(nhid, num_classes))

        self.efc1 = torch.nn.Embedding(len(classes_per_task), nhid)
        self.efc2 = torch.nn.Embedding(len(classes_per_task), nhid)

        # self-attention ==============
        self.q1 = torch.nn.Embedding(len(classes_per_task), nhid)
        self.encoder = EncoderLayer(n_head, nhid, nhid, int(nhid/n_head), int(nhid/n_head))
        # n_head, d_model, d_k, d_v


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_head, d_model, d_inner, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.position_enc = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, enc_q=None, ranking=None):
        # TODO: Positional/ranking embedding

        if enc_q is None:
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
            enc_output = self.pos_ffn(enc_output)

        else:
            enc_output, enc_slf_attn = self.slf_attn(enc_q, enc_input, enc_input)
            enc_output = self.pos_ffn(enc_output)

        enc_output = self.layer_norm(enc_output)

        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)  # sqrt d_k

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = torch.squeeze(q, 1)
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)

        if len_q == 1:
            q = q.transpose(1, 2).contiguous().view(sz_b, -1)
        else:
            q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=40):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, enc_input, ranking):
        return enc_input + self.pos_table[:, ranking].clone().detach()
