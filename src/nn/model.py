import torch
import torch.nn as nn
from torch.autograd import Variable


class TreeLSTMCell(nn.Module):
    def __init__(self, dim):
        super(TreeLSTMCell, self).__init__()
        self.dim = dim
        self.w_for = Variable(torch.randn(self.dim, 1))
        self.u_for = Variable(torch.randn(self.dim, 1))
        self.b_for = Variable(torch.randn(self.dim, 1))
        self.w_in = Variable(torch.randn(self.dim, 1))
        self.u_in = Variable(torch.randn(self.dim, 1))
        self.b_in = Variable(torch.randn(self.dim, 1))
        self.w_ce = Variable(torch.randn(self.dim, 1))
        self.u_ce = Variable(torch.randn(self.dim, 1))
        self.b_ce = Variable(torch.randn(self.dim, 1))
        self.w_out = Variable(torch.randn(self.dim, 1))
        self.u_out = Variable(torch.randn(self.dim, 1))
        self.b_out = Variable(torch.randn(self.dim, 1))

    # 读入AST的索引字典和特征向量字典
    def init_hash_tree(self, tree_dict, tree_vec):
        self.tree_dict = tree_dict
        self.tree_vec = tree_vec
        print(len(tree_dict), len(tree_vec))

    #  当前节点的计算
    def node_forward(self, feature_vec, child_h, child_c):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        child_f_sum = torch.FloatTensor(torch.zeros(self.dim,1))
        for i in range(list(child_c.size())[0]):
            child_f_sum = child_f_sum + torch.sigmoid(self.w_for.mul(feature_vec) + self.u_for.mul(child_h[i]) + self.b_for)
        it = torch.sigmoid(self.w_in.mul(feature_vec) + self.u_in.mul(child_h_sum) + self.b_in)
        ct = torch.tanh(self.w_ce.mul(feature_vec) + self.u_ce.mul(child_h_sum) + self.b_ce)
        ct = it.mul(ct) + child_f_sum
        ot = torch.sigmoid(self.w_out.mul(feature_vec) + self.u_out.mul(child_h_sum) + self.b_out)
        ht = ot.mul(torch.tanh(ct))
        return ht, ct

    def forward(self, root):
        stacked_h, stacked_c = None, None
        try:
            feature_vec = self.tree_vec[root]
        except:
            feature_vec = torch.FloatTensor(torch.zeros(self.dim, 1))

        try:
            for i in range(len(self.tree_dict[root])):
                if i == 0:
                    stacked_h, stacked_c = self.forward(self.tree_dict[root][i])
                    continue
                h, c = self.forward(self.tree_dict[root][i])
                stacked_h = torch.stack([stacked_h, h], dim=0)
                stacked_c = torch.stack([stacked_c, c], dim=0)

            if len(self.tree_dict[root]) == 0:
                stacked_h = torch.FloatTensor(torch.zeros(1, self.dim, 1))
                stacked_c = torch.FloatTensor(torch.zeros(1, self.dim, 1))

        except:
            stacked_h = torch.FloatTensor(torch.zeros(1, self.dim, 1))
            stacked_c = torch.FloatTensor(torch.zeros(1, self.dim, 1))

        return self.node_forward(feature_vec, stacked_h, stacked_c)




class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, out_dim):
        super(Classifier, self).__init__()
        self.cls_input_layer = nn.Linear(in_dim, hidden_dim_1)
        self.cls_hideen_layer = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.cls_output_layer =  nn.Linear(hidden_dim_2, out_dim)

    def forward(self, input):
        input = input.view(1, list(input.size())[0])
        x = torch.sigmoid(self.cls_input_layer(input))
        x = torch.sigmoid(self.cls_hideen_layer(x))
        output = torch.sigmoid(self.cls_output_layer(x))
        return output




class TreeLSTM(nn.Module):
    def __init__(self, embed_dim, cls_in_dim, cls_hidden_dim_1, cls_hidden_dim_2, cls_out_dim):
        super(TreeLSTM, self).__init__()
        self.lstm_cell = TreeLSTMCell(embed_dim)
        self.classifier = Classifier(cls_in_dim, cls_hidden_dim_1, cls_hidden_dim_2, cls_out_dim)

    def forward(self, tree_dict, tree_vec, root):
        print('hahha')
        self.lstm_cell.init_hash_tree(tree_dict, tree_vec)
        lstm_output_h, lstm_output_c = self.lstm_cell(root)
        cls_output = self.classifier(lstm_output_h[0])
        return cls_output