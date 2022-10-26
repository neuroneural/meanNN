import torch
from torch import nn
from torch.utils.data import Dataset



def entropyLoss(X, device):
    d = X.shape[0]
    n = X.shape[1]
    if torch.cuda.is_available():
            eigen = torch.eye(n, n).to(device)
    else:
        eigen = torch.eye(n, n)
    squared_norms = (X**2).sum(0).repeat(n, 1)
    squared_norms_T = squared_norms.T
    X_T = X.T
    arg = squared_norms + squared_norms_T - 2 * torch.mm(X_T, X)
    expression = torch.sum(torch.log(torch.abs(arg)+ eigen + 1e-12))/2
    entropy = -d/(n*(n-1)) * expression
    return entropy


class getdata(Dataset):
    def __init__(self, mix):
        self.mix = mix

    def __getitem__(self, item):
        opt = self.mix[:, item].T
        opt.requires_grad_()
        return opt

    def __len__(self):
        return self.mix.shape[1]


class infomaxICA(nn.Module):

    def __init__(self, n):
        super(infomaxICA, self).__init__()
#         self.W1 = torch.nn.Linear(n, 4, bias=False)
        self.W2 = torch.nn.Linear(n, n, bias=False)
        self.W2_bn = torch.nn.BatchNorm1d(n)
        # with torch.no_grad():
        #     self.W2.weight.copy_(torch.eye(n))
        self.init_weight()

    def weights_init(self, m, layer_type=nn.Linear):
        if isinstance(m, layer_type):
            nn.init.xavier_normal_(m.weight.data)

    def init_weight(self):
        for layer in [nn.Linear]:
            self.apply(lambda x: self.weights_init(x, layer_type=layer))

    def forward(self, input):
        input = self.W2_bn(self.W2(input))
#         input = torch.sigmoid(self.W1(input))
        return torch.sigmoid(input)