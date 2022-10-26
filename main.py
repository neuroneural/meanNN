import torch
import numpy as np
import cv2
import ipdb
import scipy

from torch import nn
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from numpy import dot
from numpy.linalg import matrix_rank, inv
from numpy.random import permutation
from scipy.linalg import eigh
from scipy.linalg import norm as mnorm

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
# from process import whitening
from process import pca_whiten

resize = 128
num_epoch = 10
learning_rate = 1e-3

grf = Image.open('./bear.png')
fox = Image.open('./flower.png')
grf = np.array(ImageOps.grayscale(grf))
fox = np.array(ImageOps.grayscale(fox))

grf = cv2.resize(grf,(resize,resize))
fox = cv2.resize(fox,(resize,resize))
img_grf = torch.from_numpy(grf).float()
img_fox = torch.from_numpy(fox).float()

A = torch.rand(2,2, dtype=torch.float32)
XX= A@np.c_[img_grf.flatten(), img_fox.flatten()].T # the mixed source

print('the mixing matrix:', A)

#---whitening----

batchSize = resize**2//2
X = pca_whiten(x2d = XX, n_comp=2)
X = X[0]#scipy.stats.zscore(X[0])
X = torch.from_numpy(X).type(torch.float32)

#-----------------------------------------------------------loss---------------

def entropyLoss(X):
    entropy = torch.tensor([0]).type(torch.float32)
    for batch in X:
        d = batch.shape[0]
        n = batch.shape[1]
        squared_norms = (batch**2).sum(0).repeat(n, 1)
        squared_norms_T = squared_norms.T
        batch_T = batch.T
        arg = squared_norms + squared_norms_T - 2 * torch.mm(batch_T, batch)
        expression = torch.sum(torch.log(torch.abs(arg)+ torch.eye(n, n) + 1e-12))/2
        entropy += -d/(n*(n-1)) * expression
    return entropy/X.size()[0]
##---------------------------------
##---------------------------------------------dataset--------------------------------------------------------------------------
class getdata(Dataset):
    def __init__(self, mix, length):
        self.mix = mix
        # self.n = mix.shape[0]
        self.d = mix.shape[1]
        self.length = length

    def __getitem__(self, item):
        batch_id = item%self.length
        opt = self.mix[:, batch_id*self.length:(batch_id+1)*self.length].T
        opt.requires_grad_()
        # num_batch = self.d / self.length
        return opt

    def __len__(self):
        # print(type(self.d/self.length))
        return self.d//self.length
##--------------------------------------------------model-------------------------------------------------------------------------
class infomaxICA(nn.Module):

    def __init__(self, n):
        super(infomaxICA, self).__init__()
#         self.W1 = torch.nn.Linear(n, n, bias=False)
        self.W2 = torch.nn.Linear(n, n, bias=False)
        self.init_weight()

    def weights_init(self, m, layer_type=nn.Linear):
        if isinstance(m, layer_type):
            nn.init.xavier_normal_(m.weight.data)

    def init_weight(self):
        for layer in [nn.Linear]:
            self.apply(lambda x: self.weights_init(x, layer_type=layer))

    def forward(self, input):
#         input = self.W1(input)
        return torch.sigmoid(self.W2(input))
##-------------------------------------training---------------------------------

dataset = getdata(X, 1024)
sampler = RandomSampler(dataset)
loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=4)
model = infomaxICA(2)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate,
                           eps=0.99)
scheduler = CosineAnnealingLR(optimizer, T_max = num_epoch)


loss_tracker = []

for epoch in range(num_epoch):
    for step, ipt in enumerate(loader):
        model.zero_grad()
        opt = model.forward(ipt)
        loss = entropyLoss(opt.permute(0, 2, 1))
        loss.backward()
        optimizer.step()

    loss_tracker.append(loss.detach().numpy())
    print(epoch, loss.detach().numpy())
    if loss.detach().numpy() < -15.5: 
        break
    # scheduler.step()
##--------------------------------------------------------------testing--------------------------


plt.figure(figsize=(10, 5))
plt.plot(loss_tracker)
plt.savefig('loss_curve.png')
plt.show()

plt.figure(figsize=(14,5))
data = np.random.rand(resize**2, 2)
plt.subplot(1,2,1)
plt.plot(data[:, 0], data[:, 1], '.', ms=0.5)
plt.title('Random uniform data\n Entropy loss '+str(entropyLoss(torch.from_numpy(data).float().T.unsqueeze(0)).detach().numpy()));

plt.subplot(1,2,2)
with torch.no_grad():
    data = model.forward(X.T).detach().numpy() # B.detach().numpy()#
plt.plot(data[:, 0], data[:, 1], '.', ms=0.5)
plt.title('Reconstructed sources\n Entropy loss '+str(entropyLoss(torch.from_numpy(data).float().T.unsqueeze(0)).detach().numpy()));
plt.savefig('source_plot.png')
plt.show()


plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
i0 = data[:,0]
i0_s = np.reshape(i0, (resize,resize))
plt.imshow(i0_s, cmap='gray')

plt.subplot(1,2,2)
i1 = data[:,1]
i1_s = np.reshape(i1, (resize,resize))
plt.imshow(i1_s, cmap='gray')
plt.savefig('generated_image.png')
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
j0 = X[0]
j0_s = np.reshape(j0, (resize,resize))
plt.imshow(j0_s, cmap='gray')

plt.subplot(1,2,2)
j1 = X[1]
j1_s = np.reshape(j1, (resize,resize))
plt.imshow(j1_s, cmap='gray')
plt.savefig('mixture.png')
plt.show()

