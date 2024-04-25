import torch
import torch.nn as nn
import math

class DUDCLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, eps=1e-5):
        super(DUDCLoss, self).__init__()
        self.eps = eps
        self.alpha = None
        self.out1=self.out2=self.loss_multi=self.loss_sample=self.new1=self.new2=self.loss_single=None

    def forward(self, out1, out2, target,alpha):
        """"
        Parameters
        ----------
        out1: input probability from one stream
        out2: input probability from other stream
        """

        self.target = target
        self.alpha=alpha

        # Calculating Probabilities on single
        self.out1 = out1
        self.out2 = out2
        self.loss_sample = []
        for i in range(self.out1.size(0)):
            self.new1 = multi_sharp(self.out1[i], self.target[i]).cuda()
            self.new2 = multi_sharp(self.out2[i], self.target[i]).cuda()
            self.loss_sample.append(cross(self.new1, self.new2, self.eps) + cross(self.new2, self.new1, self.eps))
        self.loss_sample = torch.Tensor(self.loss_sample)
        self.loss_single = self.loss_sample.mean()

        # Calculating Probabilities on multi
        self.out1 = nn.functional.sigmoid(out1)
        self.out2 = nn.functional.sigmoid(out2)
        self.loss_multi = cross(self.out1, self.out2, self.eps) + cross(self.out2, self.out1, self.eps)


        loss = self.loss_multi * self.alpha + (1-self.alpha)*self.loss_single

        return loss

class ISDLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, tau=0.005, eps=1e-5):
        super(ISDLoss, self).__init__()
        self.tau = tau
        self.eps = eps
        self.f1=self.f2=self.length_b=self.mean_f1=self.mean_f2=self.length=self.diag=self.new1=self.new2=self.out1=self.out2=None

    def forward(self, feat1, feat2):
        """"
        Parameters
        ----------
        feat1: input features from one stream
        feat2: input features from other stream
        """

        self.f1 = feat1
        self.f2 = feat2

        self.length_b = feat1.size(0)  # batch_size

        self.mean_f1 = torch.reshape(self.f1, (self.length_b, -1))
        self.mean_f2 = torch.reshape(self.f2, (self.length_b, -1))
        self.mean_f1 = nn.functional.normalize(self.mean_f1, 2, dim=1)
        self.mean_f2 = nn.functional.normalize(self.mean_f2, 2, dim=1)

        self.length = self.mean_f1.size(0)
        self.diag = torch.eye(self.length).cuda()
        self.new1 = torch.mm(self.mean_f1, self.mean_f1.t()) / self.tau
        self.new2 = torch.mm(self.mean_f2, self.mean_f2.t()) / self.tau

        self.new1 = self.new1 - self.new1 * self.diag
        self.new2 = self.new2 - self.new2 * self.diag

        self.out1 = self.new1.flatten()[:-1].view(self.length - 1, self.length + 1)[:, 1:].flatten().view(self.length, self.length - 1)  # B*(B-1)
        self.out2 = self.new2.flatten()[:-1].view(self.length - 1, self.length + 1)[:, 1:].flatten().view(self.length, self.length - 1)  # B*(B-1)

        self.out1 = nn.functional.softmax(self.out1)
        self.out2 = nn.functional.softmax(self.out2)

        loss = KL(self.out1, self.out2, self.eps) + KL(self.out2, self.out1, self.eps)
        return loss

def EH(probs, eps):

    ent = - (probs * (probs + eps).log()).sum(dim=1)
    mean = ent.mean()
    torch.distributed.all_reduce(mean)
    return mean

def KL(out1,out2,eps):
    kl = (out1 * (out1 + eps).log() - out1 * (out2 + eps).log()).sum(dim=1)
    kl = kl.mean()
    torch.distributed.all_reduce(kl)
    return kl

def cross(out1,out2,eps):
    loss = -(out1 * (out2 + eps).log()).sum(dim=1)
    loss = loss.mean()
    torch.distributed.all_reduce(loss)
    return loss

def multi_sharp(out,target):
    non = torch.nonzero(target)  # index
    count = len(non)
    non_label_logit = torch.zeros((len(target)-count+1))
    count_num = 0
    for i in range(len(target)):
        if target[i] == 0:
            non_label_logit[count_num] = out[i]
            count_num = count_num + 1
    sharp_mar = non_label_logit.repeat(count,1)
    for i in range(count):
        sharp_mar[i,-1]=out[int(non[i])]
    sharp_mar = nn.functional.softmax(sharp_mar)

    return sharp_mar
