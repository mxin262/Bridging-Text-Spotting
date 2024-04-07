import torch
import torch.nn as nn


class SeqSimCLRLoss(nn.Module):
    def __init__(self, batch_size, temperature, num_windows=5, patch_shape=None):
        super(SeqSimCLRLoss, self).__init__()
        self.batch_size = batch_size * num_windows
        self.temperature = temperature
        self.num_windows = num_windows
        self.patch_shape = patch_shape

        # self.mask = self.mask_correlated_samples(self.batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        # self.adaptive_avgpool = nn.AdaptiveAvgPool1d(num_windows)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, num_windows))

        # timer to show the similarity of pos and neg pairs every 1000.
        self.timer = 0

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """

        # sequential features to fixed windows
        B, _, C = z_i.shape
        # z_i = z_i.reshape(B, self.patch_shape[0], self.patch_shape[1], C).permute(0, 3, 1, 2)
        # z_i = self.adaptive_avgpool(z_i).permute(0, 2, 3, 1).view(-1, C) # [B*num_windows, C]

        # z_j = z_j.reshape(B, self.patch_shape[0], self.patch_shape[1], C).permute(0, 3, 1, 2)
        # z_j = self.adaptive_avgpool(z_j).permute(0, 2, 3, 1).view(-1, C) # [B*num_windows, C]

        z_i = z_i.reshape(B, 1, self.patch_shape[1], C).permute(0, 3, 1, 2)
        z_i = self.adaptive_avgpool(z_i).permute(0, 2, 3, 1).view(-1, C) # [B*num_windows, C]

        z_j = z_j.reshape(B, 1, self.patch_shape[1], C).permute(0, 3, 1, 2)
        z_j = self.adaptive_avgpool(z_j).permute(0, 2, 3, 1).view(-1, C) # [B*num_windows, C]

        batch_size = z_i.size(0)
        mask = self.mask_correlated_samples(batch_size)

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # print(sim_i_j[0], sim[0,:5])
        # print(sim_j_i[:5])

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        if self.timer % 1000 == 0:
            print((positive_samples * self.temperature).mean().item(), 
            (negative_samples * self.temperature).mean().item())
        self.timer += 1

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss