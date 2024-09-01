import torch
from torch import nn
import numpy as np
# from datas.dataset_3d import  *
from torch.nn import functional as F


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)

    
class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc1 = nn.ModuleList([nn.Linear(dim_in, 2 * dim_in) for i in range(k)])
        #self.fc2 = nn.ModuleList([nn.Linear(2 * dim_in, 2 * dim_in) for i in range(k)])
        self.fc3 = nn.ModuleList([nn.Linear(2 * dim_in, dim_out) for i in range(k)])
        #self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
        self.relu =  nn.ReLU()

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = tokens[i].transpose(0,1)
                tokens[i] = self.relu(self.fc1[i](tokens[i][:, 1:, :]))
                tokens[i] = self.fc3[i](tokens[i])
                # tokens[i] = self.relu(self.fc3[i](tokens[i]))
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.relu(self.fc1[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous()))
                tokens[i] = self.fc3[i](tokens[i])
                # tokens[i] = self.relu(self.fc3[i](tokens[i]))
        return tokens
    
class PromptLearner(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 36x36 -> 18x18

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 18x18 -> 9x9

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in * 64, dim_out, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Flatten(),
            nn.Linear(dim_out * 3 * 3, dim_out * 9)
            # nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            # # nn.BatchNorm2d(dim_in * 1024),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2), # 7 * 7
            #
            # nn.Conv2d(dim_in * 1024, dim_out, kernel_size=5, padding=0),
            # nn.BatchNorm2d(dim_out),
            # nn.ReLU(inplace=True),
        )
        self.base_prompts = nn.Parameter(torch.randn((9, dim_out)),requires_grad=True)
        self.dim_out = dim_out



    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        #print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,self.dim_out,9).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)
        return output

# class ProjectionLayer(nn.Module):
#     def __init__(self, visual_hidden_size, gemma_hidden_size):
#         super(ProjectionLayer, self).__init__()
#         self.gemma_proj = nn.Linear(visual_hidden_size, gemma_hidden_size)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         return self.relu(self.gemma_proj(x))

# Define the RMSNorm
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# Define the repeated feedforward block in bind network
class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        # normalize the input
        self.norm = RMSNorm(dim)

        # Define 3 linear projection layers whose parameters are w1, w2 and w3 respectively.
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # cascade linear linears with RMSNorm, SiLU activation functions and residual connections
        x = self.norm(x)
        return x + self.w3(F.silu(self.w1(x)) * self.w2(x))


class ProjectionLayer(nn.Module):
    def __init__(self, image_dim, model_dim):
        super(ProjectionLayer, self).__init__()
        self.image_dim = image_dim  # e.g., 1024, encoded by ImageBind
        self.model_dim = model_dim  # e.g., 4096
        self.ffn_dim = self.model_dim * 4  #

        self.linear_0 = nn.Linear(self.image_dim, self.model_dim)
        self.linear_1 = nn.Linear(self.model_dim, 4*self.model_dim)

        self.feed_forward_1 = FeedForwardBlock(dim=self.model_dim, hidden_dim=self.ffn_dim)
        self.feed_forward_2 = FeedForwardBlock(dim=self.model_dim, hidden_dim=self.ffn_dim)
        self.feed_forward_3 = FeedForwardBlock(dim=self.model_dim, hidden_dim=self.ffn_dim)

        self.relu = nn.ReLU()

    def forward(self, image_feature):
        # image_feature, (1,C1) / (1,image_dim)
        batch_size = image_feature.shape[0]
        # Adopt the linear projection layer at first
        image_feature = self.linear_0(image_feature)  # image_feature, (1, model_dim)

        # Cascade 3 projection blocks
        image_feature = self.feed_forward_1(image_feature)
        image_feature = self.feed_forward_2(image_feature)
        image_feature = self.feed_forward_3(image_feature)
        transformed_image_feature = self.relu(self.linear_1(image_feature))
        #print(transformed_image_feature.shape)
        return transformed_image_feature.view(batch_size,-1,self.model_dim)
