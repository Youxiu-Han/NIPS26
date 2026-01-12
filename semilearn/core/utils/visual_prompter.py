import torch
import torch.nn as nn
import numpy as np
import random

import random

class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.v_prompt_size
        image_size = args.img_size

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size # 224
        self.psize = args.prompt_size # 30
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt

# class RandomMultiPatchPrompter(nn.Module):
#     def __init__(self, args):
#         super(RandomMultiPatchPrompter, self).__init__()
#         self.isize = args.img_size # 224 * 224
#         self.psize = args.v_prompt_size # if 32 * 32, the num of pathch is 49
#         self.num_patches = (self.isize // self.psize) ** 2
#         self.learnable_ratio = int(self.num_patches * args.learnable_ratio)
#         self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))
#         self.all_coordinates = [(x, y) for x in range(self.isize // self.psize) for y in range(self.isize // self.psize)]
#         self.selected_indices = np.random.choice(len(self.all_coordinates), size=self.learnable_ratio, replace=False)
#         self.selected_coordinates = [self.all_coordinates[i] for i in self.selected_indices]
#     def forward(self, x):
#         prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
#         for x_, y_ in self.selected_coordinates:
#             prompt[:, :, x_ * self.psize:(x_ + 1) * self.psize, y_ * self.psize:(y_ + 1) * self.psize] += self.patch.cuda()

#         return x + prompt

class RandomMultiPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomMultiPatchPrompter, self).__init__()
        self.isize = args.img_size # 224 * 224
        self.psize = args.v_prompt_size # if 32 * 32, the num of pathch is 49
        self.num_patches = (self.isize // self.psize) ** 2
        self.learnable_ratio = int(self.num_patches * args.learnable_ratio)
        self.all_coordinates = [(x, y) for x in range(self.isize) for y in range(self.isize)]
        self.selected_indices = np.random.choice(len(self.all_coordinates), size=self.learnable_ratio, replace=False)
        self.selected_coordinates = [self.all_coordinates[i] for i in self.selected_indices]
        print('The number of pixels is :', len(self.selected_coordinates))
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))
    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        for x_, y_ in self.selected_coordinates:
            prompt[:, :, x_ * self.psize:(x_ + 1) * self.psize, y_ * self.psize:(y_ + 1) * self.psize] += self.patch.cuda()

        return x + prompt



class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt


def padding(args):
    return PadPrompter(args)

def fixed_patch(args):
    return FixedPatchPrompter(args)

def random_multi_patches(args):
    return RandomMultiPatchPrompter(args)
def random_patch(args):
    return RandomPatchPrompter(args)


