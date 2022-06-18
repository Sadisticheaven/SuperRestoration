# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import torch
from SRFlowNet_arch import SRFlowNet
from base_model import BaseModel

class SRFlowModel(BaseModel):
    def __init__(self, opt, step=0):
        super(SRFlowModel, self).__init__(opt)
        self.opt = opt

        self.heats = opt['val']['heats']
        self.n_sample = opt['val']['n_sample']
        # define network and load pretrained models
        opt_net = opt['network_G']
        self.netG = SRFlowNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                              scale=opt['scale'], K=opt_net['flow']['K'], opt=opt, step=step).to(self.device)

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses)[0]

    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None):
        self.netG.eval()

        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z

        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses)
        self.netG.train()
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        if seed: torch.manual_seed(seed)
        C = self.netG.flowUpsamplerNet.C
        H = int(self.opt['scale'] * lr_shape[2] // self.netG.flowUpsamplerNet.scaleH)
        W = int(self.opt['scale'] * lr_shape[3] // self.netG.flowUpsamplerNet.scaleW)
        z = torch.normal(mean=0, std=heat, size=(batch_size, C, H, W)) if heat > 0 else torch.zeros(
            (batch_size, C, H, W))
        return z
