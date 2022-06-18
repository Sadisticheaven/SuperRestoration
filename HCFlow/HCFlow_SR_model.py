# base model for HCFlow
import logging
from collections import OrderedDict
import torch
from HCFlowNet_SR_arch import HCFlowNet_SR
from base_model import BaseModel
logger = logging.getLogger('base')


class HCFlowSRModel(BaseModel):
    def __init__(self, opt, heats=[0.0], step=0):
        super(HCFlowSRModel, self).__init__(opt)
        self.opt = opt

        self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = HCFlowNet_SR(opt, step).to(self.device)
        self.heats = heats
        # val
        if 'val' in opt:
            # self.heats = opt['val']['heats']
            self.n_sample = opt['val']['n_sample']
            self.sr_mode = opt['val']['sr_mode']

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
        else:
            self.real_H = None

    def test(self):
        self.netG.eval()
        self.fake_H = {}

        with torch.no_grad():
            if self.real_H is None:
                nll = torch.zeros(1)
            else:
                # hr->lr+z, calculate nll
                self.fake_L_from_H, nll = self.netG(hr=self.real_H, lr=self.var_L, u=None, reverse=False, training=False)

            # lr+z->hr
            for heat in self.heats:
                for sample in range(self.n_sample):
                    # z = self.get_z(heat, seed=1, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
                    self.fake_H[(heat, sample)] = self.netG(lr=self.var_L,
                                  z=None, u=None, eps_std=heat, reverse=True, training=False)

        self.netG.train()

        return nll.mean().item()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        for heat in self.heats:
            for i in range(self.n_sample):
                out_dict[('SR', heat, i)] = self.fake_H[(heat, i)].detach()[0].float().cpu()

        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
            out_dict['LQ_fromH'] = self.fake_L_from_H.detach()[0].float().cpu()

        return out_dict

