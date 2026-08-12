from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from CLIP_ReturnFea import clip
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_iqa_metrics
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class KADID10KDISTSModel(BaseModel):
    """Stage-2-style CLIP-DISTS model supervised by KADID-10k MOS."""

    def __init__(self, opt):
        super().__init__(opt)
        self.clip_model, _ = clip.load(opt['path_CLIP']['pretrain_CLIP_path'], device='cpu', jit=False)
        if opt['path_CLIP'].get('mode', 'Original').lower() == 'finetune':
            checkpoint = torch.load(opt['path_CLIP']['finetune_CLIP_path'], map_location='cpu')
            self.clip_model.load_state_dict(checkpoint)
        self.clip_model = self.clip_model.to(self.device)

        self.net_dhead = self.model_to_device(build_network(opt['network_dhead']))
        self.print_network(self.net_dhead)
        load_path = opt.get('path_dhead', {}).get('pretrain_network_dhead')
        if load_path:
            path_opt = opt['path_dhead']
            self.load_network(self.net_dhead, load_path, path_opt.get('strict_load_dhead', True),
                              path_opt.get('param_key_dhead', 'params'))
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']
        self.net_dhead.train()
        self.finetune_clip = train_opt.get('finetune_CLIP', False)
        self.clip_model.train(self.finetune_clip)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = self.finetune_clip
        self.cri_mos = build_loss(train_opt['mos_opt']).to(self.device)
        optim_opt = train_opt['optim_dhead'].copy()
        optim_type = optim_opt.pop('type')
        self.optimizer_dhead = self.get_optimizer(optim_type, self.net_dhead.parameters(), **optim_opt)
        self.optimizers.append(self.optimizer_dhead)
        if self.finetune_clip:
            optim_opt = train_opt['optim_clip'].copy()
            optim_type = optim_opt.pop('type')
            self.optimizer_clip = self.get_optimizer(optim_type, self.clip_model.parameters(), **optim_opt)
            self.optimizers.append(self.optimizer_clip)
        self.setup_schedulers()

    def feed_data(self, data):
        self.reference = data['reference'].to(self.device)
        self.distorted = data['distorted'].to(self.device)
        self.mos = data['mos'].float().to(self.device)
        self.sample_ids = data.get('id')

    def _target_distance(self, mos):
        score = self.opt.get('score', {})
        minimum, maximum = score.get('min', 1.0), score.get('max', 5.0)
        normalized = (mos - minimum) / (maximum - minimum)
        if score.get('higher_better', True):
            normalized = 1 - normalized
        return normalized.clamp(0, 1)

    def _predict(self, distorted, reference, with_grad=False):
        context = torch.enable_grad() if with_grad else torch.no_grad()
        with context:
            _, distorted_features = self.clip_model.encode_image(distorted)
            _, reference_features = self.clip_model.encode_image(reference)
        return self.net_dhead(distorted, reference, distorted_features, reference_features).view(-1)

    def optimize_parameters(self, current_iter):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        prediction = self._predict(self.distorted, self.reference, self.finetune_clip)
        loss = self.cri_mos(prediction, self._target_distance(self.mos))
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(l_mos=loss))

    def test(self):
        self.net_dhead.eval()
        self.clip_model.eval()
        height, width = self.distorted.shape[-2:]
        pad_h, pad_w = (-height) % 16, (-width) % 16
        distorted = F.pad(self.distorted, (0, pad_w, 0, pad_h), mode='reflect') if pad_h or pad_w else self.distorted
        reference = F.pad(self.reference, (0, pad_w, 0, pad_h), mode='reflect') if pad_h or pad_w else self.reference
        with torch.no_grad():
            self.output = self._predict(distorted, reference)
        if self.is_train:
            self.net_dhead.train()
            self.clip_model.train(self.finetune_clip)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        predictions, targets = [], []
        pbar = tqdm(total=len(dataloader), unit='pair') if self.opt['val'].get('pbar', False) else None
        for data in dataloader:
            self.feed_data(data)
            self.test()
            # DISTS is a distance; validation metrics are quality-oriented.
            predictions.extend((-self.output).detach().cpu().tolist())
            targets.extend(self.mos.detach().cpu().tolist())
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
        results = calculate_iqa_metrics(predictions, targets, self.opt['val'].get('logistic_mapping', True))
        configured = self.opt['val'].get('metrics', {name: {} for name in results})
        self.metric_results = {name: results[name] for name in configured}
        self._initialize_best_metric_results(dataset_name)
        for name, value in self.metric_results.items():
            self._update_best_metric_result(dataset_name, name, value, current_iter)
        logger = get_root_logger()
        logger.info('Validation %s\n%s', dataset_name,
                    '\n'.join(f'\t# {name}: {value:.6f}' for name, value in self.metric_results.items()))
        if tb_logger:
            for name, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{name}', value, current_iter)

    def save(self, epoch, current_iter):
        self.save_network(self.net_dhead, 'net_dhead', current_iter)
        if self.opt['train'].get('finetune_CLIP', False):
            self.save_network(self.clip_model, 'clip_model', current_iter, param_key='params')
        self.save_training_state(epoch, current_iter)
