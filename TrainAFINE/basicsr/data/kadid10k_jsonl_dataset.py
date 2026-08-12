import json
import math
import os
import random

import torch.nn.functional as F
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment
from basicsr.utils import img2tensor, imfromfile
from basicsr.utils.registry import DATASET_REGISTRY


def _resolve_path(image_root, path):
    return path if os.path.isabs(path) else os.path.join(image_root, path)


def parse_kadid10k_record(record, image_root='', strict_task_type=True):
    """Convert one conversation-style KADID-10k record to an IQA sample."""
    sample_id = record.get('id', '<unknown>')
    if strict_task_type and record.get('task_type') != 'kadid-10k':
        raise ValueError(f'{sample_id}: expected task_type "kadid-10k"')

    user = next((message for message in record.get('messages', []) if message.get('role') == 'user'), None)
    assistant = next((message for message in record.get('messages', []) if message.get('role') == 'assistant'), None)
    if user is None or assistant is None:
        raise ValueError(f'{sample_id}: both user and assistant messages are required')

    images = [item.get('image') for item in user.get('content', []) if item.get('type') == 'image']
    if len(images) != 2 or not all(images):
        raise ValueError(f'{sample_id}: user message must contain exactly two images')
    texts = [item.get('text') for item in assistant.get('content', []) if item.get('type') == 'text']
    if not texts:
        raise ValueError(f'{sample_id}: assistant message must contain a MOS text value')
    try:
        mos = float(texts[0])
    except (TypeError, ValueError) as error:
        raise ValueError(f'{sample_id}: invalid MOS value {texts[0]!r}') from error
    if not math.isfinite(mos):
        raise ValueError(f'{sample_id}: MOS must be finite')

    return {
        'id': sample_id,
        'reference_path': _resolve_path(image_root, images[0]),
        'distorted_path': _resolve_path(image_root, images[1]),
        'mos': mos
    }


@DATASET_REGISTRY.register()
class KADID10KJsonlDataset(data.Dataset):
    """KADID-10k reference/distorted pairs stored as conversation JSONL."""

    def __init__(self, opt):
        self.opt = opt
        self.phase = opt.get('phase', 'train')
        self.gt_size = opt.get('gt_size')
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        image_root = os.path.expanduser(opt['image_root'])
        jsonl_path = os.path.expanduser(opt['jsonl_path'])
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    sample = parse_kadid10k_record(record, image_root, opt.get('strict_task_type', True))
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f'{jsonl_path}:{line_number}: {error}') from error
                if opt.get('check_files', True):
                    for key in ('reference_path', 'distorted_path'):
                        if not os.path.isfile(sample[key]):
                            raise FileNotFoundError(f'{jsonl_path}:{line_number}: missing image {sample[key]}')
                self.samples.append(sample)
        if not self.samples:
            raise ValueError(f'No samples found in {jsonl_path}')

    def __getitem__(self, index):
        sample = self.samples[index]
        reference = imfromfile(path=sample['reference_path'], float32=True)
        distorted = imfromfile(path=sample['distorted_path'], float32=True)
        if reference.shape != distorted.shape:
            raise ValueError(f'{sample["id"]}: reference and distorted image shapes differ')
        if self.phase == 'train':
            reference, distorted = augment([reference, distorted], self.opt.get('use_hflip', False),
                                           self.opt.get('use_rot', False))
        reference, distorted = img2tensor([reference, distorted], bgr2rgb=True, float32=True)

        _, height, width = reference.shape
        if self.gt_size:
            pad_h, pad_w = max(0, self.gt_size - height), max(0, self.gt_size - width)
            if pad_h or pad_w:
                reference = F.pad(reference, (0, pad_w, 0, pad_h), 'reflect')
                distorted = F.pad(distorted, (0, pad_w, 0, pad_h), 'reflect')
            if self.phase == 'train':
                top = 0 if reference.shape[1] == self.gt_size else random.randint(
                    0, reference.shape[1] - self.gt_size)
                left = 0 if reference.shape[2] == self.gt_size else random.randint(
                    0, reference.shape[2] - self.gt_size)
                reference = reference[:, top:top + self.gt_size, left:left + self.gt_size]
                distorted = distorted[:, top:top + self.gt_size, left:left + self.gt_size]
        if self.mean is not None or self.std is not None:
            normalize(reference, self.mean, self.std, inplace=True)
            normalize(distorted, self.mean, self.std, inplace=True)
        return {
            'id': sample['id'], 'reference': reference, 'distorted': distorted,
            'mos': sample['mos'], 'reference_path': sample['reference_path'],
            'distorted_path': sample['distorted_path'], 'ori_h': height, 'ori_w': width
        }

    def __len__(self):
        return len(self.samples)
