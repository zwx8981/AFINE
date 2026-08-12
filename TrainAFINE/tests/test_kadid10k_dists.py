import json

import numpy as np
import pytest
import torch

from basicsr.archs.afine_arch import AFINEDhead
from basicsr.data.kadid10k_jsonl_dataset import KADID10KJsonlDataset, parse_kadid10k_record
from basicsr.metrics.iqa_metrics import calculate_iqa_metrics


def _record(mos='4.5'):
    return {
        'id': 'kadid10k_000001',
        'task_type': 'kadid-10k',
        'messages': [
            {'role': 'user', 'content': [
                {'type': 'image', 'image': 'I07.png'},
                {'type': 'image', 'image': 'I07_01_01.png'},
                {'type': 'text', 'text': 'Assess quality.'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': mos}]}
        ]
    }


def test_parse_record_preserves_image_order(tmp_path):
    sample = parse_kadid10k_record(_record(), str(tmp_path))
    assert sample['reference_path'] == str(tmp_path / 'I07.png')
    assert sample['distorted_path'] == str(tmp_path / 'I07_01_01.png')
    assert sample['mos'] == 4.5


def test_jsonl_dataset(tmp_path):
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    from PIL import Image
    Image.fromarray(image).save(tmp_path / 'I07.png')
    Image.fromarray(image).save(tmp_path / 'I07_01_01.png')
    jsonl = tmp_path / 'train.jsonl'
    jsonl.write_text(json.dumps(_record()) + '\n', encoding='utf-8')
    dataset = KADID10KJsonlDataset({
        'jsonl_path': str(jsonl), 'image_root': str(tmp_path), 'phase': 'val',
        'mean': [0, 0, 0], 'std': [1, 1, 1]
    })
    sample = dataset[0]
    assert sample['reference'].shape == (3, 32, 32)
    assert torch.equal(sample['reference'], sample['distorted'])
    assert sample['mos'] == 4.5


def test_dists_head_identity_and_backward():
    head = AFINEDhead(chns=(3, 4))
    image = torch.rand(2, 3, 8, 8)
    features = [torch.rand(2, 16, 4)]
    identity = head(image, image, features, features)
    assert torch.allclose(identity, torch.zeros_like(identity), atol=1e-5)
    head(image, image * 0.8, features, [features[0] * 0.8]).mean().backward()
    assert head.alpha.grad is not None


def test_iqa_metrics_perfect_order():
    result = calculate_iqa_metrics([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], logistic_mapping=False)
    assert result['SRCC'] == pytest.approx(1)
    assert result['PLCC'] == pytest.approx(1)
    assert result['KRCC'] == pytest.approx(1)
