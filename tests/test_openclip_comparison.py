"""Offline tests for the OpenAI/OpenCLIP comparison plumbing."""

import math
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from models import OPENCLIP_PRETRAINED, load_model
from test import run_test_methods
from utils import clip_classifier, get_clip_logits, get_res_logits


class FakeVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(2.0)))

    def encode_text(self, tokens):
        return tokens.float()

    def encode_image(self, images):
        return images.float()


def fake_tokenizer(texts):
    rows = []
    for index, _text in enumerate(texts, start=1):
        rows.append([float(index), 1.0, 0.5])
    return torch.tensor(rows)


class ComparisonTests(unittest.TestCase):
    def test_classifier_uses_injected_tokenizer_and_native_logit_scale(self):
        model = FakeVLM()
        features = clip_classifier(
            ['cat', 'dog'],
            ['a photo of a {}', 'a sketch of a {}'],
            model,
            fake_tokenizer,
            torch.device('cpu'),
        )

        self.assertEqual(features.shape, (2, 2, 3))
        norms = features.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.full_like(norms, 2.0)))

    def test_score_functions_are_model_agnostic(self):
        model = FakeVLM()
        text_features = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ]
        )
        images = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        prompt_logits = get_clip_logits(images, model, text_features)
        self.assertEqual(prompt_logits.shape, (2, 2, 2))

        weights = torch.full((2, 2), 0.5)
        result_logits = get_res_logits(images, model, text_features, weights)
        self.assertEqual(result_logits.shape, (2, 2))

    def test_method_metrics_count_images_instead_of_averaging_batches(self):
        model = FakeVLM()
        text_features = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )
        loader = [
            (
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([0, 1]),
            ),
            (
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([0]),
            ),
        ]

        metrics = run_test_methods(
            loader,
            model,
            text_features,
            temp=1.0,
            score_lambdas=[0.01],
        )

        for method in ('MPE', 'WPE', 'CARPRT-lambda-0.01', 'CARPRT-release'):
            self.assertEqual(metrics[method]['correct'], 3)
            self.assertEqual(metrics[method]['total'], 3)
            self.assertEqual(metrics[method]['accuracy'], 100.0)

    def test_openclip_loader_uses_requested_laion_checkpoint(self):
        model = FakeVLM()
        preprocess = object()
        calls = {}

        def create_model_and_transforms(model_name, pretrained):
            calls['model_name'] = model_name
            calls['pretrained'] = pretrained
            return model, None, preprocess

        fake_open_clip = SimpleNamespace(
            create_model_and_transforms=create_model_and_transforms,
            get_tokenizer=lambda model_name: fake_tokenizer,
        )

        with mock.patch.dict(sys.modules, {'open_clip': fake_open_clip}):
            loaded = load_model(
                'openclip',
                'ViT-B/16',
                torch.device('cpu'),
                openclip_pretrained=OPENCLIP_PRETRAINED,
            )

        self.assertEqual(calls['model_name'], 'ViT-B-16')
        self.assertEqual(calls['pretrained'], 'laion2b_s34b_b88k')
        self.assertIs(loaded.preprocess, preprocess)
        self.assertIs(loaded.tokenizer, fake_tokenizer)

    def test_openai_rn50_uses_fp32_on_mps(self):
        model = FakeVLM().half()

        with mock.patch('models.clip.load', return_value=(model, object())):
            loaded = load_model('openai', 'RN50', torch.device('mps'))

        self.assertEqual(next(loaded.model.parameters()).dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
