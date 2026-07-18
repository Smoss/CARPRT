"""Model loading helpers for the OpenAI CLIP/OpenCLIP comparison."""

from dataclasses import dataclass
from typing import Callable

import clip
import torch


OPENCLIP_MODEL = 'ViT-B-16'
OPENCLIP_PRETRAINED = 'laion2b_s34b_b88k'


@dataclass
class LoadedModel:
    """A VLM plus the preprocessing/tokenization functions CARPRT needs."""

    name: str
    model: torch.nn.Module
    preprocess: Callable
    tokenizer: Callable


def resolve_device(device_name):
    """Resolve ``auto`` to the fastest PyTorch device available."""
    if device_name != 'auto':
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _load_openai_clip(backbone, device):
    # The bundled OpenAI CLIP loader handles its own checkpoint download.
    model, preprocess = clip.load(backbone, device=device, jit=False)
    if device.type in {'cpu', 'mps'}:
        # OpenAI CLIP converts CUDA models to fp16, but fp32 has broader CPU/MPS
        # operator support.
        model = model.float()
    return LoadedModel(
        name=f'OpenAI CLIP {backbone}',
        model=model.eval(),
        preprocess=preprocess,
        tokenizer=clip.tokenize,
    )


def _load_openclip(backbone, pretrained, device):
    if backbone != 'ViT-B/16':
        raise ValueError('The OpenCLIP comparison is fixed to ViT-B/16.')

    # Import lazily so the original OpenAI-only path still works without the
    # optional comparison dependency installed.
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL,
        pretrained=pretrained,
    )
    model = model.to(device)
    if device.type in {'cpu', 'mps'}:
        model = model.float()

    return LoadedModel(
        name=f'OpenCLIP {OPENCLIP_MODEL} ({pretrained})',
        model=model.eval(),
        preprocess=preprocess,
        tokenizer=open_clip.get_tokenizer(OPENCLIP_MODEL),
    )


def load_model(source, backbone, device, openclip_pretrained=OPENCLIP_PRETRAINED):
    """Load one of the two VLMs used by the comparison experiment."""
    if source == 'openai':
        return _load_openai_clip(backbone, device)
    if source == 'openclip':
        return _load_openclip(backbone, openclip_pretrained, device)
    raise ValueError(f'Unknown model source: {source}')
