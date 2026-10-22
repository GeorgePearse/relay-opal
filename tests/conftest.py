import pytest
import torch

from relay_opal.models import GPTConfig, NoiseGPTSpec, make_model

TINY = GPTConfig(sequence_len=64, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)


@pytest.fixture
def tiny():
    """A 2-block, width-64 Noise GPT with the Relay script's init, on CPU, and one batch of two sequences."""
    torch.manual_seed(0)
    model = make_model(torch.device("cpu"), 0, TINY)
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, TINY.vocab_size, (2, 64), generator=g)
    y = torch.randint(0, TINY.vocab_size, (2, 64), generator=g)
    return NoiseGPTSpec(model), x, y
