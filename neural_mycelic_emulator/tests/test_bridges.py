import asyncio
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# ContemplativeAI bridge test (requires project root on PYTHONPATH)
# ---------------------------------------------------------------------------
from ContemplativeAI.mycelic_emulator_bridge import (
    NeuralMycelicBridge,
    EmulatorResponseType,
    Phase,
)

# ---------------------------------------------------------------------------
# Spiramycel adapter test (no dependency on Spiramycel itself)
# ---------------------------------------------------------------------------
from neural_mycelic_emulator.models.lstm_emulator import LSTMEmulator
from neural_mycelic_emulator.spiramycel_bridge import _EmulatorAdapter  # type: ignore

from neural_mycelic_emulator.preprocessor.glyph_encoder import GLYPHS


@pytest.mark.asyncio
async def test_mycelic_bridge_exhale_response(tmp_path: Path):
    # Create minimal checkpoint so bridge loads real adapter
    ckpt = tmp_path / "dummy.pt"
    model = LSTMEmulator(vocab_size=16)
    import torch

    torch.save(model.state_dict(), ckpt)

    bridge = NeuralMycelicBridge(
        model_tag="cordyceps_small", checkpoint=ckpt, max_generate_len=16
    )

    resp = await bridge.exhale_exchange("status please", Phase.EXHALE)
    assert resp.response_type == EmulatorResponseType.GLYPH_SEQUENCE
    assert resp.glyph_sequence and len(resp.glyph_sequence) > 0

    # Non-exhale should yield silence
    silence = await bridge.exhale_exchange("ignored", Phase.INHALE)
    assert silence.response_type == EmulatorResponseType.SILENCE


def test_emulator_adapter_shapes():
    import torch

    base = LSTMEmulator(vocab_size=16)
    adapter = _EmulatorAdapter(base)  # type: ignore

    tokens = torch.randint(0, 16, (2, 5))
    glyph_logits, eff, sil, *_ = adapter(tokens)
    assert glyph_logits.shape[:2] == tokens.shape
    assert eff.shape == sil.shape == (*tokens.shape, 1)


def test_glyph_spec_alignment():
    # Ensure we keep exactly 8 activity glyphs (ids 0–7)
    activity_ids = [v for k, v in GLYPHS.items() if k != "PAD"]
    assert max(activity_ids) == 7
    assert len(GLYPHS) == 8 