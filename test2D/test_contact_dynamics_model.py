import pytest

torch = pytest.importorskip("torch")

from learning.contact_dynamics.model import ContactImpulseNet


def test_contact_impulse_constraints_and_shapes():
    model = ContactImpulseNet()
    features = torch.randn(12, 8)
    friction = torch.full((12,), 0.4)
    impulse, mode_logits = model(features, friction)
    assert impulse.shape == (12, 2)
    assert mode_logits.shape == (12, 2)
    assert torch.all(impulse[:, 0] >= 0)
    assert torch.all(impulse[:, 1].abs() <= friction * impulse[:, 0] + 1e-6)
    assert torch.isfinite(impulse).all()
