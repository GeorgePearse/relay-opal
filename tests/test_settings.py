import pytest

from relay_opal import RelaySettings


def test_defaults_are_the_submitted_relay():
    s = RelaySettings().validate(2048)
    assert (s.routes, s.keys, s.values, s.hidden, s.head, s.norm, s.mlp_dtype) == (
        512,
        "exact",
        "closed_form",
        "all",
        "closed_form",
        "closed_form",
        "bf16",
    )
    assert s.freeze == ()


def test_round_trip():
    s = RelaySettings(routes=256, freeze=["k"])
    assert RelaySettings.from_dict(s.to_dict()) == RelaySettings(routes=256, freeze=("k",))


@pytest.mark.parametrize(
    "bad",
    [
        dict(routes=-1),
        dict(routes=4096),
        dict(keys="guess"),
        dict(cycles=0),
        dict(chain_sigma=0),
        dict(hidden="topk", hidden_topk=100),
        dict(freeze=("heads",)),
        dict(tokens=True),
    ],
)
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        RelaySettings(**bad).validate(2048)


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown"):
        RelaySettings.from_dict(dict(route=1))
