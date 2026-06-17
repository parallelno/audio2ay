from audio2ay.synth.volume_quantizer import amp_to_vol, dac_level, db_to_vol, envelope_bit


def test_amp_to_vol_endpoints() -> None:
    assert amp_to_vol(0.0) == 0
    assert amp_to_vol(1.0) == 15
    # Negative or zero → silence.
    assert amp_to_vol(-0.5) == 0


def test_dac_levels_monotonic() -> None:
    levels = [dac_level(i) for i in range(16)]
    for a, b in zip(levels, levels[1:]):
        assert b > a, (a, b)


def test_db_to_vol_quiet_threshold() -> None:
    assert db_to_vol(-100.0) == 0
    # 0 dBFS should saturate.
    assert db_to_vol(0.0) == 15
    # -3 dB should be a couple of steps below max.
    assert db_to_vol(-3.0) <= 14


def test_envelope_bit() -> None:
    assert envelope_bit(0, False) == 0
    assert envelope_bit(15, False) == 15
    assert envelope_bit(7, True) == 0x10
    # Out-of-range tolerated.
    assert envelope_bit(-1, False) == 0
    assert envelope_bit(99, False) == 15
