import math

import pytest

from pave.signal import physics as ph


def test_paper_reference_values():
    """Jung et al. 2023 Table 1: 3010 rpm -> FTF 19.94, BPFI 272.07, BPFO 179.43 Hz.

    The printed values follow from fr = 3010/60 = 50.1667 Hz exactly; using the
    rounded fr = 50.17 Hz gives BPFI = 272.09 Hz (difference 0.02 Hz).
    """
    fr = ph.PAPER_RPM / 60.0
    assert ph.ftf(fr) == pytest.approx(19.94, abs=0.005)
    assert ph.bpfi(fr) == pytest.approx(272.07, abs=0.005)
    assert ph.bpfo(fr) == pytest.approx(179.43, abs=0.005)
    assert ph.bpfi(ph.PAPER_SHAFT_HZ) == pytest.approx(272.09, abs=0.005)


def test_paper_bsf_is_twice_standard_bsf():
    fr = ph.PAPER_RPM / 60.0
    assert ph.bsf(fr) == pytest.approx(117.10, abs=0.01)
    assert 2 * ph.bsf(fr) == pytest.approx(ph.PAPER_VALUES_HZ["BSF_paper"], abs=0.01)


def test_rpm_consistency():
    assert ph.PAPER_RPM / 60 == pytest.approx(ph.PAPER_SHAFT_HZ, abs=0.01)


def test_identities_and_linearity():
    g = ph.NSK_6205_DDU
    for fr in (10.0, 49.89, 50.17):
        assert ph.bpfo(fr) + ph.bpfi(fr) == pytest.approx(g.n_balls * fr)
        assert ph.bpfo(fr) == pytest.approx(g.n_balls * ph.ftf(fr))
    assert ph.bpfi(2 * 50.0) == pytest.approx(2 * ph.bpfi(50.0))


def test_contact_angle_effect():
    g30 = ph.BearingGeometry("t", 9, 7.9, 38.5, 30.0)
    assert ph.bpfo(50, g30) > ph.bpfo(50)  # cos(30) < 1 reduces the d/D term
    assert g30.ratio == pytest.approx(7.9 / 38.5 * math.cos(math.radians(30)))


def test_orders_speed_independent():
    o = ph.orders()
    assert o["BPFO"] == pytest.approx(179.43 / 50.17, rel=1e-3)
    assert o["BPFI"] == pytest.approx(272.07 / 50.17, rel=1e-3)


def test_shaft_harmonic_collision():
    assert ph.shaft_harmonic_collisions(250.0, 50.0, 1.0) == [5]
    assert ph.shaft_harmonic_collisions(ph.bpfi(50.17), 50.17, 2.7) == []
