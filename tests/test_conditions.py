import pytest

from pave.data.conditions import parse_filename, parse_internal_label


@pytest.mark.parametrize("stem,label7,rid", [
    ("0Nm_Normal", "H", "L0_H"),
    ("0Nm_BPFI_03", "IR_0p3", "L0_IR_0p3"),
    ("2Nm_BPFI_10", "IR_1p0", "L2_IR_1p0"),
    ("4Nm_BPFO_30", "OR_3p0", "L4_OR_3p0"),
    ("4Nm_BPFO_03", "OR_0p3", "L4_OR_0p3"),
])
def test_bearing_filenames(stem, label7, rid):
    c = parse_filename(stem)
    assert c.label7 == label7 and c.recording_id == rid and c.in_bearing_scope


def test_out_of_scope_and_typo():
    c = parse_filename("2Nm_Unbalalnce_0583mg")  # typo in distributed archive
    assert c.fault_group == "Unbalance" and c.severity_value == 583 and c.severity_unit == "mg"
    assert c.label7 is None and not c.in_bearing_scope
    m = parse_filename("4Nm_Misalign_05")
    assert m.fault_group == "Misalign" and m.severity_value == pytest.approx(0.5)


def test_bad_filename():
    with pytest.raises(ValueError):
        parse_filename("0Nm_Foo_03")
    with pytest.raises(ValueError):
        parse_filename("Normal_0Nm")


@pytest.mark.parametrize("text,label7", [
    ("210625_0Nm_Inner_0.3mm", "IR_0p3"),
    ("LogFile_0Nm_Inner_1mm", "IR_1p0"),
    ("LogFile_0Nm_Inner_03mm", "IR_0p3"),
    ("LogFile_2Nm_Outer_3mm", "OR_3p0"),
    ("210630_2Nm_Outer_1mm _RPMFix", "OR_1p0"),
    ("LogFile_4Nm_Normal", "H"),
])
def test_internal_labels(text, label7):
    assert parse_internal_label(text).label7 == label7


@pytest.mark.parametrize("text,mg", [
    ("LogFile_0Nm_Unbalance_0.583", 583), ("LogFile_2Nm_Unbalance_0.1169", 1169),
    ("LogFile_4Nm_Unbalance_1751mg", 1751), ("210625_0Nm_Unbal_584mg", 584),
])
def test_internal_unbalance_masses(text, mg):
    c = parse_internal_label(text)
    assert c.fault_group == "Unbalance" and c.severity_value == mg


def test_same_condition_tolerance():
    from pave.data.conditions import same_condition
    a = parse_filename("0Nm_Unbalance_0583mg")
    assert same_condition(a, parse_internal_label("210625_0Nm_Unbal_584mg"))
    assert not same_condition(a, parse_internal_label("210625_0Nm_Unbal_1169mg"))
    assert not same_condition(parse_filename("0Nm_BPFI_03"), parse_internal_label("LogFile_0Nm_Inner_1mm"))
    assert same_condition(parse_filename("2Nm_Normal"), parse_internal_label("LogFile_2Nm_Normal"))


def test_internal_label_without_load_returns_none():
    # the 0 Nm OR 0.3 vibration run name has no load token
    assert parse_internal_label("210630_Outer0.3mm") is None
