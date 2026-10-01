import pytest

from numilab_human.open_knee_febio_log import CONNECTOR_IDS, FIELDS, RIGID_BODY_IDS, parse_febio_log


def log(time="1.046542", banner="1.04654"):
    result = f"--- v e r s i o n - 2 . 9 . 1 ---\n------- converged at time : {banner}\n"
    for i, (name, (width, _)) in enumerate(FIELDS.items(), 1):
        result += (f"\nData Record #{i}\n================\nStep = 1\nTime = {time}\nData = {name}\n")
        ids = CONNECTOR_IDS if name.startswith('Rigid_Connector') else RIGID_BODY_IDS
        for identifier in sorted(ids):
            result += str(identifier) + ' ' + ' '.join(['0'] * width) + '\n'
    return result + "\nNumber of time steps completed .... : 1\nN O R M A L   T E R M I N A T I O N\n"


def test_printed_time_precision_is_preserved_without_false_rejection():
    r = parse_febio_log(log())
    assert r['version'] == '2.9.1'
    assert r['status'] == 'normal_termination_with_complete_observations'
    assert r['records'][0]['continuation_time'] == 1.046542
    assert r['accepted_times'] == [1.04654]


def test_rejected_trial_diagnostics_do_not_become_failed_accepted_states():
    text = "* ERROR *\nNegative jacobian was detected at element 42\n" + log()
    r = parse_febio_log(text)
    assert r['status'] == 'normal_termination_with_complete_observations'
    assert r['negative_jacobian_trial_diagnostics'] == 1
    assert r['error_blocks'] == 1


@pytest.mark.parametrize('change', [
    lambda s: s.replace('Time = 1.046542', 'Time = 1.047'),
    lambda s: s.replace('17 0 0 0\n', ''),
    lambda s: s.replace('N O R M A L   T E R M I N A T I O N', ''),
    lambda s: s.replace('completed .... : 1', 'completed .... : 2'),
    lambda s: s + 'E R R O R   T E R M I N A T I O N\n',
])
def test_footer_cannot_promote_truncated_failed_or_misaligned_observations(change):
    assert parse_febio_log(change(log()))['status'] == 'failed_or_incomplete'


def test_nonfinite_vector_is_rejected():
    with pytest.raises(ValueError, match='invalid source observation'):
        parse_febio_log(log().replace('1 0 0 0\n', '1 nan 0 0\n', 1))


def test_duplicate_records_are_rejected():
    with pytest.raises(ValueError, match='duplicate source observation'):
        parse_febio_log(log() + 'Data Record' + log().split('Data Record', 1)[1])
