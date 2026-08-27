from src.power import check_sample_size, required_sample_size


def test_required_sample_size_larger_for_smaller_effect():
    n_small_effect = required_sample_size(0.10, 0.003)
    n_large_effect = required_sample_size(0.10, 0.05)
    assert n_small_effect > n_large_effect


def test_sample_size_check_flags_inadequate():
    check = check_sample_size(n_actual=100, n_required=10000)
    assert not check.is_adequate

    check_ok = check_sample_size(n_actual=20000, n_required=10000)
    assert check_ok.is_adequate
