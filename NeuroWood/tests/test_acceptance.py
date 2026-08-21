from core.acceptance import evaluate_acceptance


def _summary(small=0, medium=0, large=0):
    defects = []
    defects += [{"area_pct": 0.8} for _ in range(small)]
    defects += [{"area_pct": 2.0} for _ in range(medium)]
    defects += [{"area_pct": 4.0} for _ in range(large)]
    return {"patches": [{"frame_id": 0, "defects": defects}]}


def _cfg():
    return {
        "small_max_area_pct": 1.0,
        "medium_max_area_pct": 3.0,
        "max_small": 3,
        "max_medium": 2,
        "max_large": 1,
    }


def test_exact_limits_are_accepted():
    summary = _summary(small=3, medium=2, large=1)
    classification, details = evaluate_acceptance(summary, _cfg())
    assert classification == "ACCEPT"
    assert details["accepted"] is True


def test_more_than_three_small_rejects():
    summary = _summary(small=4)
    classification, _ = evaluate_acceptance(summary, _cfg())
    assert classification == "REJECT"


def test_more_than_two_medium_rejects():
    summary = _summary(medium=3)
    classification, _ = evaluate_acceptance(summary, _cfg())
    assert classification == "REJECT"


def test_more_than_one_large_rejects():
    summary = _summary(large=2)
    classification, _ = evaluate_acceptance(summary, _cfg())
    assert classification == "REJECT"
