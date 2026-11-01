"""Public candidate supports fresh experiments only."""
def validate_recovery_memory_entry(cases, evaluator_fixture=None, root=None):
    if evaluator_fixture or any(any(k in c for k in ('memory_continuation','recovery_reference','recovery_pose')) for c in cases):
        raise ValueError('Public candidate supports fresh inputs only; no historical continuation qualification is supplied')
