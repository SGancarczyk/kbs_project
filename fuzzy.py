def _clamp01(x):
    return max(0.0, min(1.0, x))
def climate_memberships(temp):
    cold = _clamp01((17 - temp) / (17 - 10))
    warm = _clamp01((temp - 18) / (25 - 18))
    if temp <= 12 or temp >= 24:
        mild = 0.0
    elif temp <= 18:
        mild = (temp - 12) / (18 - 12)
    else:
        mild = (24 - temp) / (24 - 18)
    return {"cold": cold, "mild": _clamp01(mild), "warm": warm}
def climate_membership(temp, term):
    return climate_memberships(temp).get(term, 0.0)
_BUDGET_ORDER = {"Budget": 0, "Mid-range": 1, "Luxury": 2}
_ORDINAL_MEMBERSHIP = {0: 1.0, 1: 0.5, 2: 0.0}
def budget_membership(city_level, requested_level):
    ci = _BUDGET_ORDER.get(city_level)
    ri = _BUDGET_ORDER.get(requested_level)
    if ci is None or ri is None:
        return 0.0
    return _ORDINAL_MEMBERSHIP[abs(ci - ri)]
_DURATION_ORDER = {"Day trip": 0, "Weekend": 1, "Short trip": 2, "One week": 3, "Long trip": 4}
def duration_membership(city_durations, requested):
    req = _DURATION_ORDER.get(requested)
    if req is None:
        return 0.0
    best = 0.0
    for d in city_durations:
        idx = _DURATION_ORDER.get(d)
        if idx is None:
            continue
        degree = max(0.0, 1 - abs(idx - req) / 2)
        if degree > best:
            best = degree
    return best
def fuzzy_and(degrees):
    if not degrees:
        return 0.0
    return min(degrees)
def fuzzy_or(degrees):
    if not degrees:
        return 0.0
    return max(degrees)
