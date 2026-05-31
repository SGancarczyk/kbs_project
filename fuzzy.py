# fuzzy.py
# ----------------------------------------------------------------------------
# THE FUZZY-LOGIC LAYER (course deck "2.2 Fuzzy Logic").
# Plain rules use hard cutoffs, which are brittle: "warm = temp >= 22" treats a
# 21.9 C city as "not warm at all". Fuzzy logic replaces true/false with a
# DEGREE OF MEMBERSHIP in [0.0, 1.0]. A 21 C city can be "mild 0.5 and warm 0.43"
# at once. The shapes that turn a raw value into a degree are MEMBERSHIP
# FUNCTIONS (the triangle/trapezoid pictures in the slides); turning values into
# degrees is FUZZIFICATION. NOTE: we keep full precision here (no rounding) so
# the reasoning stays accurate; rounding happens only when we print to the user.
# ----------------------------------------------------------------------------
def _clamp01(x):
    # Keep a number inside the valid membership range [0.0, 1.0].
    return max(0.0, min(1.0, x))
# ----- CLIMATE: a continuous value (temperature) -> 3 overlapping fuzzy sets --
def climate_memberships(temp):
    # "cold": fully 1.0 at/below 10 C, ramps down to 0.0 by 17 C (left shoulder).
    cold = _clamp01((17 - temp) / (17 - 10))
    # "warm": 0.0 until 18 C, ramps up to 1.0 by 25 C, stays 1.0 above (right shoulder).
    warm = _clamp01((temp - 18) / (25 - 18))
    # "mild": a TRIANGLE peaking at 18 C, zero at 12 C and 24 C.
    if temp <= 12 or temp >= 24:
        mild = 0.0
    elif temp <= 18:
        mild = (temp - 12) / (18 - 12)
    else:
        mild = (24 - temp) / (24 - 18)
    # Return full-precision degrees (no rounding) for accurate downstream maths.
    return {"cold": cold, "mild": _clamp01(mild), "warm": warm}
def climate_membership(temp, term):
    # How much does this temperature belong to ONE climate term ("warm" etc.)?
    return climate_memberships(temp).get(term, 0.0)
# ----- BUDGET: an ordered category -> graded match ----------------------------
_BUDGET_ORDER = {"Budget": 0, "Mid-range": 1, "Luxury": 2}
# distance-in-steps -> membership: same level=1.0, one step off=0.4, two off=0.0
_ORDINAL_MEMBERSHIP = {0: 1.0, 1: 0.4, 2: 0.0}
def budget_membership(city_level, requested_level):
    # How well does a city's budget satisfy the level the user asked for? We use
    # .get and guard against unknown labels so bad/unknown input returns 0.0
    # instead of crashing with a KeyError (robustness).
    ci = _BUDGET_ORDER.get(city_level)
    ri = _BUDGET_ORDER.get(requested_level)
    if ci is None or ri is None:
        return 0.0
    return _ORDINAL_MEMBERSHIP[abs(ci - ri)]
# ----- DURATION: an ordered category; a city accepts SEVERAL ------------------
_DURATION_ORDER = {"Day trip": 0, "Weekend": 1, "Short trip": 2, "One week": 3, "Long trip": 4}
def duration_membership(city_durations, requested):
    # Score how close the user's requested duration is to the city's BEST
    # matching ideal duration: exact=1.0, one step=0.5, two+=0.0. Unknown labels
    # are skipped/zeroed so we never crash on unexpected data.
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
    # FUZZY "AND" = MINIMUM (course rule). The whole condition is only as strong
    # as its weakest member. Kept here for course consistency / general use.
    if not degrees:
        return 0.0
    return min(degrees)
def fuzzy_or(degrees):
    # FUZZY "OR" = MAXIMUM (course rule). Satisfying any one option is enough; we
    # take the strongest. Used when the user gives several acceptable values for
    # one criterion (e.g. "europe or asia", "warm or mild").
    if not degrees:
        return 0.0
    return max(degrees)
