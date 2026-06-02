# inference.py
# ----------------------------------------------------------------------------
# THE INFERENCE ENGINE - the reasoning core of the Knowledge-Based System.
# It turns the user's request (a "query") plus the Knowledge Base into a ranked
# list of recommendations, each with a confidence, and runs forward-chaining
# advisory rules. Techniques, each with one clear job:
#   - crisp rule        -> hard region filter (with a fallback if it empties out)
#   - fuzzy logic       -> graded match of soft criteria (climate/budget/duration)
#   - cosine + strength -> lifestyle taste match that also rewards strong cities
#   - certainty factors -> EACH criterion becomes signed evidence (CF = 2m-1,
#                          i.e. MB-MD), combined MYCIN-style into one confidence
#   - forward chaining  -> advisory hints from the request AND the results
#
# query dict the chatbot fills in:
#   { "regions_include": [...], "regions_exclude": [...],
#     "climate": [...], "climate_exclude": [...],
#     "budget": [...], "budget_exclude": [...],
#     "duration": [...], "duration_exclude": [...],
#     "lifestyle": {dim: importance 1..5},
#     "lifestyle_exclude": [...],
#     "travel_months": [1..12, ...] }   <- NEW: set by the season slot
#
# IMPROVEMENT: when travel_months is present, score_destination() uses the
# city's actual temperature for those months (from avg_temp_monthly in the
# Knowledge Base) rather than the yearly average. This means "warm in December"
# correctly favours Sydney over Stockholm, for example.
# ----------------------------------------------------------------------------
import math
import fuzzy
import rules
import knowledge_base as kb


def cosine_similarity(a, b):
    # ANGLE between two vectors, ignoring length: same direction -> 1.0,
    # perpendicular -> 0.0. dot(a,b) / (|a| * |b|).
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cf_propagate(rule_cf, evidence):
    # Simple CF propagation: rule reliability x evidence strength.
    return rule_cf * evidence


def evidence_cf_from_membership(membership, rule_cf):
    # Turn a fuzzy membership (0..1) into SIGNED certainty-factor evidence using
    # the course definition CF = MB - MD. We read the membership as the Measure
    # of Belief (MB) and its complement (1-membership) as the Measure of
    # Disbelief (MD): CF = m - (1-m) = 2m - 1. So a full match (m=1) is strong
    # positive evidence, a full miss (m=0) is strong NEGATIVE evidence, and a
    # half match (m=0.5) is neutral (0). We then multiply by how reliable this
    # kind of rule is (propagation). Result lies in [-rule_cf, +rule_cf].
    return rule_cf * (2 * membership - 1)


def cf_combine(cf1, cf2):
    # Combine two certainty factors that bear on the SAME conclusion, using the
    # three MYCIN cases from the slides:
    #   both positive -> CF1 + CF2*(1-CF1)         (evidence reinforces)
    #   both negative -> CF1 + CF2*(1+CF1)         (disbelief reinforces)
    #   mixed signs   -> (CF1+CF2)/(1-min(|CF1|,|CF2|))  (they partly cancel)
    if cf1 >= 0 and cf2 >= 0:
        return cf1 + cf2 * (1 - cf1)
    if cf1 <= 0 and cf2 <= 0:
        return cf1 + cf2 * (1 + cf1)
    denominator = 1 - min(abs(cf1), abs(cf2))
    if denominator == 0:
        return 0.0
    return (cf1 + cf2) / denominator


def cf_combine_many(cfs):
    # Fold cf_combine across a list of evidence CFs (starting from 0 = "no
    # opinion"). Each fold step applies whichever of the THREE MYCIN cases fits
    # the running pair (both +, both -, or mixed) - see cf_combine above. Order
    # does not matter for the MYCIN formula, so accumulating left-to-right is safe.
    total = 0.0
    for cf in cfs:
        total = cf_combine(total, cf)
    return total


def _negative_cf_from_membership(membership, rule_cf):
    # For explicit exclusions, high membership is bad. A city that strongly has
    # something the user rejected receives strong negative evidence; a city that
    # lacks it receives no penalty.
    return -rule_cf * membership


def _effective_budget_preferences(query):
    # Align the forward-chaining advice with the ranking. If the user asks for
    # both Budget and Luxury, the rule base says to use Mid-range as a bridge,
    # so the scorer actually ranks Mid-range as the effective compromise.
    budgets = list(query.get("budget") or [])
    if "Budget" in budgets and "Luxury" in budgets:
        return ["Mid-range"]
    return budgets


def _effective_temp(dest, query):
    # NEW: return the temperature that should be used for climate scoring.
    # If the user specified travel months (via the season slot), we look up
    # the city's average temperature specifically for those months from the
    # monthly data stored in the Knowledge Base. This is more accurate than
    # the yearly average because seasonal variation matters: a city that is
    # warm in July may be cold in December.
    # If no months were given we fall back to the yearly average, which is
    # exactly the original behaviour so nothing breaks for existing queries.
    months = query.get("travel_months") or []
    if months:
        seasonal = kb.seasonal_avg_temp(dest["avg_temp_monthly"], months)
        if seasonal is not None:
            return seasonal
    return dest["avg_temp_yearly"]


def score_destination(dest, query):
    # Score ONE destination. Included preferences become signed CF evidence.
    # Explicit exclusions become negative evidence. Missing criteria add no
    # evidence, so absence is not treated as a perfect match.
    evidences = []
    details = {}
    include_memberships = {}

    # CLIMATE: use _effective_temp() so that a travel month/season narrows the
    # temperature to the relevant period instead of the yearly average.
    # Both the include and exclude paths go through the same helper so they are
    # always consistent with each other.
    if query.get("climate"):
        temp = _effective_temp(dest, query)
        m = fuzzy.fuzzy_or([fuzzy.climate_membership(temp, t) for t in query["climate"]])
        cf = evidence_cf_from_membership(m, rules.RULE_CF["climate"])
        evidences.append(cf)
        details["climate"] = (m, cf)
        include_memberships["climate"] = m
    elif query.get("travel_months"):
        # Even without an explicit climate preference, knowing the travel month
        # lets us give a small nudge toward cities with pleasant weather at that
        # time of year. We score against "mild" as a neutral baseline and scale
        # the influence down to 30 % of its normal weight so it only breaks ties
        # rather than dominating the ranking.
        temp = _effective_temp(dest, query)
        m = fuzzy.climate_membership(temp, "mild")
        cf = evidence_cf_from_membership(m, rules.RULE_CF["climate"]) * 0.3
        evidences.append(cf)
        details["seasonal_temp"] = (m, cf)
    if query.get("climate_exclude"):
        temp = _effective_temp(dest, query)
        m = fuzzy.fuzzy_or([fuzzy.climate_membership(temp, t) for t in query["climate_exclude"]])
        cf = _negative_cf_from_membership(m, rules.RULE_CF["climate"])
        evidences.append(cf)
        details["climate_exclude"] = (m, cf)

    budget_preferences = _effective_budget_preferences(query)
    if budget_preferences:
        m = fuzzy.fuzzy_or([fuzzy.budget_membership(dest["budget_level"], lvl) for lvl in budget_preferences])
        cf = evidence_cf_from_membership(m, rules.RULE_CF["budget"])
        evidences.append(cf)
        details["budget"] = (m, cf)
        include_memberships["budget"] = m
    if query.get("budget_exclude"):
        m = fuzzy.fuzzy_or([fuzzy.budget_membership(dest["budget_level"], lvl) for lvl in query["budget_exclude"]])
        cf = _negative_cf_from_membership(m, rules.RULE_CF["budget"])
        evidences.append(cf)
        details["budget_exclude"] = (m, cf)

    if query.get("duration"):
        m = fuzzy.fuzzy_or([fuzzy.duration_membership(dest["ideal_durations"], dur) for dur in query["duration"]])
        cf = evidence_cf_from_membership(m, rules.RULE_CF["duration"])
        evidences.append(cf)
        details["duration"] = (m, cf)
        include_memberships["duration"] = m
    if query.get("duration_exclude"):
        m = fuzzy.fuzzy_or([fuzzy.duration_membership(dest["ideal_durations"], dur) for dur in query["duration_exclude"]])
        cf = _negative_cf_from_membership(m, rules.RULE_CF["duration"])
        evidences.append(cf)
        details["duration_exclude"] = (m, cf)

    lifestyle = query.get("lifestyle") or {}
    lifestyle_fit = None
    if lifestyle:
        dims = [d for d in lifestyle.keys() if d in dest]
        if dims:
            user_vec = [lifestyle[d] for d in dims]
            city_vec = [dest[d] for d in dims]
            cos = cosine_similarity(user_vec, city_vec)
            strength = (sum(city_vec) / len(city_vec)) / 5.0
            lifestyle_fit = cos * strength
            cf = evidence_cf_from_membership(lifestyle_fit, rules.RULE_CF["lifestyle"])
            evidences.append(cf)
            details["lifestyle"] = (lifestyle_fit, cf)
    if query.get("lifestyle_exclude"):
        dims = [d for d in query["lifestyle_exclude"] if d in dest]
        if dims:
            m = max(dest[d] / 5.0 for d in dims)
            cf = _negative_cf_from_membership(m, rules.RULE_CF["lifestyle"])
            evidences.append(cf)
            details["lifestyle_exclude"] = (m, cf)

    confidence = cf_combine_many(evidences) if evidences else 0.0

    # Confidence gate: an explicitly requested criterion with zero membership is
    # a hard miss. It should not still receive a very high confidence only from
    # unrelated lifestyle or budget evidence.
    hard_failures = [name for name, m in include_memberships.items() if m <= 0.0]
    if hard_failures and confidence > 0.35:
        confidence = 0.35

    return {
        "confidence": confidence,
        "details": details,
        "lifestyle_fit": lifestyle_fit,
        "criteria_given": bool(evidences),
        "hard_failures": hard_failures,
    }


def build_working_memory(query):
    # Turn the query into symbolic FACTS for the forward-chaining advisory rules.
    facts = set()
    for level in query.get("budget") or []:
        facts.add("budget:" + level)
    for term in query.get("climate") or []:
        facts.add("climate:" + term)
    for dur in query.get("duration") or []:
        facts.add("duration:" + dur)
    include = query.get("regions_include") or []
    if include:
        facts.add("regions:specified")
    if len(include) >= 2:
        facts.add("regions:multiple")
    # A lifestyle dimension is a strong "want" only if rated 4 or 5.
    for dim, weight in (query.get("lifestyle") or {}).items():
        if weight >= 4:
            facts.add("wants:" + dim)
    return facts


def forward_chain(initial_facts, advisory_rules):
    # FORWARD CHAINING. Start from known facts; repeatedly fire every rule whose
    # "if" facts are ALL present and whose "then" is new; add that fact. Loop
    # until a pass adds nothing (fixpoint). New facts can satisfy further rules,
    # so rules CHAIN. Higher-priority rules are tried first (conflict resolution).
    # This matches the course definition exactly: data-driven inference, run to a
    # fixpoint, with PRIORITY (salience) as the conflict-resolution strategy.
    facts = set(initial_facts)
    fired = []
    messages = []
    ordered = sorted(advisory_rules, key=lambda r: r.get("priority", 0), reverse=True)
    changed = True
    while changed:
        changed = False
        for rule in ordered:
            if rule["then"] in facts:
                continue
            if all(cond in facts for cond in rule["if"]):
                facts.add(rule["then"])
                fired.append(rule["id"])
                if rule.get("message"):
                    messages.append((rule.get("priority", 0), rule["message"]))
                changed = True
    messages = [m for _, m in sorted(messages, key=lambda x: -x[0])]
    return facts, fired, messages


def recommend(destinations, query, top_n=5):
    # 1) crisp filter, with a FALLBACK: if the user named regions/countries but
    # none match, we drop the include filter (still honouring exclusions) and
    # flag that we broadened the search. 2) score, 3) sort, 4) advisory rules
    # using facts from BOTH the request and the results.
    #
    # NEW: countries_include / countries_exclude work exactly like their region
    # counterparts but filter on dest["country"] instead of dest["region"].
    # Country filtering is checked FIRST (more specific), then region filtering.
    # If a country filter is active the region filter is skipped for that dest
    # because the country already implies the region.
    include = query.get("regions_include") or []
    exclude = query.get("regions_exclude") or []
    countries_include = query.get("countries_include") or []
    countries_exclude = query.get("countries_exclude") or []

    def build_pool(apply_include):
        pool = []
        for dest in destinations:
            # Country exclusion: drop if the city's country is in the exclude list.
            if dest["country"] in countries_exclude:
                continue
            # Region exclusion: drop if the city's region is in the exclude list.
            if dest["region"] in exclude:
                continue
            if apply_include:
                # If a country include list is active, use it as the primary
                # filter (exact match on country). A city that passes the country
                # filter is automatically included regardless of region filter,
                # because "I want Japan" implies the right region already.
                if countries_include:
                    if dest["country"] not in countries_include:
                        continue
                elif include and dest["region"] not in include:
                    # No country filter — fall back to the normal region filter.
                    continue
            pool.append(dest)
        return pool

    broadened = False
    pool = build_pool(True)
    # FALLBACK: if include filters produced an empty pool, broaden the search.
    if (include or countries_include) and not pool:
        pool = build_pool(False)
        broadened = True
    scored = []
    for dest in pool:
        result = score_destination(dest, query)
        scored.append((result["confidence"], dest, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:top_n]
    # Facts about the request PLUS facts about the outcome, so advice reacts to
    # what actually happened.
    facts = build_working_memory(query)
    if broadened:
        facts.add("results:broadened")
    if not top:
        facts.add("results:empty")
    elif top[0][0] < 0.3:
        facts.add("results:weak")
    _, fired, messages = forward_chain(facts, rules.ADVISORY_RULES)
    return {"pool_size": len(pool), "results": top, "advisories": messages, "fired_rules": fired, "broadened": broadened}


def explain(dest, result, query):
    # EXPLANATION FACILITY: a human sentence justifying the pick, now showing the
    # membership AND the signed certainty factor of every piece of evidence, plus
    # the final combined confidence (full transparency of the reasoning).
    parts = []
    for name, (membership, cf) in result["details"].items():
        sign = "+" if cf >= 0 else ""
        parts.append(name + " " + str(round(membership, 2)) + " (cf " + sign + str(round(cf, 2)) + ")")
    components = ", ".join(parts) if parts else "no preferences given yet"
    drivers = []
    for dim, weight in (query.get("lifestyle") or {}).items():
        if weight >= 4 and dest[dim] >= 4:
            drivers.append(dim)
    driver_text = (" | strong on " + ", ".join(drivers)) if drivers else ""
    return "confidence " + str(round(result["confidence"], 2)) + " | " + components + driver_text


# ----------------------------------------------------------------------------
# HUMAN-FRIENDLY EXPLANATION FACILITY (used by the GUI "Why this pick?" panel).
# explain() above is the TECHNICAL trace (memberships + signed CFs) kept for the
# CLI and tests. explain_human() is a SEPARATE, additional function that hides
# every raw number and turns the reasoning into a few plain-English bullets, so
# a non-technical user understands *why* a city was suggested.
# ----------------------------------------------------------------------------
# Human phrasing for the requested trip length (prose only; the engine itself
# never needs words). Kept local so inference stays independent of the chatbot.
_DURATION_HUMAN = {
    "Day trip": "a day trip",
    "Weekend": "a weekend",
    "Short trip": "a short trip",
    "One week": "a one-week stay",
    "Long trip": "a long trip",
}


def _quality_word(membership):
    # Translate a fuzzy membership (0..1) into a linguistic term. This is the
    # inverse of fuzzification taught in class - a small "defuzzification into
    # words" - so the user reads "great match" instead of a number like 0.93.
    if membership >= 0.9:
        return "great match"
    if membership >= 0.7:
        return "good match"
    if membership >= 0.5:
        return "partial match"
    return "weak match"


def explain_human(dest, score, query):
    # Return a list of short, plain-English bullet points (no raw numbers) that
    # justify this pick. Rules:
    #   - skip any criterion whose |CF| < 0.05 (negligible evidence)
    #   - skip "_exclude" entries whose membership ~ 0 (the exclusion had no
    #     effect, so it is not worth mentioning)
    #   - any criterion with CF <= -0.1 is flagged as "not an ideal fit"
    #   - keep at most the 4 most influential bullets, ranked by |CF|
    details = score.get("details", {})
    bullets = []  # list of (abs_cf, text); we sort by abs_cf and keep the top 4

    # --- CLIMATE (fuzzy temperature match) ---
    # NEW: if travel months were given, the membership was computed against the
    # seasonal temperature rather than the yearly average, so we label the bullet
    # accordingly so the user can see which figure was actually used.
    if "climate" in details:
        m, cf = details["climate"]
        if cf <= -0.1:
            bullets.append((abs(cf), "⚠️ Climate — not an ideal fit"))
        elif abs(cf) >= 0.05:
            label = dest.get("climate", "")
            months = query.get("travel_months") or []
            if months:
                # Show the seasonal temperature rather than the yearly average
                # so the explanation matches what the engine actually scored on.
                temp = kb.seasonal_avg_temp(dest["avg_temp_monthly"], months)
                temp_str = (str(round(temp, 1)) + "°C for your travel period") if temp is not None else label
            else:
                temp = dest.get("avg_temp_yearly")
                temp_str = str(temp) + "°C yearly average"
            bullets.append((abs(cf), "🌡 Climate — " + _quality_word(m)
                            + " (" + label + ", " + temp_str + ")"))

    # --- SEASONAL TEMP NUDGE (travel months given but no explicit climate pref) ---
    # NEW: when only a travel month was given (no explicit climate preference),
    # score_destination() adds a small "seasonal_temp" entry. We surface it here
    # as a lightweight note so the user understands why seasonality played a role.
    if "seasonal_temp" in details:
        m, cf = details["seasonal_temp"]
        if abs(cf) >= 0.05:
            temp = kb.seasonal_avg_temp(dest["avg_temp_monthly"], query.get("travel_months") or [])
            temp_str = (str(round(temp, 1)) + "°C") if temp is not None else "n/a"
            bullets.append((abs(cf), "🌡 Seasonal temperature — " + temp_str + " during your travel period"))

    # --- BUDGET (ordinal fuzzy match) ---
    if "budget" in details:
        m, cf = details["budget"]
        level = dest.get("budget_level", "")
        if cf <= -0.1:
            bullets.append((abs(cf), "⚠️ Budget — not an ideal fit (" + level + ")"))
        elif abs(cf) >= 0.05:
            phrase = "perfect fit" if m >= 0.9 else _quality_word(m)
            bullets.append((abs(cf), "💰 Budget — " + phrase + " (" + level + " as requested)"))

    # --- DURATION (ordinal fuzzy match) ---
    if "duration" in details:
        m, cf = details["duration"]
        wanted = query.get("duration") or []
        human = _DURATION_HUMAN.get(wanted[0], "your trip length") if wanted else "your trip length"
        if cf <= -0.1:
            bullets.append((abs(cf), "⚠️ Duration — not an ideal fit"))
        elif abs(cf) >= 0.05:
            bullets.append((abs(cf), "🗓 Duration — " + _quality_word(m)
                            + " (suits " + human + ")"))

    # --- LIFESTYLE / interests (cosine taste match) ---
    if "lifestyle" in details:
        m, cf = details["lifestyle"]
        if cf <= -0.1:
            bullets.append((abs(cf), "⚠️ Interests — not an ideal fit"))
        elif abs(cf) >= 0.05:
            wanted = query.get("lifestyle") or {}
            # Name the specific dimensions that are BOTH wanted (>=4) and strong
            # in the city (>=4), which is the most concrete thing to tell a user.
            strong = [d for d, w in wanted.items() if w >= 4 and dest.get(d, 0) >= 4]
            if strong:
                bullets.append((abs(cf), "🎯 Interests — strong overlap on " + ", ".join(strong)))
            else:
                bullets.append((abs(cf), "🎯 Interests — " + _quality_word(m) + " on what you enjoy"))

    # --- EXCLUSIONS: only mention when they actually bit (membership not ~0) ---
    exclude_labels = {
        "climate_exclude": "Climate",
        "budget_exclude": "Budget",
        "duration_exclude": "Duration",
        "lifestyle_exclude": "Interests",
    }
    for key, label in exclude_labels.items():
        if key in details:
            m, cf = details[key]
            if m >= 0.05 and cf <= -0.1:
                bullets.append((abs(cf), "⚠️ " + label + " — has some of what you wanted to avoid"))

    if not bullets:
        return ["A broad, balanced pick — no single criterion stood out."]
    bullets.sort(key=lambda b: b[0], reverse=True)
    return [text for _, text in bullets[:4]]