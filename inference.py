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
#     "lifestyle_exclude": [...] }
# ----------------------------------------------------------------------------
import math
import fuzzy
import rules
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
    # opinion"). Order does not matter for the MYCIN formula.
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


def score_destination(dest, query):
    # Score ONE destination. Included preferences become signed CF evidence.
    # Explicit exclusions become negative evidence. Missing criteria add no
    # evidence, so absence is not treated as a perfect match.
    evidences = []
    details = {}
    include_memberships = {}

    if query.get("climate"):
        m = fuzzy.fuzzy_or([fuzzy.climate_membership(dest["avg_temp_yearly"], t) for t in query["climate"]])
        cf = evidence_cf_from_membership(m, rules.RULE_CF["climate"])
        evidences.append(cf)
        details["climate"] = (m, cf)
        include_memberships["climate"] = m
    if query.get("climate_exclude"):
        m = fuzzy.fuzzy_or([fuzzy.climate_membership(dest["avg_temp_yearly"], t) for t in query["climate_exclude"]])
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
    # 1) crisp region filter, with a FALLBACK: if the user named regions but none
    # match, we drop the include filter (still honouring excluded regions) and
    # flag that we broadened the search. 2) score, 3) sort, 4) advisory rules
    # using facts from BOTH the request and the results.
    include = query.get("regions_include") or []
    exclude = query.get("regions_exclude") or []
    def build_pool(apply_include):
        pool = []
        for dest in destinations:
            if apply_include and include and dest["region"] not in include:
                continue
            if dest["region"] in exclude:
                continue
            pool.append(dest)
        return pool
    broadened = False
    pool = build_pool(True)
    if include and not pool:
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
