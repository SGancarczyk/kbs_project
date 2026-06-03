import math
import fuzzy
import rules
import knowledge_base as kb
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
def cf_propagate(rule_cf, evidence):
    return rule_cf * evidence
def evidence_cf_from_membership(membership, rule_cf):
    return rule_cf * (2 * membership - 1)
def cf_combine(cf1, cf2):
    if cf1 >= 0 and cf2 >= 0:
        return cf1 + cf2 * (1 - cf1)
    if cf1 <= 0 and cf2 <= 0:
        return cf1 + cf2 * (1 + cf1)
    denominator = 1 - min(abs(cf1), abs(cf2))
    if denominator == 0:
        return 0.0
    return (cf1 + cf2) / denominator
def cf_combine_many(cfs):
    total = 0.0
    for cf in cfs:
        total = cf_combine(total, cf)
    return total
def _negative_cf_from_membership(membership, rule_cf):
    return -rule_cf * membership
def _effective_budget_preferences(query):
    budgets = list(query.get("budget") or [])
    if "Budget" in budgets and "Luxury" in budgets:
        return ["Mid-range"]
    return budgets
def _effective_temp(dest, query):
    months = query.get("travel_months") or []
    if months:
        seasonal = kb.seasonal_avg_temp(dest["avg_temp_monthly"], months)
        if seasonal is not None:
            return seasonal
    return dest["avg_temp_yearly"]
def score_destination(dest, query):
    importance = query.get("importance") or {}
    RCF = {k: max(0.05, min(0.97, v * importance.get(k, 1.0)))
           for k, v in rules.RULE_CF.items()}
    evidences = []
    details = {}
    include_memberships = {}
    if query.get("climate"):
        temp = _effective_temp(dest, query)
        m = fuzzy.fuzzy_or([fuzzy.climate_membership(temp, t) for t in query["climate"]])
        cf = evidence_cf_from_membership(m, RCF["climate"])
        evidences.append(cf)
        details["climate"] = (m, cf)
        include_memberships["climate"] = m
    elif query.get("travel_months"):
        temp = _effective_temp(dest, query)
        m = fuzzy.climate_membership(temp, "mild")
        cf = evidence_cf_from_membership(m, RCF["climate"]) * 0.3
        evidences.append(cf)
        details["seasonal_temp"] = (m, cf)
    if query.get("climate_exclude"):
        temp = _effective_temp(dest, query)
        m = fuzzy.fuzzy_or([fuzzy.climate_membership(temp, t) for t in query["climate_exclude"]])
        cf = _negative_cf_from_membership(m, RCF["climate"])
        evidences.append(cf)
        details["climate_exclude"] = (m, cf)
    budget_preferences = _effective_budget_preferences(query)
    if budget_preferences:
        m = fuzzy.fuzzy_or([fuzzy.budget_membership(dest["budget_level"], lvl) for lvl in budget_preferences])
        cf = evidence_cf_from_membership(m, RCF["budget"])
        evidences.append(cf)
        details["budget"] = (m, cf)
        include_memberships["budget"] = m
    if query.get("budget_exclude"):
        m = fuzzy.fuzzy_or([fuzzy.budget_membership(dest["budget_level"], lvl) for lvl in query["budget_exclude"]])
        cf = _negative_cf_from_membership(m, RCF["budget"])
        evidences.append(cf)
        details["budget_exclude"] = (m, cf)
    if query.get("duration"):
        m = fuzzy.fuzzy_or([fuzzy.duration_membership(dest["ideal_durations"], dur) for dur in query["duration"]])
        cf = evidence_cf_from_membership(m, RCF["duration"])
        evidences.append(cf)
        details["duration"] = (m, cf)
        include_memberships["duration"] = m
    if query.get("duration_exclude"):
        m = fuzzy.fuzzy_or([fuzzy.duration_membership(dest["ideal_durations"], dur) for dur in query["duration_exclude"]])
        cf = _negative_cf_from_membership(m, RCF["duration"])
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
            total_weight = sum(user_vec)
            if total_weight > 0:
                weighted_strength = sum(u * c for u, c in zip(user_vec, city_vec)) / (5.0 * total_weight)
            else:
                weighted_strength = 0.0
            lifestyle_fit = cos * weighted_strength
            cf = evidence_cf_from_membership(lifestyle_fit, RCF["lifestyle"])
            evidences.append(cf)
            details["lifestyle"] = (lifestyle_fit, cf)
    if query.get("lifestyle_exclude"):
        dims = [d for d in query["lifestyle_exclude"] if d in dest]
        if dims:
            m = max(dest[d] / 5.0 for d in dims)
            cf = _negative_cf_from_membership(m, RCF["lifestyle"])
            evidences.append(cf)
            details["lifestyle_exclude"] = (m, cf)
    confidence = cf_combine_many(evidences) if evidences else 0.0
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
    for dim, weight in (query.get("lifestyle") or {}).items():
        if weight >= 4:
            facts.add("wants:" + dim)
    for dim in query.get("lifestyle_exclude") or []:
        facts.add("avoids:" + dim)
    if query.get("lifestyle_exclude"):
        facts.add("has:exclusions")
    if query.get("budget_exclude"):
        facts.add("has:exclusions")
    months = set(query.get("travel_months") or [])
    if months & {12, 1, 2}:
        facts.add("travel:winter")
    if months & {6, 7, 8}:
        facts.add("travel:summer")
    return facts
def forward_chain(initial_facts, advisory_rules):
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
    include = query.get("regions_include") or []
    exclude = query.get("regions_exclude") or []
    countries_include = query.get("countries_include") or []
    countries_exclude = query.get("countries_exclude") or []
    def build_pool(apply_include):
        pool = []
        for dest in destinations:
            if dest["country"] in countries_exclude:
                continue
            if dest["region"] in exclude:
                continue
            if apply_include and (countries_include or include):
                in_country = bool(countries_include) and dest["country"] in countries_include
                in_region = bool(include) and dest["region"] in include
                if not (in_country or in_region):
                    continue
            pool.append(dest)
        return pool
    broadened = False
    pool = build_pool(True)
    if (include or countries_include) and not pool:
        pool = build_pool(False)
        broadened = True
    scored = []
    for dest in pool:
        result = score_destination(dest, query)
        scored.append((result["confidence"], dest, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:top_n]
    facts = build_working_memory(query)
    if broadened:
        facts.add("results:broadened")
    if not top:
        facts.add("results:empty")
    elif top[0][0] < 0.3:
        facts.add("results:weak")
    if top and query.get("lifestyle_exclude"):
        for exc_dim in query["lifestyle_exclude"]:
            if any(dest.get(exc_dim, 0) >= 4 for _, dest, _ in top):
                facts.add("results:contain_avoided")
                break
    _, fired, messages = forward_chain(facts, rules.ADVISORY_RULES)
    return {"pool_size": len(pool), "results": top, "advisories": messages, "fired_rules": fired, "broadened": broadened}
def explain(dest, result, query):
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
_DURATION_HUMAN = {
    "Day trip": "a day trip",
    "Weekend": "a weekend",
    "Short trip": "a short trip",
    "One week": "a one-week stay",
    "Long trip": "a long trip",
}
def match_label(confidence):
    if confidence >= 0.85:
        return "Excellent match"
    if confidence >= 0.6:
        return "Great match"
    if confidence >= 0.35:
        return "Good match"
    if confidence >= 0.1:
        return "Fair match"
    if confidence >= -0.1:
        return "Loose match"
    return "Weak match"
def _quality_word(membership):
    if membership >= 0.9:
        return "great match"
    if membership >= 0.7:
        return "good match"
    if membership >= 0.5:
        return "partial match"
    return "weak match"
def explain_human(dest, score, query):
    details = score.get("details", {})
    bullets = []
    if "climate" in details:
        m, cf = details["climate"]
        if cf <= -0.1:
            bullets.append((abs(cf), "⚠️ Climate — not an ideal fit"))
        elif abs(cf) >= 0.05:
            label = dest.get("climate", "")
            months = query.get("travel_months") or []
            if months:
                temp = kb.seasonal_avg_temp(dest["avg_temp_monthly"], months)
                temp_str = (str(round(temp, 1)) + "°C for your travel period") if temp is not None else label
            else:
                temp = dest.get("avg_temp_yearly")
                temp_str = str(temp) + "°C yearly average"
            bullets.append((abs(cf), "🌡 Climate — " + _quality_word(m)
                            + " (" + label + ", " + temp_str + ")"))
    if "seasonal_temp" in details:
        m, cf = details["seasonal_temp"]
        if abs(cf) >= 0.05:
            temp = kb.seasonal_avg_temp(dest["avg_temp_monthly"], query.get("travel_months") or [])
            temp_str = (str(round(temp, 1)) + "°C") if temp is not None else "n/a"
            bullets.append((abs(cf), "🌡 Seasonal temperature — " + temp_str + " during your travel period"))
    if "budget" in details:
        m, cf = details["budget"]
        level = dest.get("budget_level", "")
        if cf <= -0.1:
            bullets.append((abs(cf), "⚠️ Budget — not an ideal fit (" + level + ")"))
        elif abs(cf) >= 0.05:
            phrase = "perfect fit" if m >= 0.9 else _quality_word(m)
            bullets.append((abs(cf), "💰 Budget — " + phrase + " (" + level + " as requested)"))
    if "duration" in details:
        m, cf = details["duration"]
        wanted = query.get("duration") or []
        human = _DURATION_HUMAN.get(wanted[0], "your trip length") if wanted else "your trip length"
        if cf <= -0.1:
            bullets.append((abs(cf), "⚠️ Duration — not an ideal fit"))
        elif abs(cf) >= 0.05:
            bullets.append((abs(cf), "🗓 Duration — " + _quality_word(m)
                            + " (suits " + human + ")"))
    if "lifestyle" in details:
        m, cf = details["lifestyle"]
        if cf <= -0.1:
            bullets.append((abs(cf), "⚠️ Interests — not an ideal fit"))
        elif abs(cf) >= 0.05:
            wanted = query.get("lifestyle") or {}
            strong = [d for d, w in wanted.items() if w >= 4 and dest.get(d, 0) >= 4]
            if strong:
                bullets.append((abs(cf), "🎯 Interests — strong overlap on " + ", ".join(strong)))
            else:
                bullets.append((abs(cf), "🎯 Interests — " + _quality_word(m) + " on what you enjoy"))
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
