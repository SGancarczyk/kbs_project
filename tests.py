# tests.py
# ----------------------------------------------------------------------------
# TEST SUITE for the Travel Destination KBS. Run it with:  python3 tests.py
# Each check() prints PASS or FAIL. This file is both our development safety net
# and the "tests that can be performed" the project README must point to. It
# also doubles as the "experimental design / validation" evidence for the
# written report: it shows the language understanding, the fuzzy maths, the
# certainty-factor maths, and the recommender behaviour all do what we claim.
# ----------------------------------------------------------------------------
import knowledge_base as kb
import nlp
import fuzzy
import inference as inf
PASSED = 0
FAILED = 0
FAILS = []
def check(name, condition):
    # Record and print the outcome of one assertion.
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print("  [PASS]", name)
    else:
        FAILED += 1
        FAILS.append(name)
        print("  [FAIL]", name)
def approx(a, b, tol=0.02):
    # Floats are rarely exactly equal, so we compare within a small tolerance.
    return abs(a - b) <= tol
print("=== 1. KNOWLEDGE BASE: the data loads and is clean ===")
dests = kb.load_destinations()
check("loads all 560 destinations", len(dests) == 560)
check("lifestyle scores are integers", all(isinstance(d["culture"], int) for d in dests))
check("every region is a known region", all(d["region"] in kb.REGIONS for d in dests))
check("every budget level is valid", all(d["budget_level"] in kb.BUDGET_LEVELS for d in dests))
check("derived climate label is valid", all(d["climate"] in kb.CLIMATES for d in dests))
check("derived yearly temperature is a number", all(isinstance(d["avg_temp_yearly"], float) for d in dests))
check("ideal_durations parsed into a list", all(isinstance(d["ideal_durations"], list) for d in dests))
print("=== 2. NLP: language understanding (typos + negation + multi-value) ===")
check("edit_distance kitten->sitting = 3", nlp.edit_distance("kitten", "sitting") == 3)
check("edit_distance chaep->cheap = 2", nlp.edit_distance("chaep", "cheap") == 2)
check("edit_distance cuisne->cuisine = 1", nlp.edit_distance("cuisne", "cuisine") == 1)
check("lemmatize beaches -> beach", nlp.lemmatize("beaches") == "beach")
r = nlp.extract("hello europe, not go asia.. also africa not bad", kb.CONTINENT_SYNONYMS)
check("sentence -> includes europe + africa", set(r["include"]) == {"europe", "africa"})
check("sentence -> excludes asia (negation works)", r["exclude"] == ["asia"])
r2 = nlp.extract("avoid nightlife, i love culture and seclusion", kb.LIFESTYLE_SYNONYMS)
check("lifestyle -> includes culture + seclusion", set(r2["include"]) == {"culture", "seclusion"})
check("lifestyle -> excludes nightlife", r2["exclude"] == ["nightlife"])
r3 = nlp.extract("i want a chaep trip", kb.BUDGET_SYNONYMS)
check("typo 'chaep' -> Budget (spell-correction)", r3["include"] == ["Budget"])
r4 = nlp.extract("a long weekend please", kb.DURATION_SYNONYMS)
check("duration -> Long trip + Weekend", set(r4["include"]) == {"Long trip", "Weekend"})
print("=== 3. FUZZY LOGIC: graded membership, no hard cutoffs ===")
cm = fuzzy.climate_memberships(21)
check("21C is mild ~0.5", approx(cm["mild"], 0.5))
check("21C is also warm ~0.43 (overlap)", approx(cm["warm"], 0.43))
check("21C is not cold (0.0)", cm["cold"] == 0.0)
check("28C is fully warm (1.0)", fuzzy.climate_memberships(28)["warm"] == 1.0)
check("8C is fully cold (1.0)", fuzzy.climate_memberships(8)["cold"] == 1.0)
check("all memberships stay within [0,1]", all(0.0 <= v <= 1.0 for t in range(-10, 45) for v in fuzzy.climate_memberships(t).values()))
check("budget exact match = 1.0", fuzzy.budget_membership("Budget", "Budget") == 1.0)
check("budget neighbour = 0.4 (partial)", fuzzy.budget_membership("Mid-range", "Budget") == 0.4)
check("budget far = 0.0", fuzzy.budget_membership("Luxury", "Budget") == 0.0)
check("duration exact = 1.0", fuzzy.duration_membership(["Short trip", "One week"], "Short trip") == 1.0)
check("duration adjacent = 0.5", fuzzy.duration_membership(["Short trip", "One week"], "Weekend") == 0.5)
check("fuzzy AND = min", fuzzy.fuzzy_and([0.5, 0.43, 1.0]) == 0.43)
check("fuzzy OR = max", fuzzy.fuzzy_or([0.5, 0.43, 1.0]) == 1.0)
print("=== 4. INFERENCE MATHS: cosine + certainty factors ===")
check("cosine of identical vectors = 1.0", approx(inf.cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0))
check("cosine of perpendicular vectors = 0.0", inf.cosine_similarity([1, 0], [0, 1]) == 0.0)
check("cosine ignores length (same direction = 1.0)", approx(inf.cosine_similarity([1, 1], [2, 2]), 1.0))
check("CF propagation 0.9 x 0.5 = 0.45", approx(inf.cf_propagate(0.9, 0.5), 0.45))
check("CF combine 0.6 & 0.7 = 0.88 (both positive)", approx(inf.cf_combine(0.6, 0.7), 0.88))
check("CF combine 0.7 & -0.5 = 0.4 (mixed, slide)", approx(inf.cf_combine(0.7, -0.5), 0.4))
check("CF combine -0.5 & -0.6 = -0.8 (both neg, slide)", approx(inf.cf_combine(-0.5, -0.6), -0.8))
check("CF combine many [0.9,0.85] = 0.985", approx(inf.cf_combine_many([0.9, 0.85]), 0.985))
check("evidence CF: full match m=1 -> +ruleCF", approx(inf.evidence_cf_from_membership(1.0, 0.8), 0.8))
check("evidence CF: full miss m=0 -> -ruleCF (NEGATIVE evidence)", approx(inf.evidence_cf_from_membership(0.0, 0.8), -0.8))
check("evidence CF: half match m=0.5 -> 0 (neutral)", approx(inf.evidence_cf_from_membership(0.5, 0.8), 0.0))
print("=== 5. RECOMMENDER: filtering, ranking, edge cases ===")
q = {"regions_include": ["europe"], "climate": ["warm"], "budget": ["Budget"], "duration": ["One week"], "lifestyle": {"culture": 5, "cuisine": 5}}
out = inf.recommend(dests, q, top_n=5)
check("crisp filter keeps only europe", all(d["region"] == "europe" for _, d, _ in out["results"]))
check("returns exactly 5 results", len(out["results"]) == 5)
check("every confidence is in [-1,1]", all(-1.0 <= c <= 1.0 for c, _, _ in out["results"]))
check("results sorted by confidence desc", all(out["results"][i][0] >= out["results"][i + 1][0] for i in range(len(out["results"]) - 1)))
qx = {"regions_exclude": ["asia"], "lifestyle": {"nature": 5}}
outx = inf.recommend(dests, qx, top_n=10)
check("excluded region never appears", all(d["region"] != "asia" for _, d, _ in outx["results"]))
oute = inf.recommend(dests, {}, top_n=5)
check("empty query -> whole pool (560)", oute["pool_size"] == 560)
check("empty query -> still returns 5", len(oute["results"]) == 5)
print("=== 6. FORWARD CHAINING: advisory rules fire (incl. a chained rule) ===")
qc = {"regions_include": ["europe", "asia", "africa"], "climate": ["cold"], "budget": ["Budget", "Luxury"], "duration": ["Day trip"], "lifestyle": {"beaches": 5, "nightlife": 4, "seclusion": 5}}
outc = inf.recommend(dests, qc, top_n=3)
fired = set(outc["fired_rules"])
check("conflict_budget fired", "conflict_budget" in fired)
check("assume_midrange fired (CHAINED off conflict_budget)", "assume_midrange" in fired)
check("daytrip_multi fired", "daytrip_multi" in fired)
check("beach_cold fired", "beach_cold" in fired)
check("night_seclusion fired", "night_seclusion" in fired)
check("advisory messages were produced", len(outc["advisories"]) >= 4)
print("=== 7. UPGRADES: negative evidence, strength factor, fallback, input guards ===")
milan = next(d for d in dests if d["city"] == "Milan")
sb = inf.score_destination(milan, {"budget": ["Budget"]})
check("Luxury city vs Budget wish -> NEGATIVE budget CF", sb["details"]["budget"][1] < 0)
def life_conf(c1, c2):
    fake = {k: 3 for k in kb.LIFESTYLE_DIMENSIONS}
    fake["culture"] = c1
    fake["cuisine"] = c2
    return inf.score_destination(fake, {"lifestyle": {"culture": 5, "cuisine": 5}})["confidence"]
check("strength factor: strong (5,5) outranks weak (1,1)", life_conf(5, 5) > life_conf(1, 1))
check("strength factor: weak (1,1) is actively penalised (<0)", life_conf(1, 1) < 0)
fb = inf.recommend(dests, {"regions_include": ["atlantis"], "lifestyle": {"nature": 5}}, top_n=3)
check("unknown region triggers broadened fallback", fb["broadened"] is True)
check("fallback still returns results", len(fb["results"]) == 3)
check("budget guard: unknown level -> 0.0 (no crash)", fuzzy.budget_membership("Budget", "Unknown") == 0.0)
check("duration guard: unknown duration -> 0.0 (no crash)", fuzzy.duration_membership(["One week"], "Unknown") == 0.0)
print("=== 8. CHATBOT: 'skip all' jumps straight to recommendations ===")
from chatbot import TravelChatbot, render_response_text
bot = TravelChatbot(dests)
bot.respond("europe")
reply = bot.respond("skip all")
# A successful recommendation is now a structured dict (so the GUI can render
# cards); error/empty replies remain plain strings.
check("'skip all' produces recommendations", isinstance(reply, dict) and reply.get("type") == "recommendations")
check("'skip all' recommendation has 5 result cards", isinstance(reply, dict) and len(reply["results"]) == 5)
check("recommendation text renders 'top picks' for the CLI", "top picks" in render_response_text(reply))
check("expanded vocab: country name 'japan' -> asia", nlp.extract("i want to visit japan", kb.CONTINENT_SYNONYMS)["include"] == ["asia"])
check("expanded vocab: 'temples' lemmatised -> culture", "culture" in nlp.extract("i love ancient temples", kb.LIFESTYLE_SYNONYMS)["include"])

print("=== 9. PATCH VALIDATION: safer NLP, exclusions, gates ===")
check("NLP: cheap does not become warm", nlp.extract("cheap", kb.CLIMATE_SYNONYMS)["include"] == [])
check("NLP: mild does not become Mid-range", nlp.extract("mild", kb.BUDGET_SYNONYMS)["include"] == [])
check("NLP: south does not become Long trip", nlp.extract("south", kb.DURATION_SYNONYMS)["include"] == [])
check("NLP: mid range is budget", nlp.extract("mid range", kb.BUDGET_SYNONYMS)["include"] == ["Mid-range"])
check("NLP: mid range does not become Europe", nlp.extract("mid range", kb.CONTINENT_SYNONYMS)["include"] == [])
check("NLP: short trip is duration", nlp.extract("short trip", kb.DURATION_SYNONYMS)["include"] == ["Short trip"])
check("NLP: short trip does not become beaches", nlp.extract("short trip", kb.LIFESTYLE_SYNONYMS)["include"] == [])
check("NLP: short trip does not become warm", nlp.extract("short trip", kb.CLIMATE_SYNONYMS)["include"] == [])
check("NLP: south america only maps to South America", nlp.extract("south america", kb.CONTINENT_SYNONYMS)["include"] == ["south_america"])
check("NLP: latin america only maps to South America", nlp.extract("latin america", kb.CONTINENT_SYNONYMS)["include"] == ["south_america"])
check("NLP: north america phrase maps correctly", nlp.extract("north america", kb.CONTINENT_SYNONYMS)["include"] == ["north_america"])
check("NLP: new zealand phrase maps correctly", nlp.extract("new zealand", kb.CONTINENT_SYNONYMS)["include"] == ["oceania"])
check("NLP: united states phrase maps correctly", nlp.extract("united states", kb.CONTINENT_SYNONYMS)["include"] == ["north_america"])
check("NLP: middle east phrase maps correctly", nlp.extract("middle east", kb.CONTINENT_SYNONYMS)["include"] == ["middle_east"])
rneg = nlp.extract("not africa or asia", kb.CONTINENT_SYNONYMS)
check("NLP: not Africa or Asia excludes both", set(rneg["exclude"]) == {"africa", "asia"} and rneg["include"] == [])

bot_empty = TravelChatbot(dests)
empty_reply = bot_empty.respond("go")
check("chatbot: empty go asks for a preference", "at least one preference" in empty_reply)
bot_ex = TravelChatbot(dests)
bot_ex.greeting()
bot_ex.respond("not expensive and avoid nightlife")
check("chatbot: not expensive creates budget exclusion", bot_ex.query["budget_exclude"] == ["Luxury"])
check("chatbot: avoid nightlife creates lifestyle exclusion", bot_ex.query["lifestyle_exclude"] == ["nightlife"])

luxury_city = next(d for d in dests if d["budget_level"] == "Luxury")
excluded_budget_score = inf.score_destination(luxury_city, {"budget_exclude": ["Luxury"]})
check("inference: excluded Luxury produces negative evidence", excluded_budget_score["details"]["budget_exclude"][1] < 0)
night_city = {k: 3 for k in kb.LIFESTYLE_DIMENSIONS}
night_city["nightlife"] = 5
quiet_city = {k: 3 for k in kb.LIFESTYLE_DIMENSIONS}
quiet_city["nightlife"] = 1
check("inference: avoiding nightlife penalises high nightlife more", inf.score_destination(night_city, {"lifestyle_exclude": ["nightlife"]})["confidence"] < inf.score_destination(quiet_city, {"lifestyle_exclude": ["nightlife"]})["confidence"])
lviv = next(d for d in dests if d["city"] == "Lviv")
gated = inf.score_destination(lviv, {"climate": ["warm"], "budget": ["Budget"], "duration": ["Weekend"], "lifestyle": {"culture": 5, "cuisine": 5}})
check("inference: hard zero climate match is confidence gated", gated["confidence"] <= 0.35 and "climate" in gated["hard_failures"])
conflict_out = inf.recommend(dests, {"budget": ["Budget", "Luxury"]}, top_n=10)
check("inference: Budget + Luxury conflict actually prefers Mid-range", all(d["budget_level"] == "Mid-range" for _, d, _ in conflict_out["results"]))

# Extra regression checks after manual review: negation must not leak across unrelated slots.
bot_leak = TravelChatbot(dests)
bot_leak.greeting()
bot_leak.respond("not expensive avoid nightlife south america short trip")
check("chatbot: not expensive does not exclude South America", bot_leak.query["regions_include"] == ["south_america"] and bot_leak.query["regions_exclude"] == [])
check("chatbot: avoid nightlife does not exclude Short trip", bot_leak.query["duration"] == ["Short trip"] and bot_leak.query["duration_exclude"] == [])
check("chatbot: not expensive still excludes Luxury", bot_leak.query["budget_exclude"] == ["Luxury"])
check("chatbot: avoid nightlife still excludes nightlife", bot_leak.query["lifestyle_exclude"] == ["nightlife"])

bot_typo = TravelChatbot(dests)
bot_typo.greeting()
bot_typo.respond("chaep asia cuisine")
check("chatbot: all-in-one typo chaep still fills Budget", bot_typo.query["budget"] == ["Budget"])

print("=== SUMMARY ===")
print("  passed:", PASSED, "| failed:", FAILED)
if FAILED:
    print("  failing checks:", FAILS)
print("  external libraries used: NONE (pure Python standard library). All logic")
print("  (NLP pipeline, fuzzy, certainty factors, cosine, rules) is our own.")
