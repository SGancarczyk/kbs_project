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
check("budget neighbour = 0.5 (one band off = neutral, matches duration)", fuzzy.budget_membership("Mid-range", "Budget") == 0.5)
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
check("recommendation text renders the picks header for the CLI", "Here are a few destinations" in render_response_text(reply))
check("country vocab: 'japan' -> Japan (country slot)", nlp.extract("i want to visit japan", kb.COUNTRY_SYNONYMS)["include"] == ["Japan"])
check("country_to_region: Japan -> asia", kb.country_to_region("Japan") == "asia")
check("'japan' is NOT in the continent vocab anymore", nlp.extract("i want to visit japan", kb.CONTINENT_SYNONYMS)["include"] == [])
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
check("NLP: 'swim' -> beaches (not only 'swimming')", nlp.extract("i would love to swim", kb.LIFESTYLE_SYNONYMS)["include"] == ["beaches"])
check("NLP: 'good weather' -> warm", nlp.extract("i want good weather", kb.CLIMATE_SYNONYMS)["include"] == ["warm"])
check("NLP: 'nice weather' -> warm", nlp.extract("somewhere with nice weather", kb.CLIMATE_SYNONYMS)["include"] == ["warm"])
check("NLP: 'swim so a good weather' -> beaches + warm", set(nlp.extract("I would love to swim so a good weather", kb.LIFESTYLE_SYNONYMS)["include"]) == {"beaches"} and nlp.extract("I would love to swim so a good weather", kb.CLIMATE_SYNONYMS)["include"] == ["warm"])
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

print("=== 10. REVIEW FIXES: spell guard, union, sanitize, NLTK setup, explanation ===")
# edit_distance is a real helper (used by the conservative corrector).
check("edit_distance is callable and correct", nlp.edit_distance("flaw", "lawn") == 2)
# The corrector fixes a transposition typo but NOT a one-letter substitution,
# so common words never get rewritten into lookalike keywords.
check("spell guard: transposition 'chaep' -> Budget", nlp.extract("chaep", kb.BUDGET_SYNONYMS)["include"] == ["Budget"])
check("spell guard: 'short' is NOT corrected to 'sport'/adventure", nlp.extract("short", kb.LIFESTYLE_SYNONYMS)["include"] == [])
check("spell guard: 'cheap' is NOT corrected to 'crisp'/cold", nlp.extract("cheap", kb.CLIMATE_SYNONYMS)["include"] == [])
# Country + region given together = UNION, not country silently overriding region.
union_q = {"countries_include": ["Japan"], "regions_include": ["europe"], "lifestyle": {"culture": 5}}
union_out = inf.recommend(dests, union_q, top_n=560)
union_regions = {d["region"] for _, d, _ in union_out["results"]}
union_countries = {d["country"] for _, d, _ in union_out["results"]}
check("union: Japanese cities are kept", "Japan" in union_countries)
check("union: European cities are kept too", "europe" in union_regions)
check("union: nothing outside Japan-or-Europe leaks in", all((d["country"] == "Japan" or d["region"] == "europe") for _, d, _ in union_out["results"]))
# Sanitize: if the user jumps to results mid-rating, a -1 placeholder (a dim
# picked but not yet rated) must NOT reach the cosine vector, where it would be
# an invalid negative weight. _format_recommendation() must clean it up first.
bot_san = TravelChatbot(dests)
bot_san.query["regions_include"] = ["europe"]
bot_san.query["lifestyle"] = {"culture": -1, "nature": -1}   # picked, unrated
san_reply = bot_san._format_recommendation()
check("sanitize: no -1 placeholder survives into the query", all(w != -1 for w in bot_san.query["lifestyle"].values()))
check("sanitize: unrated picked dims default to a valid 1-5 weight", all(1 <= w <= 5 for w in bot_san.query["lifestyle"].values()))
check("sanitize: results still produced after jumping mid-rating", isinstance(san_reply, dict) and san_reply.get("type") == "recommendations")
# NLTK data setup helper exists and runs cleanly (data already present here).
check("ensure_nltk_data() runs without raising", (nlp.ensure_nltk_data() or True))
# Explanation facility returns a readable, transparent justification string.
some_dest = next(d for d in dests if d["region"] == "europe")
exp_score = inf.score_destination(some_dest, {"climate": ["warm"], "budget": ["Mid-range"]})
exp_text = inf.explain(some_dest, exp_score, {"climate": ["warm"], "budget": ["Mid-range"]})
check("explanation: explain() returns a confidence string", isinstance(exp_text, str) and "confidence" in exp_text)
check("explanation: explain_human() returns plain-English bullets", isinstance(inf.explain_human(some_dest, exp_score, {"climate": ["warm"]}), list))
# Conversational lifestyle: free text (not just the numbered menu) populates the
# interest vector, while a non-lifestyle answer adds nothing.
bot_free = TravelChatbot(dests)
bot_free.greeting()
bot_free.respond("europe, i love food and culture")
check("free-text lifestyle: 'food and culture' populate interests", set(bot_free.query["lifestyle"]) >= {"cuisine", "culture"})
check("free-text lifestyle: a volunteered interest gets a strong weight", bot_free.query["lifestyle"].get("culture") == 4)
bot_free2 = TravelChatbot(dests)
bot_free2.greeting()
bot_free2.respond("somewhere in europe")
check("free-text lifestyle: a non-lifestyle answer adds no interests", bot_free2.query["lifestyle"] == {})

print("=== 11. CUSTOMER-FACING OUTPUT: no technical maths, >= 15 distinct responses ===")
# The end user must never see certainty factors or raw membership numbers.
import random as _rnd
_rnd.seed(0)
bot_disp = TravelChatbot(dests)
bot_disp.greeting()
bot_disp.respond("a cheap warm week in asia, i love food and culture")
rec_reply = bot_disp.respond("go")
rec_text = render_response_text(rec_reply)
why_text = bot_disp.respond("why")
combined = rec_text + "\n" + why_text
check("recommendation text shows a friendly match label", "match" in rec_text.lower())
check("recommendation text hides certainty factors", "(cf " not in rec_text and "cf=" not in rec_text)
check("recommendation text hides raw membership numbers", "m=" not in rec_text)
check("'why' explanation hides certainty factors", "cf=" not in why_text and "(cf " not in why_text and "m=" not in why_text)
check("'why' still gives a real reason", len(why_text) > 30 and "why" in why_text.lower())

# At least 15 DIFFERENT responses (course requirement). We collect the distinct
# replies the bot can produce across its intents and conversational situations.
responses = set()
def _add(r):
    t = render_response_text(r)
    if isinstance(t, str) and t.strip():
        responses.add(t.strip())
b0 = TravelChatbot(dests)
_add(b0.greeting())                 # 1 greeting
_add(b0.help_text())                # 2 help
_add(b0.respond(""))                # 3 empty-input nudge
_add(b0.respond("askjdhfg"))        # 4 confused reply
_add(b0.respond("hi"))              # 5 small talk: greeting
_add(b0.respond("thanks"))          # 6 small talk: thanks
_add(b0.respond("europe"))          # 7 acknowledgement + next question
_add(b0.respond("skip"))            # 8 skip acknowledgement
_add(b0.respond("why"))             # 9 "no recommendations yet"
b1 = TravelChatbot(dests); b1.greeting()
_add(b1.respond("go"))              # 10 empty-query preference prompt
b2 = TravelChatbot(dests); b2.greeting()
_add(b2.respond("warm cheap week in asia, i love culture"))  # 11 ack of preferences
_add(b2.respond("skip all"))        # 12 recommendations (cards)
_add(b2.respond("why"))             # 13 plain-English why
_add(b2.respond("not the first one"))  # 14 reject-city update
_add(b2.respond("restart"))         # 15 restart
_add(b2.respond("exit"))            # 16 exit
b3 = TravelChatbot(dests); b3.greeting()
_add(b3.respond("not africa or asia, luxury, cold, day trip, beaches and nightlife and seclusion"))  # advisories/tensions
b4 = TravelChatbot(dests); b4.greeting()
_add(b4.respond("tell me about Athens"))     # city data profile
_add(b4.respond("more like Athens"))         # data-driven look-alikes
check("chatbot produces at least 15 DISTINCT responses", len(responses) >= 15)
print("    (distinct responses collected: " + str(len(responses)) + ")")

print("=== 12. DATA-DRIVEN RESPONSES: city profile, more-like, best-time, summary ===")
# Knowledge-base data helpers read columns we already have.
ath = next(d for d in dests if d["city"] == "Athens")
wm = kb.warmest_month(ath["avg_temp_monthly"]); cm = kb.coldest_month(ath["avg_temp_monthly"])
check("kb.warmest_month returns the hottest month", wm is not None and wm[1] >= cm[1])
check("kb.comfortable_months returns warm-band months", all(18 <= ath["avg_temp_monthly"][str(m)]["avg"] <= 30 for m in kb.comfortable_months(ath["avg_temp_monthly"])))
check("kb.format_month_ranges collapses runs", kb.format_month_ranges([5, 6, 7, 8]) == "May-Aug")
check("kb.format_month_ranges wraps the new year", kb.format_month_ranges([12, 1, 2]) == "Dec-Feb")
syd = next((d for d in dests if d["city"] == "Sydney"), None)
if syd is not None:
    check("kb.hemisphere: Sydney is southern", kb.hemisphere(syd["latitude"]) == "southern")
check("kb.hemisphere: Athens is northern", kb.hemisphere(ath["latitude"]) == "northern")
# "tell me about <city>" -> a data profile (no certainty factors).
bot_city = TravelChatbot(dests)
bot_city.greeting()
prof = bot_city.respond("tell me about Athens")
check("describe_city: returns a profile mentioning the city", isinstance(prof, str) and "Athens" in prof and "Climate:" in prof)
check("describe_city: profile hides technical maths", "cf" not in prof.lower().replace("pacific", "") and "m=" not in prof)
# A plain preference that happens to name a city is NOT hijacked by describe.
bot_city2 = TravelChatbot(dests)
bot_city2.greeting()
bot_city2.respond("i want to visit Tokyo")
check("describe_city: a plain 'visit Tokyo' is still a preference, not a profile", bot_city2.query["countries_include"] == ["Japan"])
# "more like <city>" -> recommendations cloning that city's vibe.
bot_like = TravelChatbot(dests)
bot_like.greeting()
like_reply = bot_like.respond("somewhere more like Athens")
check("more_like: clones the city's budget into the query", bot_like.query["budget"] == [ath["budget_level"]])
check("more_like: clones the city's climate into the query", bot_like.query["climate"] == [ath["climate"]])
check("more_like: produces recommendations", isinstance(like_reply, dict) and like_reply.get("type") == "recommendations")
check("more_like: does not recommend the seed city back", all(c["city"] != "Athens" for c in like_reply["results"]))
# Data summary line is present and reports the pool size.
bot_sum = TravelChatbot(dests)
bot_sum.greeting()
bot_sum.respond("europe, warm, i love culture")
sum_reply = bot_sum.respond("skip all")
check("recommendations carry a data summary line", isinstance(sum_reply, dict) and "compared" in sum_reply.get("summary", ""))
check("data summary appears in the CLI text", "compared" in render_response_text(sum_reply))

print("=== 13. GUIDED REFINE: profile -> 'yes'/tweak -> tailored recommendations ===")
lis = next(d for d in dests if d["city"] == "Lisbon")
# After a city profile the bot remembers the city and invites a yes/tweak answer.
bot_g = TravelChatbot(dests)
bot_g.greeting()
prof = bot_g.respond("tell me about Lisbon")
check("profile asks a follow-up and remembers the city", bot_g._suggested_city is not None and bot_g._suggested_city["city"] == "Lisbon")
check("profile invites a yes/tweak answer", "say \"yes\"" in prof or "just say" in prof)
# "yes" -> similar places, seed city not recommended back, flag cleared.
yes_reply = bot_g.respond("yes")
check("'yes' after a profile yields recommendations", isinstance(yes_reply, dict) and yes_reply.get("type") == "recommendations")
check("'yes' clones the city's vibe (mild + Mid-range)", bot_g.query["climate"] == [lis["climate"]] and bot_g.query["budget"] == [lis["budget_level"]])
check("the suggested-city flag is cleared after acting", bot_g._suggested_city is None)
# Tweaks: "warmer and cheaper, more beaches" override the cloned template.
bot_t = TravelChatbot(dests)
bot_t.greeting()
bot_t.respond("tell me about Lisbon")
tw_reply = bot_t.respond("but warmer and cheaper, more beaches")
check("tweak overrides climate to warm", bot_t.query["climate"] == ["warm"])
check("tweak overrides budget to Budget (cheaper)", bot_t.query["budget"] == ["Budget"])
check("tweak boosts the named interest (beaches -> 5)", bot_t.query["lifestyle"].get("beaches") == 5)
check("tweak still produces recommendations", isinstance(tw_reply, dict) and tw_reply.get("type") == "recommendations")
check("tweaked picks are actually warm", all(c["dest"]["climate"] == "warm" for c in tw_reply["results"]))
# Decline steps back without recommending.
bot_n = TravelChatbot(dests)
bot_n.greeting()
bot_n.respond("tell me about Lisbon")
no_reply = bot_n.respond("no")
check("'no' after a profile steps back politely", isinstance(no_reply, str) and "No problem" in no_reply and bot_n._suggested_city is None)
# Comparative climate words resolve.
check("vocab: 'warmer' -> warm", nlp.extract("somewhere warmer", kb.CLIMATE_SYNONYMS)["include"] == ["warm"])
check("vocab: 'cooler' -> cold", nlp.extract("a bit cooler", kb.CLIMATE_SYNONYMS)["include"] == ["cold"])
check("vocab: 'pricier' -> Luxury", nlp.extract("something pricier", kb.BUDGET_SYNONYMS)["include"] == ["Luxury"])

print("=== 14. PER-USER IMPORTANCE: priorities scale the certainty factors ===")
# Default: no stated priority -> behaves exactly like the fixed base weights.
city = next(d for d in dests if d["city"] == "Athens")
q_base = {"climate": ["warm"], "budget": ["Budget"], "lifestyle": {"culture": 5}}
base_conf = inf.score_destination(city, q_base)["confidence"]
check("importance default ({}) leaves scoring unchanged", inf.score_destination(city, dict(q_base, importance={}))["confidence"] == base_conf)
# Athens is only weakly 'warm' (penalty). Caring MORE about climate must lower the
# score; caring LESS must raise it.
hi = inf.score_destination(city, dict(q_base, importance={"climate": 1.3}))["confidence"]
lo = inf.score_destination(city, dict(q_base, importance={"climate": 0.6}))["confidence"]
check("raising climate importance amplifies its (negative) effect", hi < base_conf)
check("lowering climate importance softens its effect", lo > base_conf)
# Effective rule CF stays clamped to a sane range even with a big multiplier.
big = inf.score_destination(city, dict(q_base, importance={"lifestyle": 5.0}))
check("importance is clamped (no CF blows past 0.97)", all(-0.97 <= cf <= 0.97 for _, (m, cf) in big["details"].items()))
# The chatbot detects a stated priority and attaches it to the RIGHT criterion.
def imp_of(msg):
    b = TravelChatbot(dests); b.greeting(); b.respond(msg)
    return b.query["importance"]
check("detect: 'budget is the most important' -> budget up", imp_of("warm, cheap, i love culture, but budget is the most important").get("budget") == 1.3)
check("detect: 'warm weather is a must' -> climate up", imp_of("i want it cheap, warm weather is a must").get("climate") == 1.3)
check("detect: 'dont care about the weather' -> climate down", imp_of("europe, i love nature, i dont really care about the weather").get("climate") == 0.6)
check("detect: 'culture matters most' -> lifestyle up", imp_of("culture matters most to me").get("lifestyle") == 1.3)
check("detect: a normal sentence sets no priority", imp_of("a warm cheap week in asia") == {})
# The acknowledgement confirms the weighting back to the user.
b_imp = TravelChatbot(dests); b_imp.greeting()
ack_imp = b_imp.respond("cheap, warm, but budget matters most")
check("acknowledgement confirms the weighting", "weight" in ack_imp and "budget" in ack_imp)

print("=== SUMMARY ===")
print("  passed:", PASSED, "| failed:", FAILED)
if FAILED:
    print("  failing checks:", FAILS)
print("  assisting libraries: NLTK (tokenize/lemmatize/POS) and pandas (CSV load).")
print("  All KBS reasoning (fuzzy, certainty factors, cosine, rules, inference,")
print("  spell correction, negation) is our own code.")
