# rules.py
# ----------------------------------------------------------------------------
# THE RULE BASE - the "procedural knowledge" (IF-THEN rules) of our expert
# system, kept as plain data, separate from the engine in inference.py
# ("separation of knowledge from processing").
# ----------------------------------------------------------------------------
# --- Per-criterion certainty factors ---
# A CERTAINTY FACTOR (CF) is how much we trust a rule (0..1). We give EACH kind
# of evidence its own CF, reflecting how decisive it is for a travel choice.
# The engine turns a fuzzy membership m into signed evidence with CF = MB - MD =
# m - (1-m) = 2m-1, then multiplies by the rule CF below (CF "propagation").
# So a perfect match -> +ruleCF, a total miss -> -ruleCF (negative evidence),
# a half match -> 0. Lifestyle taste is the most decisive signal, duration the
# least, which is why their CFs differ.
RULE_CF = {
    "lifestyle": 0.90,
    "budget": 0.85,
    "climate": 0.80,
    "duration": 0.70,
}
# --- Advisory (forward-chaining) production rules ---
# Each rule: IF every fact in "if" is true THEN add fact "then" (and show
# "message"). The engine fires them to a fixpoint, so one rule's "then" can
# trigger another (CHAINING). "priority" drives conflict resolution (higher
# reported first). Facts come from the user's request AND from the results
# (e.g. 'results:empty'), so advice is aware of both what was asked and what was
# found. Tensions warn about contradictions; synergies highlight good combos;
# result rules react to the outcome.
ADVISORY_RULES = [
    # ---- contradictions in the request ----
    {"id": "conflict_budget", "priority": 6, "if": ["budget:Budget", "budget:Luxury"], "then": "conflict:budget", "message": "You mentioned both cheap and luxury - those pull apart, so I'll lean toward mid-range as a bridge."},
    {"id": "assume_midrange", "priority": 3, "if": ["conflict:budget"], "then": "action:assume_midrange", "message": "(I'll treat your budget as flexible / mid-range while ranking.)"},
    {"id": "daytrip_multi", "priority": 6, "if": ["duration:Day trip", "regions:multiple"], "then": "warn:daytrip_multi", "message": "A single day trip across several continents isn't realistic - one region would make more sense."},
    {"id": "beach_cold", "priority": 5, "if": ["wants:beaches", "climate:cold"], "then": "tension:beach_cold", "message": "Beaches usually shine in warm places - in a cold climate good beach matches may be few."},
    {"id": "night_seclusion", "priority": 5, "if": ["wants:nightlife", "wants:seclusion"], "then": "tension:night_seclusion", "message": "Lively nightlife and deep seclusion pull in opposite directions - I'll try to balance them."},
    {"id": "night_wellness", "priority": 4, "if": ["wants:nightlife", "wants:wellness"], "then": "tension:night_wellness", "message": "Partying and wellness/relaxation are a bit at odds - expect a compromise."},
    {"id": "urban_seclusion", "priority": 5, "if": ["wants:urban", "wants:seclusion"], "then": "tension:urban_seclusion", "message": "A buzzing city and quiet seclusion rarely live in the same place - I'll find a middle ground."},
    {"id": "urban_nature", "priority": 4, "if": ["wants:urban", "wants:nature"], "then": "tension:urban_nature", "message": "Big-city life and untouched nature can conflict - I'll favour places that blend both."},
    {"id": "adventure_wellness", "priority": 4, "if": ["wants:adventure", "wants:wellness"], "then": "tension:adventure_wellness", "message": "Adventure and pure relaxation are different moods - I'll balance energetic and calm options."},
    {"id": "luxury_budgetdest", "priority": 5, "if": ["budget:Luxury", "climate:cold", "wants:beaches"], "then": "tension:lux_cold_beach", "message": "Luxury cold-climate beaches are a very narrow niche - results may be sparse."},
    # ---- synergies (good combinations) ----
    {"id": "beach_warm", "priority": 2, "if": ["wants:beaches", "climate:warm"], "then": "synergy:beach_warm", "message": "Nice - warm beach destinations are a classic sweet spot, so expect strong matches."},
    {"id": "culture_cuisine", "priority": 2, "if": ["wants:culture", "wants:cuisine"], "then": "synergy:culture_cuisine", "message": "Culture and cuisine pair beautifully - many cities excel at both."},
    {"id": "nature_adventure", "priority": 2, "if": ["wants:nature", "wants:adventure"], "then": "synergy:nature_adventure", "message": "Nature and adventure go hand in hand - lots of great fits for that."},
    # ---- reactions to the actual results ----
    {"id": "results_broadened", "priority": 7, "if": ["results:broadened"], "then": "note:broadened", "message": "No destination matched your exact region, so I widened the search (still respecting any region you excluded)."},
    {"id": "results_empty", "priority": 7, "if": ["results:empty"], "then": "note:empty", "message": "I couldn't find anything matching all of that - try relaxing one wish (a region, the budget, or the climate)."},
    {"id": "results_weak", "priority": 4, "if": ["results:weak"], "then": "note:weak", "message": "These are the closest I found, but the matches are weak - loosening a criterion would help."},
]
