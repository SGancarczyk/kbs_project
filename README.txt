Travel Destination Expert System Chatbot
========================================

This project is a pure-Python KNOWLEDGE-BASED SYSTEM (KBS) that recommends
travel destinations. A chatbot is the intermediary: the user describes an ideal
trip in free text, the system turns that text into structured facts, reasons
over a knowledge base of 560 cities, and returns ranked recommendations with an
explanation. It can be driven from the command line (main.py) or through a
Streamlit chat GUI (app.py); both front ends use the exact same TravelChatbot
class, so the reasoning is identical.


How to run
==========

Command line (no installation needed):
   python3 main.py

Streamlit GUI (one optional dependency):
   pip install streamlit        # the ONLY external package, used only by the GUI
   streamlit run app.py
   A browser chat page opens. The bot greets you automatically; describe your
   trip and press Enter. User messages appear on the right, the bot's on the
   left. Each recommendation is shown as a CARD with a confidence value, a
   "Strong on:" line, and a "Why this pick?" panel that explains the choice in
   plain English. Commands still work in the chat box: go / why / restart /
   help / exit, and after picks appear you can say e.g. "not Paris" to drop a
   city.

Validation tests:
   python3 tests.py


Knowledge-Based System architecture (which file plays which KBS role)
=====================================================================
The project deliberately mirrors the KBS component structure from the course
(Knowledge Base + Inference Engine + Working Memory + Explanation Facility +
User Interface) and keeps "knowledge" cleanly separated from "processing":

- knowledge_base.py  -> KNOWLEDGE BASE (facts + vocabulary ONLY).
     Loads the CSV of 560 cities into clean dictionaries, derives a yearly
     average temperature and a crisp climate label, and holds the domain
     vocabulary (synonym maps from human words to dataset values). No reasoning
     lives here. This is "simple relational knowledge" (a table) plus the
     language vocabulary the NLP layer looks words up in.

- rules.py           -> RULE BASE (procedural knowledge, plain data ONLY).
     Holds the per-criterion certainty factors (RULE_CF) and the advisory
     IF-THEN production rules (ADVISORY_RULES) used by forward chaining. Kept as
     data, separate from the engine, so the same engine could run a different
     rule base ("separation of knowledge from processing").

- inference.py       -> INFERENCE ENGINE (reasoning ONLY).
     Scores each city and ranks them. Combines: a crisp region filter, fuzzy
     membership for soft criteria, cosine similarity for lifestyle taste, signed
     certainty factors combined MYCIN-style, and forward-chaining advisories.
     Also the EXPLANATION FACILITY: explain() (technical trace) and
     explain_human() (plain-English bullets).

- fuzzy.py           -> FUZZY-LOGIC layer used by the engine.
     Membership functions that turn raw values into a degree in [0,1], plus
     fuzzy AND (min) and OR (max).

- nlp.py             -> NATURAL-LANGUAGE INTERFACE (hand-written, stdlib only).
     normalize -> tokenize -> lemmatize -> match vocabulary -> spell-correct ->
     handle negation. Builds the FACTS the engine reasons over.

- chatbot.py         -> DIALOGUE MANAGER + WORKING MEMORY.
     Slot filling, skip/restart/help/why/exit commands, natural confirmations,
     the per-session rejected-cities memory (dynamic facts), and turning the
     engine output into either a structured card payload (GUI) or text (CLI).

- main.py            -> USER INTERFACE (command line).
- app.py             -> USER INTERFACE (Streamlit GUI / presentation layer).
- tests.py           -> validation harness (93 assertions).


AI / KBS techniques used, and where
===================================
- Fuzzy logic (fuzzy.py): climate is a continuous temperature mapped to three
  overlapping fuzzy sets (cold = left shoulder/trapezoid, warm = right
  shoulder/trapezoid, mild = triangle peaking at 18C). Budget and duration are
  ordered categories scored with a discrete ordinal membership (a sampled
  triangular kernel: exact = 1.0, one step off = partial, far = 0.0). Combining
  several acceptable values per criterion uses fuzzy OR = max.

- Certainty factors, MYCIN-style (inference.py + rules.py): each criterion's
  fuzzy membership m becomes signed evidence with CF = MB - MD = m - (1-m) =
  2m-1, then is multiplied by that rule's reliability (CF propagation). Per-city
  evidence is combined with the three MYCIN cases (both positive, both negative,
  mixed signs) in cf_combine(), folded over all criteria in cf_combine_many().

- Forward chaining (inference.py): advisory production rules are fired to a
  fixpoint over a working-memory fact set built from the request AND the
  results; conflict resolution is by PRIORITY (salience), highest first.

- Crisp rule + fallback (inference.py): a hard region filter, with a documented
  fallback that broadens the search if the requested region yields nothing.

- Cosine similarity (inference.py): lifestyle "taste" match between the user's
  interest vector and the city's lifestyle scores, scaled by a strength factor.

- Slot filling + spell correction (chatbot.py + nlp.py): the bot fills one slot
  at a time from free text; a conservative, hand-written Levenshtein corrector
  fixes typos.

NLP independence note: nlp.py uses ONLY Python's standard library (just `re`).
There is no NLTK, spaCy, transformers, or any external NLP API - the whole
language pipeline is implemented by us, as the brief requires.


Changes in this session
=======================
1. Structured recommendation CARDS. A successful recommendation is now a
   structured payload; the GUI renders one bordered card per pick (confidence
   metric, "Strong on:" line, "Why this pick?" panel). The CLI flattens the same
   payload back to text, so it behaves exactly as before.

2. Human-readable explanations. inference.explain_human() turns the technical
   certainty-factor trace into a few plain-English bullets (e.g. "💰 Budget —
   perfect fit (Mid-range as requested)", "🌡 Climate — great match (warm,
   27C yearly average)"). Negligible criteria and no-effect exclusions are
   hidden; poor matches are flagged with a warning. The original explain()
   (technical, used by the CLI/tests) is unchanged.

3. Duration is no longer guessed. The chatbot fills the duration slot only from
   an EXPLICIT duration word (spell correction is disabled for duration), so the
   bot asks about trip length instead of inferring it from unrelated words.

4. "sports" NLP fix. A spell-correction blocklist (nlp.SPELLCHECK_BLOCKLIST)
   stops common English words from being mis-corrected into lookalike keywords
   (the classic "sports" -> "short" -> Short trip). "sports"/"sport" are also
   mapped positively to the adventure interest in knowledge_base.py.


Important behavior
==================
Example inputs:
- a cheap warm week in Asia, I love food and culture
- not Africa or Asia, mid-range, mild, nature and wellness
- South America, short trip, not expensive, avoid nightlife
- Europe, weekend, beaches, warm, and I love sports

The NLP layer first expands informal shorthand (wanna -> want to, gonna -> going
to, smth -> something, pls/plz -> please, u -> you, bc/cuz -> because, w/ ->
with, w/o -> without, & -> and) so messy chat input still matches the
vocabulary. It then matches exact phrases before single words, so phrases like
south america, north america, latin america, middle east, new zealand, and
united states are handled correctly. Spell correction is conservative: it is
disabled entirely for the duration slot, it refuses to rewrite common words on a
blocklist (so "sports" never becomes "Short trip"), and it preserves first/last
letters - preventing false positives such as cheap -> warm, mild -> Mid-range,
or south -> Long trip.

Exclusions are preserved and used by inference. For example, "not expensive"
penalizes Luxury destinations, and "avoid nightlife" penalizes cities with high
nightlife scores. If the user gives no meaningful preference, the chatbot asks
for at least one preference instead of returning arbitrary zero-confidence
recommendations.

Memory (rejected cities): after picks are shown you can say "not Paris",
"remove Barcelona", "not the first one", or "exclude those cities". The bot
remembers the rejected cities for the rest of the session and backfills fresh
picks; "restart" clears the memory.


Dependencies
============
The core system (CLI, NLP, fuzzy logic, inference, tests) requires NO external
Python libraries - standard library only. The optional Streamlit GUI (app.py)
needs the 'streamlit' package (pip install streamlit); nothing else is added.
The project does not require NLTK, spaCy, pandas, scipy, scikit-learn, an LLM
API, or any external recommender system.


Validation
==========
Run: python3 tests.py
The suite checks 93 conditions, including: data loading, the NLP pipeline and
known false-positive cases (cheap->warm, mild->Mid-range, short->Long trip,
multi-word regions, "not Africa or Asia"), fuzzy membership values, the
certainty-factor maths (CF = MB-MD and the three MYCIN combination cases),
cosine similarity, recommender filtering/ranking, forward-chaining advisory
rules (including a chained rule), exclusion penalties, budget-conflict handling,
confidence gating for hard failures, and chatbot-level edge cases.
