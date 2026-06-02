Travel Destination Expert System Chatbot
========================================

This project is a KNOWLEDGE-BASED SYSTEM (KBS) that recommends travel
destinations. A chatbot is the intermediary: the user describes an ideal trip in
free text, the system turns that text into structured facts, reasons over a
knowledge base of 560 cities, and returns ranked recommendations with an
explanation. It can be driven from the command line (main.py) or through a
Streamlit chat GUI (app.py); both front ends use the exact same TravelChatbot
class, so the reasoning is identical.

The expert-system reasoning (knowledge base, fuzzy logic, certainty factors,
cosine lifestyle matching, rule-based advisories, forward chaining, inference
engine, explanation facility) is our own. We use established NLP/data libraries
ONLY as assisting tools: NLTK for tokenization/lemmatization and pandas for
loading the CSV. There is no LLM, no external recommender, and no pre-built
expert-system shell.


Setup
=====

1. Install the dependencies:
      pip install -r requirements.txt
   (pandas, nltk, and — for the GUI only — streamlit.)

2. Download the NLTK data the NLP layer needs (one-off, needs internet):
      python -c "import nltk; [nltk.download(p) for p in \
        ('punkt','punkt_tab','averaged_perceptron_tagger',\
         'averaged_perceptron_tagger_eng','wordnet','omw-1.4')]"
   nlp.py also tries this automatically on first import. If the data is missing
   and cannot be fetched (e.g. offline), the program stops with a clear message
   listing the exact command above, instead of a cryptic NLTK LookupError.

3. Dataset placement: keep the file
      Worldwide Travel Cities Dataset (Ratings and Climate).csv
   in the SAME folder as the .py files (it already is). load_destinations()
   finds it next to knowledge_base.py automatically.


How to run
==========

Command line:
   python3 main.py

Streamlit GUI:
   streamlit run app.py
   A browser chat page opens. The bot greets you automatically; describe your
   trip and press Enter. User messages appear on the right, the bot's on the
   left. Each recommendation is a CARD with a confidence value, a "Strong on:"
   line, and a "Why this pick?" panel that explains the choice in plain English.
   Commands work in the chat box: go / why / restart / help / exit, and after
   picks appear you can say e.g. "not Paris" to drop a city.

Validation tests:
   python3 tests.py


Example user inputs
===================
- a cheap warm week in Asia, I love food and culture
- not Africa or Asia, mid-range, mild, nature and wellness
- South America, short trip, not expensive, avoid nightlife
- Europe, weekend, beaches, warm, and I love sports
- somewhere in Japan or Europe in July          (country + region = a union)
- I want a chaep trip                            (typo -> Budget, spell-corrected)


Knowledge-Based System architecture (which file plays which KBS role)
=====================================================================
The project mirrors the KBS component structure from the course (Knowledge Base
+ Inference Engine + Working Memory + Explanation Facility + User Interface) and
keeps "knowledge" cleanly separated from "processing":

- knowledge_base.py  -> KNOWLEDGE BASE (facts + vocabulary).
     Loads the CSV of 560 cities WITH pandas into clean dictionaries, derives a
     yearly average temperature and a crisp climate label, and holds the domain
     vocabulary (synonym maps from human words to dataset values: continents,
     countries, climate, budget, duration, lifestyle, seasons/months). No
     reasoning lives here.

- rules.py           -> RULE BASE (procedural knowledge, plain data).
     Per-criterion certainty factors (RULE_CF) and the advisory IF-THEN
     production rules (ADVISORY_RULES) used by forward chaining. Kept as data,
     separate from the engine, so the same engine could run a different rule base.

- inference.py       -> INFERENCE ENGINE (reasoning).
     Scores and ranks each city. Combines: a crisp region/country filter (UNION
     when both are given, with a documented broaden-the-search fallback), fuzzy
     membership for soft criteria, cosine similarity scaled by a city-strength
     factor for lifestyle taste, signed certainty factors combined MYCIN-style,
     and forward-chaining advisories. Also the EXPLANATION FACILITY: explain()
     (technical trace) and explain_human() (plain-English bullets).

- fuzzy.py           -> FUZZY-LOGIC layer used by the engine.
     Membership functions mapping raw values to a degree in [0,1], plus fuzzy
     AND (min) and OR (max).

- nlp.py             -> NATURAL-LANGUAGE INTERFACE.
     normalize informal shorthand -> tokenize (NLTK) -> lemmatize (NLTK WordNet,
     POS-guided) -> match vocabulary -> conservative spell-correct -> handle
     negation. Builds the FACTS the engine reasons over.

- chatbot.py         -> DIALOGUE MANAGER + WORKING MEMORY.
     Slot filling, skip/restart/help/why/exit commands, small talk, natural
     confirmations, the per-session rejected-cities memory (dynamic facts), and
     turning the engine output into either a structured card payload (GUI) or
     text (CLI).

- main.py            -> USER INTERFACE (command line).
- app.py             -> USER INTERFACE (Streamlit GUI / presentation layer).
- tests.py           -> validation harness.


AI / KBS techniques used, and where
===================================
- Fuzzy logic (fuzzy.py): climate is a continuous temperature mapped to three
  overlapping fuzzy sets (cold = left shoulder, warm = right shoulder, mild =
  triangle peaking at 18C). Budget and duration are ordered categories scored
  with a discrete ordinal membership: exact match = 1.0, one band off = 0.5,
  far = 0.0. Combining several acceptable values per criterion uses fuzzy OR=max.

- Certainty factors, MYCIN-style (inference.py + rules.py): each criterion's
  fuzzy membership m becomes signed evidence with CF = MB - MD = m - (1-m) =
  2m-1, then is multiplied by that rule's reliability (CF propagation). So a
  full match -> +ruleCF, one band off (m=0.5) -> 0 (neutral), a full miss ->
  -ruleCF (negative evidence). Per-city evidence is combined with the three
  MYCIN cases in cf_combine(), folded over all criteria in cf_combine_many().
  A hard-failure gate caps confidence when an explicitly requested criterion has
  zero membership, so lifestyle alone cannot rescue a clear climate miss.

- Forward chaining (inference.py): advisory production rules fire to a fixpoint
  over a working-memory fact set built from the request AND the results;
  conflict resolution is by PRIORITY (salience), highest first.

- Crisp rule + fallback (inference.py): a hard region/country filter, with a
  documented fallback that broadens the search if the request yields nothing.

- Cosine similarity (inference.py): lifestyle "taste" match between the user's
  interest vector and the city's lifestyle scores, scaled by a strength factor
  so a city that is merely "balanced" does not tie with one that is genuinely
  strong on what the user cares about. Only the dimensions the user selected are
  used, so unselected dimensions never distort the score.

- Slot filling + spell correction (chatbot.py + nlp.py): the bot fills one slot
  at a time from free text; a conservative, our-own Levenshtein corrector fixes
  small typos (e.g. "chaep" -> "cheap"). It is guarded against false positives:
  it ignores short tokens and a blocklist of common words, requires matching
  first/last letters, and is disabled entirely for short collision-prone slots
  (duration, country, season).

NLP note: nlp.py uses NLTK (word_tokenize, pos_tag, WordNetLemmatizer) as an
assisting tool for tokenization and lemmatization. All vocabulary matching,
negation handling, and spell correction are our own code. No spaCy,
transformers, scipy, scikit-learn, or external NLP API is used.


Behaviour notes
===============
The NLP layer first expands informal shorthand (wanna -> want to, gonna -> going
to, smth -> something, pls/plz -> please, u -> you, bc/cuz -> because, w/ ->
with, w/o -> without, & -> and). It matches multi-word phrases before single
words, so "south america", "north america", "latin america", "middle east",
"new zealand", and "united states" are handled correctly and never collide.

Exclusions are preserved and used by inference: "not expensive" penalizes Luxury
destinations, "avoid nightlife" penalizes cities with high nightlife scores, and
"not Africa or Asia" excludes both regions. If the user gives no meaningful
preference, the chatbot asks for at least one preference instead of returning
arbitrary zero-confidence recommendations.

Region + country: if the user names both (e.g. "Japan or Europe"), the engine
keeps the UNION (Japanese cities AND all European cities) rather than dropping
one. Excluded regions/countries are always removed.

Memory (rejected cities): after picks are shown you can say "not Paris",
"remove Barcelona", "not the first one", or "exclude those cities". The bot
remembers them for the rest of the session and backfills fresh picks; "restart"
clears the memory.


Dependencies
============
- pandas      : CSV loading in knowledge_base.py (required).
- nltk        : tokenization, POS tagging, lemmatization in nlp.py (required),
                plus its data packages (see Setup).
- streamlit   : the optional GUI in app.py ONLY. The CLI (main.py) and the test
                suite (tests.py) do not need it.
The project does NOT use scipy, scikit-learn, numpy, spaCy, an LLM API, or any
external recommender system.


Validation
==========
Run: python3 tests.py
The suite checks, among other things: data loading; the NLP pipeline and known
false-positive cases (cheap->warm, mild->Mid-range, short->Long trip, multi-word
regions, "not Africa or Asia"); the Levenshtein corrector; fuzzy membership
values; the certainty-factor maths (CF = MB-MD and the three MYCIN combination
cases); cosine similarity and the lifestyle strength factor; recommender
filtering/ranking including the country+region union; forward-chaining advisory
rules (including a chained rule); exclusion penalties; budget-conflict handling;
confidence gating for hard failures; empty-query behaviour; and chatbot-level
conversation flow. The final line prints a pass/fail summary.
