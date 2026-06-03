Travel Destination Expert System Chatbot
========================================

A KNOWLEDGE-BASED SYSTEM (KBS) that recommends travel destinations through a
conversation. The user describes an ideal trip in free text; the system turns
that text into structured facts, reasons over a knowledge base of 560 cities,
and returns ranked recommendations with an explanation. It runs from the command
line (main.py) or as a Streamlit chat GUI (app.py); both front ends use the same
TravelChatbot class, so the reasoning is identical.

The expert-system reasoning is our own: the knowledge base, fuzzy logic,
MYCIN-style certainty factors, cosine lifestyle matching, rule-based advisories,
forward chaining, the inference engine and the explanation facility. Established
libraries are used ONLY as assisting tools: NLTK for tokenisation/lemmatisation
and pandas for loading the CSV. There is no LLM, no external recommender, and no
pre-built expert-system shell.

NOTE FOR EVALUATION: the source code is kept comment-free on purpose; this
README and the written report are the documentation. Every behaviour described
here is exercised by the test suite (tests.py).


Required installations
======================

1. Python 3.9 or newer.

2. Install the Python dependencies:
      pip install -r requirements.txt
   This installs:
      - pandas    : loads the dataset CSV (knowledge_base.py)
      - nltk      : tokenisation, POS tagging, lemmatisation (nlp.py)
      - streamlit : the optional GUI only (app.py). The CLI and tests do NOT
                    need streamlit.

3. Download the NLTK data the NLP layer needs (one-off, needs internet):
      python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','wordnet','omw-1.4')]"
   nlp.py also attempts this automatically on first import. If the data is
   missing and cannot be downloaded (e.g. offline), the program stops with a
   clear message listing the exact command above instead of a cryptic
   NLTK LookupError.

4. Dataset: keep the file
      Worldwide Travel Cities Dataset (Ratings and Climate).csv
   in the SAME folder as the .py files (it already is). load_destinations()
   finds it next to knowledge_base.py automatically.


How to run
==========

Command line:
      python3 main.py
   The bot greets you and asks about your trip one question at a time. Type
   normally; commands: go, why, restart, help, skip, skip all, exit.

Streamlit GUI:
      streamlit run app.py
   A browser chat page opens (http://localhost:8501). Features:
      - a sidebar "Your Trip Profile" showing everything understood so far,
      - clickable example prompts on the first screen,
      - recommendation cards written as natural sentences (match quality, city,
        region, budget, temperature, strengths and trade-offs), each with a
        "Why this pick?" panel (plain English + an Advanced reasoning tab),
        a city profile, and a lifestyle bar chart,
      - a compact comparison table under the cards,
      - follow-up buttons (More like #1, Cheaper, Warmer, Why?, Remove #1,
        Restart, ...) that send the same chat commands,
      - conversational avoidance warnings when a pick still has an avoided
        feature.

Tests:
      python3 tests.py
   Prints PASS/FAIL per check and a final summary (currently 287 checks, all
   passing). See "Validation" below.


Example user inputs
===================
- a cheap warm week in Asia, I love food and culture
- not Africa or Asia, mid-range, mild, nature and wellness
- South America, short trip, not expensive, avoid nightlife
- somewhere in Japan or Europe in July        (country + region = a union)
- I want a chaep trip                          (typo -> Budget, spell-corrected)
- budget doesn't matter, I want luxury         (de-emphasis, not exclusion)
- culture matters most but I don't care about nightlife
- tell me about Kyoto       /     more like Lisbon but warmer and cheaper


Architecture (which file plays which KBS role)
==============================================
- knowledge_base.py  KNOWLEDGE BASE: loads the 560-city CSV with pandas, derives
     a yearly temperature and climate label, and holds the domain vocabularies
     (continents, countries, climate, budget, duration, lifestyle, seasons).
- rules.py           RULE BASE: per-criterion certainty factors (RULE_CF) and the
     forward-chaining advisory IF-THEN rules (ADVISORY_RULES), kept as data.
- fuzzy.py           FUZZY LOGIC: membership functions (temperature -> cold/mild/
     warm; ordinal budget and duration) plus fuzzy AND (min) and OR (max).
- inference.py       INFERENCE ENGINE: crisp region/country filter (union of both,
     with a broaden-the-search fallback), fuzzy memberships -> signed certainty
     factors scaled by a per-user importance multiplier, cosine similarity with a
     city-strength factor for lifestyle, MYCIN-style CF combination, forward
     chaining, and the explanation facility (explain / explain_human).
- nlp.py             NATURAL-LANGUAGE INTERFACE: normalise shorthand -> tokenise
     (NLTK) -> lemmatise (NLTK WordNet) -> match vocabulary -> conservative
     spell-correction -> handle negation. Builds the facts the engine reasons on.
- chatbot.py         DIALOGUE MANAGER + WORKING MEMORY: slot filling, intents
     (greet/help/why/restart/exit/reject/describe/more-like), small talk, the
     semantic emphasis/de-emphasis layer ("budget doesn't matter" vs "no budget"),
     the rejected-cities memory, and turning engine output into a card payload
     (GUI) or text (CLI).
- main.py            command-line interface.
- app.py             Streamlit GUI (presentation only).
- tests.py           validation harness.


KBS techniques used
===================
- Fuzzy logic: climate is a temperature mapped to three overlapping fuzzy sets;
  budget and duration are ordinal (exact = 1.0, one band off = 0.5, far = 0.0).
- Certainty factors (MYCIN): membership m -> signed evidence CF = 2m-1, times the
  rule reliability (and a per-user importance factor), combined with the three
  MYCIN cases; a hard-failure gate caps confidence on an explicit zero match.
- Forward chaining: advisory rules fired to a fixpoint, conflict resolution by
  priority.
- Cosine similarity with a strength factor for the lifestyle taste match.
- Slot-filling dialogue with conservative spell-correction; the bot also reads
  free text and natural priorities ("X matters most", "I don't care about Y").


Validation
==========
Run: python3 tests.py
The suite covers: data loading; the NLP pipeline and known false positives
(cheap->warm, mild->Mid-range, short->Long trip, multi-word regions, negation);
the Levenshtein spell-corrector; fuzzy membership values; the certainty-factor
maths and the three MYCIN cases; cosine similarity and the lifestyle strength
factor; recommender filtering/ranking including the country+region union;
forward-chaining advisories (including a chained rule); exclusions vs
de-emphasis ("don't want" vs "don't care"); per-user importance weighting;
empty-query handling; proactive recommendation once all slots are gathered;
the data-driven city profile / more-like features; and at least 15 distinct
chatbot responses. The final line prints the pass/fail summary.
