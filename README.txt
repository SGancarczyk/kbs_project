Travel Destination Expert System Chatbot
========================================

This project is a pure Python knowledge-based travel recommender. It uses a chatbot interface to collect free-text preferences, translates them into structured facts, and ranks destinations from the CSV knowledge base. It can be used either from the command line (main.py) or through a Streamlit chat GUI (app.py); both front ends drive the exact same TravelChatbot class.

How to run (command line)
-------------------------
1. Keep these files in the same folder:
   - main.py
   - app.py
   - chatbot.py
   - knowledge_base.py
   - nlp.py
   - fuzzy.py
   - inference.py
   - rules.py
   - tests.py
   - Worldwide Travel Cities Dataset (Ratings and Climate).csv

2. Start the chatbot:
   python3 main.py

3. Run the validation tests:
   python3 tests.py

Streamlit GUI
-------------
A browser-based chat interface is provided in app.py. It is a thin presentation
layer around the same TravelChatbot class used by the command line, so the GUI
and the CLI behave identically.

1. Install Streamlit (the ONLY external dependency, and only for the GUI):
   pip install streamlit

2. Launch the GUI:
   streamlit run app.py

3. Your browser opens a chat page. The bot greets you automatically. Type your
   trip preferences and press Enter (or the send arrow). User messages appear on
   the right, the bot's on the left. All commands still work through the chat
   box: 'go' (see picks), 'why' (reasoning), 'restart', 'help', and 'exit'.
   When you type 'exit', the input is disabled and a farewell is shown; refresh
   the page to start a new conversation.

The GUI keeps the chatbot instance and the chat history in st.session_state, so
the bot is created once per session (not rebuilt on every message) and remembers
everything during the session, including rejected cities (see below).

Memory system (rejected cities)
-------------------------------
After the bot shows recommendations, you can tell it to drop a city and it will
remember that choice for the rest of the session. For example:
   not Paris
   remove Barcelona
   I don't want Tokyo
   skip Rome
   not the first one  /  not the top one
   exclude those cities
The bot adds the matching city (or cities) to a per-session set and immediately
re-runs the recommendation with those places filtered out, backfilling fresh
picks so you still see a full list. Rejected cities stay excluded for every
later recommendation in the same session; typing 'restart' clears the memory.

Dependencies
------------
The core system (CLI, NLP, fuzzy logic, inference, tests) requires no external
Python libraries and uses only Python's standard library. The optional Streamlit
GUI (app.py) needs the 'streamlit' package (pip install streamlit); nothing else
is added. The project does not require NLTK, pandas, scipy, scikit-learn, Google
Colab, an LLM API, or any external recommender system.

System structure
----------------
- knowledge_base.py stores the travel dataset loader and the domain vocabulary.
- nlp.py first normalizes common informal shorthand (e.g. wanna -> want to, w/o -> without, & -> and), then normalizes punctuation, tokenizes, applies a small rule-based lemmatizer, detects multi-word phrases, handles negation, and performs conservative typo correction.
- fuzzy.py contains fuzzy membership functions for climate, budget, and duration.
- inference.py scores destinations using crisp region filters, fuzzy matching, lifestyle cosine similarity with a strength factor, signed certainty factors, exclusions as negative evidence, confidence gating, and forward-chaining advisories.
- rules.py stores the rule certainty factors and advisory IF-THEN rules.
- chatbot.py manages the conversation, slot filling, skip behavior, restart/help/why commands, natural-language confirmations, the rejected-cities memory, and explanation output.
- main.py is the command-line entry point.
- app.py is the optional Streamlit GUI front end (a thin wrapper around the same TravelChatbot class as main.py).
- tests.py validates data loading, NLP, fuzzy logic, certainty-factor maths, inference behavior, advisory rules, and chatbot-level edge cases.

Important behavior
------------------
Example inputs:
- a cheap warm week in Asia, I love food and culture
- not Africa or Asia, mid-range, mild, nature and wellness
- South America, short trip, not expensive, avoid nightlife
- Europe, weekend, beaches, warm

The NLP layer first expands informal shorthand (wanna -> want to, gonna -> going to, smth -> something, pls/plz -> please, u -> you, bc/cuz -> because, w/ -> with, w/o -> without, & -> and) so messy chat input still matches the vocabulary. It then detects exact phrases before single-word matching, so phrases such as south america, north america, latin america, middle east, new zealand, and united states are handled correctly. Spell correction is conservative and can be disabled per slot by the chatbot, preventing false positives such as cheap -> warm, mild -> Mid-range, south -> Long trip, or short trip -> beaches.

Exclusions are preserved and used by inference. For example, not expensive penalizes Luxury destinations, and avoid nightlife penalizes cities with high nightlife scores. If the user gives no meaningful preference, the chatbot asks for at least one preference instead of returning arbitrary zero-confidence recommendations.

Validation
----------
Run python3 tests.py. The current test suite checks 91 conditions, including known false-positive NLP cases, multi-word region parsing, negation across phrases such as not Africa or Asia, exclusion penalties, budget-conflict handling, and confidence gating for explicit hard failures.
