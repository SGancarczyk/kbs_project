Travel Destination Expert System Chatbot
========================================

This project is a pure Python knowledge-based travel recommender. It uses a chatbot interface to collect free-text preferences, translates them into structured facts, and ranks destinations from the CSV knowledge base.

How to run
----------
1. Keep these files in the same folder:
   - main.py
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

Dependencies
------------
No external Python libraries are required. The implementation uses only Python's standard library. It does not require NLTK, pandas, scipy, scikit-learn, Google Colab, an LLM API, or any external recommender system.

System structure
----------------
- knowledge_base.py stores the travel dataset loader and the domain vocabulary.
- nlp.py normalizes text, tokenizes it, applies a small rule-based lemmatizer, detects multi-word phrases, handles negation, and performs conservative typo correction.
- fuzzy.py contains fuzzy membership functions for climate, budget, and duration.
- inference.py scores destinations using crisp region filters, fuzzy matching, lifestyle cosine similarity with a strength factor, signed certainty factors, exclusions as negative evidence, confidence gating, and forward-chaining advisories.
- rules.py stores the rule certainty factors and advisory IF-THEN rules.
- chatbot.py manages the conversation, slot filling, skip behavior, restart/help/why commands, and explanation output.
- main.py is the command-line entry point.
- tests.py validates data loading, NLP, fuzzy logic, certainty-factor maths, inference behavior, advisory rules, and chatbot-level edge cases.

Important behavior
------------------
Example inputs:
- a cheap warm week in Asia, I love food and culture
- not Africa or Asia, mid-range, mild, nature and wellness
- South America, short trip, not expensive, avoid nightlife
- Europe, weekend, beaches, warm

The NLP layer detects exact phrases before single-word matching, so phrases such as south america, north america, latin america, middle east, new zealand, and united states are handled correctly. Spell correction is conservative and can be disabled per slot by the chatbot, preventing false positives such as cheap -> warm, mild -> Mid-range, south -> Long trip, or short trip -> beaches.

Exclusions are preserved and used by inference. For example, not expensive penalizes Luxury destinations, and avoid nightlife penalizes cities with high nightlife scores. If the user gives no meaningful preference, the chatbot asks for at least one preference instead of returning arbitrary zero-confidence recommendations.

Validation
----------
Run python3 tests.py. The current test suite checks 91 conditions, including known false-positive NLP cases, multi-word region parsing, negation across phrases such as not Africa or Asia, exclusion penalties, budget-conflict handling, and confidence gating for explicit hard failures.
