# Travel Destination Chatbot

A conversational chatbot that recommends travel destinations based on the user's preferences. Instead of browsing endless lists, the user answers a few natural-language questions and receives a personalised ranked list of destinations tailored to their lifestyle.

**Authors:** Lucia Nuñez, Helena Sales, Emircan Teymur, Jan Ryska, Stanislaw Gancarczyk  
**Course:** Knowledge-Based Systems - La Salle, Universitat Ramon Llull

---

## How It's Made

**Tech used:** Python, NLTK, pandas

The chatbot is built in two stages. The first stage is a **rule-based filter** that uses an NLP pipeline to understand the user's answers in natural language. Each answer goes through tokenization ('nltk.word_tokenize'), POS tagging ('nltk.pos_tag'), and lemmatization ('WordNetLemmatizer') before being matched against keyword dictionaries for continent, country, duration, budget, climate, and season. Fuzzy logic and MYCIN-style certainty factors then turn each criterion into graded, signed evidence rather than a hard pass/fail.

The second stage is a **cosine similarity ranking**. The user selects which lifestyle categories matter to them (culture, adventure, nature, beaches, nightlife, cuisine, wellness, urban, seclusion) and rates each one 1–5. This builds a preference vector which is compared against each destination's lifestyle vector using a cosine similarity we implement ourselves (plain dot-product and norms, no scipy/sklearn), scaled by a city-strength factor. The destinations are then sorted and the top 5 are returned with an explanation.

The chatbot runs from the command line (`python3 main.py`) or as a Streamlit web app (`streamlit run app.py`), reading from a CSV dataset of worldwide travel cities that includes climate data, budget levels, lifestyle ratings, and ideal trip durations.

---

## Optimizations

The ranking function was improved to only include lifestyle dimensions that the user actually selected, rather than filling unselected ones with zeros. This means unselected categories have zero influence on the similarity score rather than quietly dragging results in the wrong direction, a city that scores high in categories the user doesn't care about would unfairly rank higher otherwise.

The extraction functions were also given a safety net that checks all words in the user's input regardless of POS tag. This handles cases where NLTK mis-tags a word and would otherwise silently miss it.

---

## Lessons Learned

Building this project made it very clear how fragile keyword-based NLP actually is, the chatbot works well when the user types expected words, but anything slightly outside the keyword dictionaries gets missed entirely. We added a conservative spelling corrector and broad synonym dictionaries to soften this, but there is still no real understanding of meaning beyond the vocabulary we hand-build.

One thing that really clicked for us was using cosine similarity for the ranking, once we realised we could represent both the user's preferences and each city's profile as a vector and just measure how closely they point in the same direction, it felt like a natural fit for the problem. It also has the nice property of being easy to explain, which matters when you want to justify why a destination ended up at the top.

We also learned that the NLP pipeline from the course materials (tokenize -> POS tag -> lemmatize -> match) translates well to real projects, but needs extra robustness like the safety net fallback when applied to unpredictable user input.
