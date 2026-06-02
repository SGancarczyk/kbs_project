# chatbot.py
# ----------------------------------------------------------------------------
# THE CHATBOT - the user interface of the Knowledge-Based System. Its job is to
# hold a real conversation: understand free-typed sentences (via nlp.py), fill
# in the user's wishes one piece ("slot") at a time, and hand the finished
# request to the inference engine, then explain the answer. There are NO
# numbered menus - the project rules forbid that. The bot also recognises
# several "intents" (greet, help, why, restart, exit) and gives many different
# responses, satisfying the "at least 15 different responses" requirement.
#
# IMPROVEMENTS over v1:
#   - New "season" slot: asks when the user is travelling so the inference
#     engine can use monthly temperature data instead of yearly average.
#   - Richer word recognition via the expanded knowledge_base vocabularies.
#   - More natural, varied dialog: small-talk recognition, calmer rotating
#     openers, conversational questions that invite full sentences, and more
#     varied clarification replies.
# ----------------------------------------------------------------------------
import random
import knowledge_base as kb
import nlp
import inference as inf
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------------------------
# LIFESTYLE TF-IDF MATCHER
# Vectorizes the LIFESTYLE_DESCRIPTIONS corpus once at import time so every
# call to match_lifestyle_dims() is just a transform + dot product — fast.
# Using the same scipy cosine approach as chatbot__4_.ipynb.
# ----------------------------------------------------------------------------
_lifestyle_dims = list(kb.LIFESTYLE_DESCRIPTIONS.keys())
_tfidf_vectorizer = TfidfVectorizer()
_tfidf_matrix = _tfidf_vectorizer.fit_transform(
    list(kb.LIFESTYLE_DESCRIPTIONS.values())
)
# Minimum cosine similarity score for a dimension to count as mentioned.
# 0.05 is intentionally low — better to ask "how much do you care about X?"
# than to silently miss something the user mentioned.
_LIFESTYLE_THRESHOLD = 0.05

# Intensity phrases ordered longest-first within each weight so "a little bit"
# is matched before "bit". Checked in descending weight order so the highest
# match wins.
# Intensity vocabulary for per-dimension rating extraction.
# Stored as a flat list sorted longest-phrase-first so "don't really care"
# always beats "really" when both appear in the same segment.
_INTENSITY_WORDS = {
    5: ["must have", "must", "essential", "obsessed", "absolutely",
        "most important", "top priority", "number one", "mainly",
        "above all", "main focus", "all about", "crucial"],
    4: ["really important", "very important", "really love", "really like",
        "love", "very much", "big", "major", "important", "definitely",
        "a lot", "so much", "super", "key", "high priority"],
    3: ["enjoy", "nice to have", "fine with", "like", "nice", "decent",
        "moderate", "okay", "ok", "alright", "not bad", "some"],
    2: ["a little bit", "little bit", "not that important", "not very",
        "don't really care", "dont really care", "not too", "not really",
        "not much", "barely", "hardly", "secondary", "minor",
        "if possible", "a little", "slight", "really"],
    1: ["don't care about", "dont care about", "don't care", "dont care",
        "irrelevant", "doesn't matter", "doesnt matter", "not important",
        "not interested", "couldn't care", "couldnt care"],
}
# Pre-sorted flat list: (weight, phrase) longest phrase first
_INTENSITY_SORTED = sorted(
    [(w, p) for w, phrases in _INTENSITY_WORDS.items() for p in phrases],
    key=lambda x: len(x[1]), reverse=True
)
# Phrases meaning "all detected dims" vs "all unmentioned dims"
_ALL_PHRASES  = {"all of them", "all of these", "all equally",
                 "all the same", "every one", "each one", "everything"}
_REST_PHRASES = {"the rest", "the others", "everything else",
                 "the other ones", "the remaining", "others", "rest of them"}


def _match_lifestyle_dims(text):
    # Return a dict of {dimension: cosine_score} for every dimension whose
    # TF-IDF description is similar enough to the user's free text.
    # Uses cosine similarity — same approach as chatbot__4_.ipynb
    # rank_by_lifestyle / cosine_similarity.
    user_vec = _tfidf_vectorizer.transform([text.lower()])
    matched = {}
    for i, dim in enumerate(_lifestyle_dims):
        dim_vec = _tfidf_matrix[i]
        dot    = (user_vec * dim_vec.T).toarray()[0][0]
        norm_u = float((user_vec.multiply(user_vec)).sum() ** 0.5)
        norm_d = float((dim_vec.multiply(dim_vec)).sum() ** 0.5)
        if norm_u == 0 or norm_d == 0:
            continue
        score = dot / (norm_u * norm_d)
        if score >= _LIFESTYLE_THRESHOLD:
            matched[dim] = score
    return dict(sorted(matched.items(), key=lambda x: -x[1]))


def _score_segment(segment):
    # Score a single text chunk: digit 1-5 wins, otherwise longest
    # intensity phrase match. Returns None if nothing found.
    import re as _re
    numbers = _re.findall(r'\b([1-5])\b', segment)
    if numbers:
        return int(numbers[0])
    for weight, phrase in _INTENSITY_SORTED:
        if phrase in segment:
            return weight
    return None


def _extract_per_dim_ratings(text, detected_dims):
    # Parse a free-text "how much" answer and return {dim: weight 1-5}
    # for each detected dimension mentioned.
    #
    # Strategy:
    #   1. Split the answer into clauses by comma / conjunction / "except"
    #   2. Each clause is scored independently (_score_segment)
    #   3. Clauses with a dim name get that score assigned to that dim
    #   4. "all of them are a 4" -> global score for unrated dims
    #   5. "the rest don't really matter" -> rest score for unmentioned dims
    import re as _re
    t        = text.lower()
    segments = [s.strip() for s in
                _re.split(r'[,;]|\band\b|\bbut\b|\bthough\b|\bwhereas\b|\balso\b|\bexcept\b', t)
                if s.strip()]
    ratings      = {}
    global_score = None
    rest_score   = None

    for seg in segments:
        dims_in_seg = [d for d in detected_dims if d in seg]
        score       = _score_segment(seg)
        is_all      = any(p in seg for p in _ALL_PHRASES)
        is_rest     = any(p in seg for p in _REST_PHRASES)

        if is_all and not dims_in_seg and score is not None:
            global_score = score
        elif is_rest and not dims_in_seg and score is not None:
            rest_score = score
        elif dims_in_seg and score is not None:
            for d in dims_in_seg:
                ratings[d] = score
        elif not dims_in_seg and score is not None and global_score is None:
            global_score = score   # bare unattributed score — use as fallback

    # "the rest" applies to explicitly unmentioned dims (more specific than global)
    if rest_score is not None:
        for d in detected_dims:
            if d not in ratings:
                ratings[d] = rest_score
    # Global score fills anything still unrated
    if global_score is not None:
        for d in detected_dims:
            if d not in ratings:
                ratings[d] = global_score

    return ratings

# Words that mean "I have no preference for the thing you just asked" - they let
# the user SKIP a slot instead of being forced to answer (un-menu-like).
SKIP_WORDS = {"any", "anywhere", "whatever", "skip", "none", "no", "nope", "idk", "anything",
             # Added: filler/function words that appear alongside skip-intent words
             # e.g. "i don't mind" -> ["i", "do", "n't", "mind"] -> "i","do","mind" after isalpha()
             "i", "do", "dont", "mind", "open", "to", "fine", "sure",
             "ok", "okay", "good", "up", "you", "me", "just", "not",
             "really", "care", "worry", "worries", "bother", "fussed",
             "doesnt", "does", "matter", "n't"}
SKIP_PHRASES = {"doesn't matter", "does not matter", "dont care", "don't care", "do not care", "no preference", "not sure",
                # Apostrophe-free / texting variants so casual phrasing still
                # registers as "no preference" instead of confusing the bot.
                "doesnt matter", "doesnt", "dnt care", "dnt matter",
                "idgaf", "idc", "no pref", "np",
                # Extra casual variants that the original list missed.
                "whatever works", "no idea", "surprise me", "up to you",
                "not bothered", "not fussed", "doesn't bother me",
                "not really", "not really sure"}

# ----------------------------------------------------------------------------
# Cues that signal the user is REJECTING a city we just recommended (used by the
# "reject_city" intent). We only treat a message as a rejection when one of
# these appears AND it points at a city that is currently on screen, so a normal
# preference like "warm" is never mistaken for a rejection.
REJECTION_CUES = ("not ", "no ", "remove", "don't want", "dont want",
                  "do not want", "without", "exclude", "skip", "drop",
                  "rather not", "get rid", "anything but", "take out",
                  "i hate", "lose the")

# Map a positional phrase the user might type ("the first one", "the top pick")
# onto an index into the list of currently shown recommendations. "last" is
# resolved separately because its index depends on how many picks are shown.
_POSITION_WORDS = {
    "first": 0, "top": 0, "1st": 0, "one": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
}

# Human-friendly phrasings used by the rewritten, more natural _acknowledge().
# We keep the dataset's exact labels on the left and a conversational phrase on
# the right, so confirmations read like a person talking rather than a form.
_BUDGET_PHRASES = {
    "Budget": "on a budget",
    "Mid-range": "mid-range",
    "Luxury": "luxury",
}
_DURATION_PHRASES = {
    "Day trip": "a day trip",
    "Weekend": "a weekend getaway",
    "Short trip": "a short trip",
    "One week": "a week-long trip",
    "Long trip": "a long trip",
}

# Rotating openers so repeated confirmations do not all sound identical.
# Kept calm and neutral rather than enthusiastic so they feel natural across
# a longer conversation without becoming repetitive or over-eager.
_ACK_OPENERS = [
    "Got it.",
    "That helps.",
    "Sounds good.",
    "Thanks for sharing.",
    "Good to know.",
    "I'll keep that in mind.",
    "Okay.",
    "Makes sense.",
]

# ----------------------------------------------------------------------------
# SMALL-TALK RECOGNITION.
# Maps frozensets of known casual phrases to a list of possible replies.
# Using frozensets lets us do fast membership tests while keeping the lookup
# table easy to read and extend. A random reply is chosen each time so the
# bot does not sound repetitive. An empty string in the replies list means
# "fall through to normal preference parsing" - useful for ambiguous one-word
# responses like "cool" that might also be a preference.
# ----------------------------------------------------------------------------
_SMALL_TALK = {
    frozenset(["hi", "hello", "hey", "yo", "sup", "howdy", "hiya", "heya"]): [
        "Hi! I'd love to help you plan your next trip.",
        "Hello! Looking for travel ideas? Just tell me what kind of trip you have in mind.",
        "Hey! Tell me a bit about the kind of trip you're looking for and I'll help narrow things down.",
    ],
    frozenset(["how are you", "how r u", "how are u", "how you doing", "hows it going",
               "hows everything", "how's things", "you ok", "you good", "all good",
               "i am good", "i am well", "i'm good", "i'm well", "i am good thanks",
               "i'm doing well", "doing well", "doing good"]): [
        "Doing well, thanks for asking. What kind of trip are you thinking about?",
        "All good here. What are you looking for in a trip?",
        "Good thanks. Now, what kind of destination are you after?",
    ],
    frozenset(["thanks", "thank you", "thx", "ty", "cheers", "appreciate it", "thanks!",
               "thank you!", "thanks so much", "many thanks"]): [
        "Glad I could help.",
        "Of course.",
        "No problem at all.",
    ],
    frozenset(["cool", "nice", "great", "awesome", "ok", "okay", "good",
               "sounds good", "perfect", "sure"]): [
        "Good to know.",
        "Glad that works.",
        "",  # empty -> fall through to preference parsing
    ],
    frozenset(["lol", "haha", "hehe", "lmao"]): [
        "Ha. Anyway, back to your trip.",
        "Right, where were we.",
    ],
    frozenset(["not sure", "i don't know", "i dont know", "idk", "no idea", "dunno"]): [
        "No problem. Let's figure it out together. Do you have a rough idea of what kind of weather you prefer, or what you like doing on holiday?",
        "That's fine. Even a small detail helps, like whether you prefer beaches, cities, or mountains.",
    ],
}

# ----------------------------------------------------------------------------
# VARIED "I DIDN'T UNDERSTAND" REPLIES.
# Rotating list so the bot never repeats the same confused message twice in a
# row, which would feel robotic. random.choice() picks one each time.
# These are written to invite a fuller sentence rather than a single keyword.
# ----------------------------------------------------------------------------
_CONFUSED_REPLIES = [
    ("I'm not quite sure what you mean. Could you tell me a little more about "
     "the kind of trip you're looking for? Things like a destination, budget, "
     "travel dates, or activities all help."),
    ("I didn't quite understand that. You can tell me about destinations, budget, "
     "weather preferences, travel dates, or the kinds of activities you enjoy."),
    ("I didn't catch anything I recognise there. Try describing the trip in your "
     "own words, for example, what you enjoy doing, roughly where you want to go, "
     "or how long you have. Or type 'go' to see picks with what I have so far."),
    ("I'm not sure I followed that. The more you describe about the trip, even "
     "just the kind of atmosphere or activities you enjoy, the better I can match "
     "you to somewhere."),
]


def _pretty_region(region):
    # Turn a stored region key like "north_america" into "North America" for
    # display. Underscores become spaces and each word is capitalised.
    return region.replace("_", " ").title()


def _join_natural(items):
    # Join a list into natural English: ["a"] -> "a"; ["a","b"] -> "a and b";
    # ["a","b","c"] -> "a, b, and c". Used everywhere we list things to the user.
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return items[0] + " and " + items[1]
    return ", ".join(items[:-1]) + ", and " + items[-1]


class TravelChatbot:
    def __init__(self, destinations):
        # Keep a reference to the Knowledge Base (the 560 destinations) and start
        # a fresh, empty conversation.
        self.destinations = destinations
        self.reset()

    def reset(self):
        # The "query" is the structured request we slowly build from the chat.
        # It is exactly the dict shape inference.recommend() expects.
        self.query = {
            "regions_include": [],
            "regions_exclude": [],
            # NEW: country-level include/exclude mirrors region include/exclude
            # but filters on the dataset "country" column for exact country pins.
            "countries_include": [],
            "countries_exclude": [],
            "climate": [],
            "climate_exclude": [],
            "budget": [],
            "budget_exclude": [],
            "duration": [],
            "duration_exclude": [],
            "lifestyle": {},
            "lifestyle_exclude": [],
            # travel_months stores the month numbers (1-12) extracted from
            # the season slot so the inference engine can use seasonal temperature
            # data instead of the yearly average when scoring climate.
            "travel_months": [],
        }
        # Conversation bookkeeping.
        self.resolved = set()        # slots the user has answered OR skipped
        self.pending_slot = None     # the slot we asked about most recently
        self.last_results = None     # remember the last recommendation for "why"
        self.finished = False        # set True when the user wants to quit
        # MEMORY of cities the user has told us to drop. Once a city is in here
        # it is filtered out of every future recommendation for this session, so
        # the user never has to reject the same place twice. Cleared on restart
        # because reset() rebuilds it as an empty set.
        self.rejected_cities = set()
        # Tracks which slots the user explicitly skipped (answered with a skip
        # phrase like "open to anything") so _slot_filled() can distinguish
        # "not yet answered" from "deliberately left open".
        self._skipped = {}
        # Counter used to rotate the friendly openers in _acknowledge() so the
        # confirmations vary instead of always starting the same way.
        self._ack_index = 0

    # --- The ordered list of slots the bot tries to fill, each with a question.
    # A slot counts as "done" when its part of the query is filled or skipped.
    # The "season" slot is new - it sits after continent so we know WHERE the
    # user is going before we ask WHEN, which makes the question feel natural.
    # Questions are written to invite a full sentence rather than a keyword.
    def _slots(self):
        return [
            # continent is asked first; if the user already named a country in
            # the same message the continent slot fills automatically via
            # country_to_region() in _update_slots, so this question is skipped.
            ("continent", "Do you already have a region in mind? You can mention somewhere specific like Europe or Southeast Asia, or just say you're open to anything."),
            # country is asked after continent. Skipped if a country was already
            # named (slot filled) OR if the user skipped the continent question
            # (meaning they are open to anywhere — no point narrowing to a country).
            ("country",   "Is there a particular country you have in mind, or are you happy to explore across a whole region?"),
            ("season",    "When are you hoping to travel? Even a rough idea like a season or month is helpful."),
            ("duration",  "How long are you thinking of going for?"),
            ("budget",    "What kind of budget are you working with? For example, are you looking to keep costs down, or is it more of a treat yourself trip?"),
            ("climate",   "Do you have a preference for the weather? Warm and sunny, cooler, or somewhere in between?"),
            # lifestyle is split into two conversational turns:
            #   lifestyle_what     -> TF-IDF detects WHICH dimensions matter
            #   lifestyle_how_much -> asks intensity only for detected dimensions
            ("lifestyle_what",      "What kind of experiences do you look for on a trip?\n"                                    "Here's what I can match you on: culture, adventure, nature, beaches, nightlife, cuisine, wellness, urban, seclusion.\n"                                    "Just tell me which ones appeal to you, pick a few, describe them, or say something like \"beaches and good food, not too much nightlife\"."),
            ("lifestyle_how_much",  ""),   # question built dynamically in _next_question()
        ]

    def _slot_filled(self, slot):
        # Has this slot already received a value in the query, either as a wish
        # or as an explicit exclusion?
        if slot == "continent":
            return bool(self.query["regions_include"]) or bool(self.query["regions_exclude"])
        # The country slot is filled when:
        #   a) at least one country was already named (include or exclude), OR
        #   b) the continent question was explicitly skipped — if the user said
        #      "open to anything" at the continent stage they clearly don't want
        #      to narrow down to a country either, so don't ask.
        # We do NOT skip when regions_include is filled — that's exactly the case
        # where we want to ask "any specific country within that region?"
        if slot == "country":
            country_given = bool(self.query["countries_include"]) or bool(self.query["countries_exclude"])
            continent_skipped = (not self.query["regions_include"] and
                                 not self.query["regions_exclude"] and
                                 self._skipped.get("continent", False))
            return country_given or continent_skipped
        # The season slot is filled when at least one travel month was extracted.
        if slot == "season":
            return bool(self.query["travel_months"])
        if slot == "lifestyle_what":
            # Filled once we have detected at least one dimension OR the user
            # explicitly skipped (nothing to follow up on).
            return (bool(self.query["lifestyle"]) or
                    bool(self.query["lifestyle_exclude"]) or
                    self._skipped.get("lifestyle_what", False))
        if slot == "lifestyle_how_much":
            # Only meaningful if lifestyle_what found some dimensions to ask
            # about. Filled once weights have been set OR skipped.
            pending = self._lifestyle_pending_dims()
            return not pending or self._skipped.get("lifestyle_how_much", False)
        if slot == "climate":
            return bool(self.query["climate"]) or bool(self.query["climate_exclude"])
        if slot == "budget":
            return bool(self.query["budget"]) or bool(self.query["budget_exclude"])
        if slot == "duration":
            return bool(self.query["duration"]) or bool(self.query["duration_exclude"])
        return bool(self.query[slot])

    def greeting(self):
        # The opening message (RESPONSE TYPE 1).
        return (
            "Hi! I'm here to help you find a trip that fits what you're looking for. "
            "Tell me as much or as little as you like about your ideal getaway, and "
            "I'll suggest some destinations. You can mention things like where you'd "
            "like to go, your budget, how long you have, the kind of weather you enjoy, "
            "or the activities you're interested in.\n\n"
            "No need to answer everything at once. We can figure it out step by step.\n\n"
            "A few helpful commands:\n"
            "- skip: move past the current question\n"
            "- skip all: jump straight to destination ideas\n"
            "- go: get recommendations based on what we've discussed\n"
            "- why: see why I suggested a destination\n"
            "- restart: start over\n"
            "- exit: end the conversation\n\n"
            + self._next_question()
        )

    def help_text(self):
        # The help message (RESPONSE TYPE 2).
        return ("I match you to travel destinations from a database of 560 cities. "
                "Just describe what you want in your own words, for example: "
                "'a cheap warm week in Asia, I love food and culture'. "
                "I understand full sentences, fix small typos, and handle negation "
                "like 'not Africa' or 'avoid nightlife'.\n"
                "Commands: 'go' or 'skip all' (get picks now), 'why' (see reasoning), "
                "'skip' (skip one question), 'restart', 'exit'.")

    def _detect_intent(self, text):
        # Decide what the user is trying to do. We check the most specific
        # intents first so, e.g., "start over" is a restart, not a preference.
        t = text.lower().strip()
        if t in {"exit", "quit", "bye", "goodbye", "stop"} or t.startswith("exit"):
            return "exit"
        if t in {"help", "?"} or "how do you work" in t or "what can you do" in t:
            return "help"
        if "restart" in t or "start over" in t or "reset" in t or "new search" in t or "search again" in t or "start again" in t:
            return "restart"
        if t.startswith("why") or "explain" in t or "reason" in t:
            return "why"
        if t in {"go", "done", "finish", "enough", "recommend", "show me", "results"} or "that's all" in t or "thats all" in t or "show me" in t:
            return "recommend"
        if t in {"hi", "hello", "hey", "selam", "yo"}:
            return "greet"
        # SMALL TALK: check before preference parsing so casual remarks get a
        # warm reply rather than a confused "I didn't understand that".
        if self._detect_small_talk(t) is not None:
            return "small_talk"
        # REJECTION: only possible once we have actually shown some picks. We
        # require a rejection cue ("not", "remove", ...) AND that it points at a
        # city/position currently on screen, so an ordinary preference such as
        # "warm" is never misread as "remove a city".
        if self.last_results and self.last_results.get("results"):
            if self._has_rejection_cue(t) and self._cities_to_reject(text):
                return "reject_city"
        # Anything else is treated as the user giving us preferences.
        return "provide"

    def _detect_small_talk(self, lowered):
        # Return the list of possible replies if the lowercased message matches
        # any small-talk entry, otherwise return None. We check exact membership
        # first (fast), then do a substring scan for longer phrases.
        for phrase_set, replies in _SMALL_TALK.items():
            if lowered in phrase_set:
                return replies
        for phrase_set, replies in _SMALL_TALK.items():
            for phrase in phrase_set:
                if len(phrase) > 4 and phrase in lowered:
                    return replies
        return None

    def _has_rejection_cue(self, t):
        # True if the (already lowercased) message contains any phrase that
        # signals the user wants something taken off the list.
        return any(cue in t for cue in REJECTION_CUES)

    def _cities_to_reject(self, text):
        # Work out WHICH currently-shown cities the message refers to. We only
        # ever look at the picks we last displayed (self.last_results), which is
        # exactly the list the user can see, so references stay unambiguous.
        if not self.last_results or not self.last_results.get("results"):
            return []
        t = text.lower()
        shown = [dest["city"] for _, dest, _ in self.last_results["results"]]
        to_reject = []
        # "exclude those cities" / "none of these" / "drop all of them" -> the
        # user is rejecting the WHOLE current list at once.
        if "those" in t or "these" in t or "all of them" in t or "all of these" in t:
            return list(shown)
        # Positional references like "not the first one" / "remove the top pick".
        for word, idx in _POSITION_WORDS.items():
            if word in t and idx < len(shown) and shown[idx] not in to_reject:
                to_reject.append(shown[idx])
        # "the last one" -> the final pick on screen.
        if "last" in t and shown and shown[-1] not in to_reject:
            to_reject.append(shown[-1])
        # Direct city-name mentions ("not Paris", "remove Barcelona").
        for city in shown:
            if city.lower() in t and city not in to_reject:
                to_reject.append(city)
        return to_reject

    def _is_skip(self, text):
        # True if the whole message means "no preference here".
        t = text.lower().strip()
        # Normalize informal shorthand first so "idc", "w/e" etc. are caught.
        t = nlp._normalize_common_phrases(t)
        if t in SKIP_PHRASES:
            return True
        # REWRITE: use word_tokenize instead of t.split() so contractions are
        # split correctly — "don't" becomes ["do", "n't"] rather than staying as
        # a single token that SKIP_WORDS doesn't contain. This means "i don't
        # mind", "open to anything" etc. are now correctly identified as skips.
        tokens = [tok for tok in nlp.tokenize(t) if tok.isalpha()]
        return len(tokens) > 0 and all(tok in SKIP_WORDS for tok in tokens)

    def _is_skip_all(self, text):
        # True if the user wants to skip ALL remaining questions and just get
        # picks now (e.g. "skip all", "skip the rest", "skip everything").
        t = text.lower().strip()
        if t in {"skip all", "skip everything", "skip the rest", "skip rest", "just recommend"}:
            return True
        return "skip" in t and ("all" in t or "rest" in t or "everything" in t)

    def _blocking_vocabulary_for(self, keyword_map):
        # All vocabulary keys except the current slot. Used by nlp.extract so a
        # negation aimed at another category does not leak into this category.
        # SEASON_SYNONYMS is included so e.g. "not summer" never bleeds into
        # a continent or lifestyle extraction.
        all_maps = [
            kb.CONTINENT_SYNONYMS,
            kb.COUNTRY_SYNONYMS,   # NEW: country keywords block each other's negation
            kb.CLIMATE_SYNONYMS,
            kb.BUDGET_SYNONYMS,
            kb.DURATION_SYNONYMS,
            kb.LIFESTYLE_SYNONYMS,
            kb.SEASON_SYNONYMS,
        ]
        keys = set()
        for m in all_maps:
            if m is not keyword_map:
                keys.update(m.keys())
        return keys

    def _lifestyle_pending_dims(self):
        # Dimensions detected in lifestyle_what that haven't had their weight
        # set yet. These are the ones we still need to ask "how much" about.
        # A dimension is pending if it's in lifestyle but still at the default
        # detection score (we stored the raw TF-IDF score as a float < 1.0
        # temporarily; once the user answers "how much" we replace it with 1-5).
        return [d for d, w in self.query["lifestyle"].items() if isinstance(w, float)]

    def _lifestyle_how_much_question(self):
        # Build a natural "how much do you care about X?" question covering only
        # the pending dimensions. Invites a single free-text answer that rates
        # all dims at once rather than asking one-by-one.
        pending = self._lifestyle_pending_dims()
        if not pending:
            return None
        if len(pending) == 1:
            dim = pending[0]
            return (f"How important is {dim} to you on this trip? "
                    f"You can say something like \"{dim} is a must\" or "
                    f"\"{dim} is maybe a 3 out of 5\", whatever feels natural.")
        dims_str = _join_natural(pending)
        return (f"You mentioned {dims_str}. How important is each one to you? "
                f"Feel free to rate them 1-5, or just describe it, "
                f"like \"cuisine is a must, nightlife maybe a 3, "
                f"culture I don't really care about\".")

    def _has_meaningful_preferences(self):
        # Do not recommend from an empty query, because that only returns dataset
        # order with zero evidence. At least one include, exclude, or interest is
        # needed for a meaningful recommendation.
        return any([
            self.query["regions_include"], self.query["regions_exclude"],
            self.query["countries_include"], self.query["countries_exclude"],   # NEW
            self.query["climate"], self.query["climate_exclude"],
            self.query["budget"], self.query["budget_exclude"],
            self.query["duration"], self.query["duration_exclude"],
            self.query["lifestyle"], self.query["lifestyle_exclude"],
            # float placeholders count — dimension was detected even if not yet weighted
            # travel_months alone counts as a meaningful preference because the
            # inference engine will use it to adjust climate scoring.
            self.query["travel_months"],
        ])

    def _extract_season(self, text):
        # Run the NLP extractor over the SEASON_SYNONYMS vocabulary and convert
        # every matched key ("july", "summer", ...) into a list of month numbers
        # via knowledge_base.season_to_months().
        # names are short and exact matches are safer here.
        result = nlp.extract(
            text, kb.SEASON_SYNONYMS,
            blocking_vocabulary=self._blocking_vocabulary_for(kb.SEASON_SYNONYMS),
        )
        months = []
        for key in result["include"]:
            for m in kb.season_to_months(key):
                if m not in months:
                    months.append(m)
        return months

    def _update_slots(self, text):
        # Run the NLP extractor once per vocabulary and merge whatever it finds
        # into the query. Exclusions are kept too, so "not expensive" and
        # "avoid nightlife" affect the final reasoning instead of being lost.
        changed = {
            "regions_include": [], "regions_exclude": [],
            # NEW: tracks countries newly added this turn for _acknowledge().
            "countries_include": [], "countries_exclude": [],
            "climate": [], "climate_exclude": [],
            "budget": [], "budget_exclude": [],
            "duration": [], "duration_exclude": [],
            "lifestyle": [], "lifestyle_exclude": [],
            # tracks which travel months were newly added this turn so
            # _acknowledge() can confirm them back to the user.
            "travel_months": [],
        }

        cont = nlp.extract(text, kb.CONTINENT_SYNONYMS, blocking_vocabulary=self._blocking_vocabulary_for(kb.CONTINENT_SYNONYMS))
        for v in cont["include"]:
            if v not in self.query["regions_include"]:
                self.query["regions_include"].append(v)
                changed["regions_include"].append(v)
        for v in cont["exclude"]:
            if v not in self.query["regions_exclude"]:
                self.query["regions_exclude"].append(v)
                changed["regions_exclude"].append(v)

        # COUNTRY: tokens only match COUNTRY_SYNONYMS (no overlap with
        # CONTINENT_SYNONYMS), so "japan" always resolves to the country Japan
        # and never accidentally fills the continent slot with "asia" directly.
        # When a country IS found, we derive its region and add it to
        # regions_include automatically — this fills the continent slot so the
        # bot never asks "which region?" after the user already named a country.
        # Multiple countries from different continents are fully supported: each
        # country appends its own region, so regions_include may end up with
        # ["asia", "south_america"] if the user names e.g. Japan and Brazil.
        cntry = nlp.extract(text, kb.COUNTRY_SYNONYMS,
                            blocking_vocabulary=self._blocking_vocabulary_for(kb.COUNTRY_SYNONYMS))
        for v in cntry["include"]:
            if v not in self.query["countries_include"]:
                self.query["countries_include"].append(v)
                changed["countries_include"].append(v)
            # Derive region and add to regions_include if not already present.
            # This fills the continent slot automatically and supports multiple
            # countries from different continents in the same query.
            region = kb.country_to_region(v)
            if region and region not in self.query["regions_include"]:
                self.query["regions_include"].append(region)
                changed["regions_include"].append(region)
        for v in cntry["exclude"]:
            if v not in self.query["countries_exclude"]:
                self.query["countries_exclude"].append(v)
                changed["countries_exclude"].append(v)
            # Also exclude the region if ALL countries in that region are excluded
            # (edge case — left to inference engine to handle gracefully).

        clim = nlp.extract(text, kb.CLIMATE_SYNONYMS, blocking_vocabulary=self._blocking_vocabulary_for(kb.CLIMATE_SYNONYMS))
        for v in clim["include"]:
            if v not in self.query["climate"]:
                self.query["climate"].append(v)
                changed["climate"].append(v)
        for v in clim["exclude"]:
            if v not in self.query["climate_exclude"]:
                self.query["climate_exclude"].append(v)
                changed["climate_exclude"].append(v)

        bud = nlp.extract(text, kb.BUDGET_SYNONYMS, blocking_vocabulary=self._blocking_vocabulary_for(kb.BUDGET_SYNONYMS))
        for v in bud["include"]:
            if v not in self.query["budget"]:
                self.query["budget"].append(v)
                changed["budget"].append(v)
        for v in bud["exclude"]:
            if v not in self.query["budget_exclude"]:
                self.query["budget_exclude"].append(v)
                changed["budget_exclude"].append(v)

        # DURATION: relies on exact token/lemma matches only (no spell correction).
        # Duration keywords are short and prone to false positives if corrected.
        dur = nlp.extract(text, kb.DURATION_SYNONYMS, blocking_vocabulary=self._blocking_vocabulary_for(kb.DURATION_SYNONYMS))
        for v in dur["include"]:
            if v not in self.query["duration"]:
                self.query["duration"].append(v)
                changed["duration"].append(v)
        for v in dur["exclude"]:
            if v not in self.query["duration_exclude"]:
                self.query["duration_exclude"].append(v)
                changed["duration_exclude"].append(v)

        # LIFESTYLE — two-phase extraction:
        #
        # Phase 1 (lifestyle_what): TF-IDF cosine similarity detects WHICH
        # dimensions the user cares about from their free-text description.
        # Detected dimensions are stored with their raw cosine score as a float
        # placeholder (e.g. 0.37). The float signals "detected but not yet
        # weighted" so _lifestyle_pending_dims() can find them.
        #
        # Phase 2 (lifestyle_how_much): intensity words in the follow-up answer
        # ("main focus", "nice to have", etc.) map to weights 1-5, replacing
        # the float placeholder with the final integer weight.
        #
        # Exclusions ("I hate nightlife") are still handled by nlp.extract +
        # LIFESTYLE_SYNONYMS so negation logic stays intact.

        if self.pending_slot == "lifestyle_what":
            # Phase 1: detect dimensions via TF-IDF.
            detected = _match_lifestyle_dims(text)
            # Run exclusion extraction FIRST so we know which dims were negated.
            # TF-IDF sees "nightlife" in "not too much nightlife" and scores it —
            # but nlp.extract correctly marks it as excluded. By running exclusions
            # first, we can skip adding negated dims to lifestyle includes.
            life_excl = nlp.extract(text, kb.LIFESTYLE_SYNONYMS,
                                    blocking_vocabulary=self._blocking_vocabulary_for(kb.LIFESTYLE_SYNONYMS))
            for v in life_excl["exclude"]:
                if v not in self.query["lifestyle_exclude"]:
                    self.query["lifestyle_exclude"].append(v)
                    changed["lifestyle_exclude"].append(v)
            # Extract any explicit intensity ratings from the same answer so we
            # can store integers immediately. This avoids the lifestyle_how_much
            # follow-up when the user has already given enough information.
            included_dims = [d for d in detected if d not in self.query["lifestyle_exclude"]]
            ratings = _extract_per_dim_ratings(text, included_dims)
            # Now add detected dims, skipping any that were just excluded.
            for dim in included_dims:
                if dim not in self.query["lifestyle"]:
                    # Use explicit rating if the user expressed one, else default 3.
                    self.query["lifestyle"][dim] = ratings.get(dim, 3)
                    changed["lifestyle"].append(dim)

        elif self.pending_slot == "lifestyle_how_much":
            # Phase 2: per-dimension rating extraction.
            # Parse the user's free-text answer to assign a weight 1-5 to each
            # pending dimension. "cuisine is a 5, nightlife maybe a 3, culture
            # I don't really care" -> cuisine:5, nightlife:3, culture:2.
            pending = self._lifestyle_pending_dims()
            ratings = _extract_per_dim_ratings(text, pending)
            for dim, weight in ratings.items():
                self.query["lifestyle"][dim] = weight
                if dim not in changed["lifestyle"]:
                    changed["lifestyle"].append(dim)
            # Any pending dim not mentioned gets default weight 3 (moderate).
            for dim in pending:
                if dim not in ratings:
                    self.query["lifestyle"][dim] = 3
                    if dim not in changed["lifestyle"]:
                        changed["lifestyle"].append(dim)
            # Also check for any NEW dimensions mentioned in this answer
            # (e.g. "food is a 5 and actually beaches too") — add them directly
            # with the score parsed from their clause.
            extra = _match_lifestyle_dims(text)
            for dim, score in extra.items():
                if dim not in self.query["lifestyle"]:
                    extra_ratings = _extract_per_dim_ratings(text, [dim])
                    self.query["lifestyle"][dim] = extra_ratings.get(dim, 3)
                    changed["lifestyle"].append(dim)

        # else: outside the lifestyle slots — do nothing.
        # Lifestyle is only extracted when the bot is actively on
        # lifestyle_what or lifestyle_how_much, so earlier answers
        # (continent, budget, etc.) never silently fill lifestyle prefs.

        # SEASON: month numbers are stored
        # in the query; the inference engine picks up the right monthly temp data.
        months = self._extract_season(text)
        for m in months:
            if m not in self.query["travel_months"]:
                self.query["travel_months"].append(m)
                changed["travel_months"].append(m)

        return changed

    def _next_opener(self):
        # Pick the next rotating opener (e.g. "Got it.", "Makes sense.") and
        # advance the counter, wrapping around. Rotating instead of random keeps
        # the behaviour deterministic (nice for tests) while still feeling varied.
        # (Previously e.g. "Perfect!", "Noted —" — updated to calmer phrasing.)
        opener = _ACK_OPENERS[self._ack_index % len(_ACK_OPENERS)]
        self._ack_index += 1
        return opener

    def _season_label(self, months):
        # Turn a list of month numbers back into a human-friendly string so the
        # confirmation sounds natural, e.g. [6,7,8] -> "summer", [7] -> "July".
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December",
        }
        season_map = {
            frozenset([12, 1, 2]): "winter",
            frozenset([3, 4, 5]):  "spring",
            frozenset([6, 7, 8]):  "summer",
            frozenset([9, 10, 11]): "autumn",
        }
        key = frozenset(months)
        if key in season_map:
            return season_map[key]
        if len(months) == 1:
            return month_names.get(months[0], str(months[0]))
        return _join_natural([month_names.get(m, str(m)) for m in sorted(months)])

    def _acknowledge(self, changed):
        # Build a SHORT, NATURAL confirmation of what we just understood, e.g.
        # "Got it. Europe, on a budget, and you enjoy culture and food."
        # The old version printed a robotic "regions: europe; budget: Budget.";
        # here we translate each slot value into everyday English and stitch the
        # clauses together with proper commas/"and".
        clauses = []
        # --- positive wishes (what the user DOES want) ---
        if changed["countries_include"] or changed["regions_include"]:
            # Build one natural location clause that shows countries AND any
            # regions that are not already implied by a named country.
            # e.g. "i want asia or usa" -> countries=["United States"],
            # regions=["north_america","asia"] -> show "United States and Asia"
            # because north_america is implied by United States but asia is not.
            parts = []
            # Regions implied by the named countries (don't repeat them).
            implied_regions = {kb.country_to_region(c) for c in changed["countries_include"]}
            # Show each region that came from a direct continent match (not from
            # a country derivation), so "asia" typed explicitly is always shown.
            for r in changed["regions_include"]:
                if r not in implied_regions:
                    parts.append(_pretty_region(r))
            # Show country names after regions so "Asia and United States" reads
            # continent-first, which feels natural.
            parts.extend(changed["countries_include"])
            if parts:
                clauses.append(_join_natural(parts))
        if changed["budget"]:
            # "Budget" -> "on a budget", "Luxury" -> "luxury", etc.
            clauses.append(_join_natural([_BUDGET_PHRASES.get(b, b.lower()) for b in changed["budget"]]))
        if changed["climate"]:
            # "warm" -> "warm climate" so the slot is unambiguous.
            clauses.append(_join_natural([c + " climate" for c in changed["climate"]]))
        if changed["duration"]:
            clauses.append(_join_natural([_DURATION_PHRASES.get(d, d.lower()) for d in changed["duration"]]))
        if changed["lifestyle"]:
            # Split into confirmed (int weight set) vs pending (float placeholder).
            # Confirmed dims say "you enjoy X", pending say "noted: X" since
            # the how_much question hasn't been answered yet.
            confirmed = [d for d in changed["lifestyle"]
                         if isinstance(self.query["lifestyle"].get(d), int)]
            pending   = [d for d in changed["lifestyle"]
                         if isinstance(self.query["lifestyle"].get(d), float)]
            if confirmed:
                clauses.append("you enjoy " + _join_natural(confirmed))
            if pending:
                clauses.append("noted: " + _join_natural(pending))
        # NEW: confirm travel months back to the user in human-readable form.
        if changed["travel_months"]:
            clauses.append("travelling in " + self._season_label(changed["travel_months"]))
        # --- exclusions (what the user does NOT want) ---
        if changed["countries_exclude"]:
            clauses.append("not " + _join_natural(changed["countries_exclude"]))
        if changed["regions_exclude"]:
            clauses.append("not " + _join_natural([_pretty_region(r) for r in changed["regions_exclude"]]))
        if changed["budget_exclude"]:
            clauses.append("nothing " + _join_natural([_BUDGET_PHRASES.get(b, b.lower()) for b in changed["budget_exclude"]]))
        if changed["climate_exclude"]:
            clauses.append("avoiding " + _join_natural(changed["climate_exclude"]) + " weather")
        if changed["duration_exclude"]:
            clauses.append("not " + _join_natural([_DURATION_PHRASES.get(d, d.lower()) for d in changed["duration_exclude"]]))
        if changed["lifestyle_exclude"]:
            clauses.append("steering clear of " + _join_natural(changed["lifestyle_exclude"]))
        # Nothing recognised -> signal the caller to ask for clarification.
        if not clauses:
            return None
        # Opener + the natural list of clauses, e.g.
        # "Got it. mid-range, warm climate, and you enjoy culture and food."
        # (Previously: "Noted — mid-range, warm climate, and you enjoy culture and food."
        #  and "Perfect! Europe, on a budget, and you enjoy culture and food.")
        return self._next_opener() + " " + _join_natural(clauses) + "."

    def _next_question(self):
        # Find the first slot that is neither filled nor skipped, remember it as
        # the pending slot, and return its question. If everything is handled,
        # return None to signal "ready to recommend".
        for slot, question in self._slots():
            if slot in self.resolved or self._slot_filled(slot):
                continue
            self.pending_slot = slot
            return question
        self.pending_slot = None
        return None

    def _sanitize_lifestyle_weights(self):
        # Replace any float placeholders (TF-IDF scores stored in phase 1 before
        # the user answered "how much") with the default weight 3 (moderate).
        # This ensures inference.py always receives integer weights 1-5.
        for dim, w in list(self.query["lifestyle"].items()):
            if isinstance(w, float):
                self.query["lifestyle"][dim] = 3

    def _format_recommendation(self):
        # Run the inference engine and turn its result into a friendly message
        # (RESPONSE TYPE 10), including advisory hints (forward chaining) and a
        # fallback if nothing matched (RESPONSE TYPE 11).
        if not self._has_meaningful_preferences():
            self.last_results = None
            return ("I need at least one preference before I can make a meaningful recommendation. "
                    "Tell me a region, budget, climate, trip length, interest, or something to avoid.")
        # Ask the engine for a few EXTRA picks beyond the 5 we want to show, so
        # that after we drop any rejected cities we can still backfill up to 5
        # good options instead of returning a short list.
        wanted = 5
        out = inf.recommend(self.destinations, self.query, top_n=wanted + len(self.rejected_cities))
        # MEMORY filter: remove every city the user has previously rejected this
        # session, then keep only the first `wanted` survivors.
        results = [
            (conf, dest, score)
            for conf, dest, score in out["results"]
            if dest["city"] not in self.rejected_cities
        ][:wanted]
        # Remember the (filtered) view we actually showed, so 'why' explains
        # these exact picks and so a follow-up "not <city>" matches the on-screen
        # list. We copy `out` but swap in the filtered results.
        self.last_results = dict(out)
        self.last_results["results"] = results
        # EMPTY / NO-MATCH state stays a PLAIN STRING (not a dict). Callers and
        # the GUI treat any string as ordinary chat text, so error/empty replies
        # keep rendering exactly as before. Only a successful list of picks is
        # returned as the structured dict below.
        if not results:
            lines = []
            for msg in out["advisories"]:
                lines.append("Note: " + msg)
            lines.append("I couldn't find any destinations that match all of those preferences "
                         "at once. If you're open to being a little more flexible on one or two "
                         "criteria, I can try again. Type 'restart' to start fresh.")
            return "\n".join(lines)
        # SUCCESS: build a STRUCTURED payload instead of one big string, so the
        # GUI can lay each pick out as its own card and the CLI can still render
        # it as text (via render_response_text()). Each card carries the already
        # computed numbers so neither front end has to recompute anything.
        cards = []
        rank = 1
        for conf, dest, score in results:
            # Rebuild the per-criterion evidence breakdown (same content as
            # inference.explain) but WITHOUT the leading "confidence .. |" and the
            # "strong on .." tail, because we expose those as separate fields.
            parts = []
            for name, (membership, cf) in score["details"].items():
                sign = "+" if cf >= 0 else ""
                parts.append(name + " " + str(round(membership, 2)) + " (cf " + sign + str(round(cf, 2)) + ")")
            explanation = ", ".join(parts) if parts else "no preferences given yet"
            # Lifestyle dimensions the CITY itself scores highly on (>=4), so the
            # card can show "Strong on: nature, cuisine" regardless of the query.
            strong_on = [dim for dim in kb.LIFESTYLE_DIMENSIONS if dest.get(dim, 0) >= 4]
            cards.append({
                "rank": rank,
                "city": dest["city"],
                "country": dest["country"],
                "region": dest["region"],
                "temp": str(dest["avg_temp_yearly"]) + "C",
                "budget": dest["budget_level"],
                "confidence": round(conf, 2),
                "explanation": explanation,        # technical trace (CLI)
                "description": dest["short_description"],
                "strong_on": strong_on,
                # Raw inputs so the GUI can build a HUMAN explanation on demand
                # via inference.explain_human(dest, score, query). These are the
                # same objects already held in self.last_results, so it is just a
                # reference, not a copy.
                "dest": dest,
                "score": score,
            })
            rank += 1
        return {
            "type": "recommendations",
            "advisories": list(out["advisories"]),
            "header": "Here are a few destinations that fit what you're looking for:",
            "results": cards,
            "footer": "Type 'why' for the reasoning, 'restart' to search again, or 'exit' to leave.",
            # Snapshot of the query so the GUI's human explanation reflects what
            # the user asked for AT THE TIME of this recommendation, even if they
            # keep refining preferences afterwards (history stays accurate).
            "query": {k: (dict(v) if isinstance(v, dict) else list(v))
                      for k, v in self.query.items()},
        }

    def _explain_last(self):
        # The "why" intent: a deeper reasoning breakdown of the last picks
        # (RESPONSE TYPE 12). If we have not recommended yet, say so.
        if not self.last_results or not self.last_results["results"]:
            return "I haven't made any recommendations yet. Tell me what you're looking for first."
        lines = ["Here's my reasoning (each criterion becomes a certainty factor: CF>0 supports the city, CF<0 counts against it; all are combined MYCIN-style into the confidence):"]
        for conf, dest, score in self.last_results["results"]:
            comps = ", ".join(name + " m=" + str(round(m, 2)) + "/cf=" + ("+" if cf >= 0 else "") + str(round(cf, 2)) for name, (m, cf) in score["details"].items()) or "no criteria given"
            lines.append(dest["city"] + ": " + comps + " -> confidence " + str(round(conf, 2)))
        return "\n".join(lines)

    def _lead(self, message, payload):
        # Attach a short conversational lead-in (e.g. "No problem.",
        # "Got it, I've removed Paris.") to a recommendation reply. Because
        # _format_recommendation() can now return EITHER a structured dict (on
        # success) OR a plain string (empty/no-match), we handle both: for a dict
        # we tuck the lead into a "lead" field the front ends render above the
        # cards; for a string we simply prepend it as before.
        if isinstance(payload, dict):
            payload = dict(payload)          # copy so we never mutate cached state
            payload["lead"] = message
            return payload
        return message + "\n" + payload

    def respond(self, text):
        # THE MAIN ENTRY POINT. Given one user message, return the bot's reply.
        # The reply is normally a STRING, but a successful recommendation is a
        # STRUCTURED dict (see _format_recommendation) so the GUI can render
        # cards; the CLI converts it back to text via render_response_text().
        if not text or not text.strip():
            return ("Tell me a bit about the trip you're looking for. The more "
                    "details you share, the better the recommendations will be.")
        intent = self._detect_intent(text)
        if intent == "exit":
            self.finished = True
            return "Thanks for stopping by. Have a great trip, and feel free to come back anytime you want more travel ideas."  # RESPONSE TYPE 15
        if intent == "help":
            return self.help_text()
        if intent == "restart":
            self.reset()
            return "Sure, let's start over.\n" + self._next_question()      # RESPONSE TYPE 14
        if intent == "why":
            return self._explain_last()
        if intent == "recommend":
            return self._format_recommendation()
        if intent == "greet":
            return "Hello! " + (self._next_question() or "Tell me about the kind of trip you have in mind and I'll suggest some destinations.")
        # SMALL TALK: pick a random reply from the matching set, then append the
        # next pending question so the conversation keeps moving. An empty string
        # reply means the phrase was ambiguous (e.g. "cool") so we fall through
        # to normal preference parsing instead of replying with nothing.
        if intent == "small_talk":
            replies = self._detect_small_talk(text.lower().strip())
            if replies:
                chosen = random.choice(replies)
                if chosen:
                    nxt = self._next_question()
                    suffix = ("\n" + nxt) if nxt else ""
                    return chosen + suffix
        if intent == "reject_city":
            # Remember the rejected cities for the rest of the session, then
            # immediately recompute the picks. _format_recommendation() filters
            # out everything in self.rejected_cities, so the rejected places are
            # gone and fresh ones take their slots.
            rejected = self._cities_to_reject(text)
            for city in rejected:
                self.rejected_cities.add(city)
            removed = _join_natural(rejected)
            return self._lead("Okay, I've removed " + removed + " from the list. Here are some updated options:",
                              self._format_recommendation())
        # intent == "provide" (or small-talk fall-through): the user is giving
        # preferences (the common case).
        # 'skip all' -> stop asking the rest and recommend with what we have.
        if self._is_skip_all(text):
            for slot, _q in self._slots():
                self.resolved.add(slot)
            self.pending_slot = None
            return self._lead("Sure, I'll work with what you've given me so far.", self._format_recommendation())  # RESPONSE TYPE 19 (skip-all)
        # If this message is a pure "skip" and we just asked a question, mark
        # that one slot resolved so we move on instead of nagging.
        if self._is_skip(text) and self.pending_slot is not None:
            self.resolved.add(self.pending_slot)
            nxt = self._next_question()
            if nxt is None:
                return self._lead("No problem.", self._format_recommendation())
            return "No problem, we can skip that. (say 'skip all' to skip the rest)\n" + nxt   # RESPONSE TYPE 18 (skip ack)
        # Otherwise, fill whatever slots the sentence contains.
        changed = self._update_slots(text)
        ack = self._acknowledge(changed)
        # If the slot we had asked about is now filled, mark it resolved.
        if self.pending_slot is not None and self._slot_filled(self.pending_slot):
            self.resolved.add(self.pending_slot)
        if ack is None:
            # Nothing recognised: pick a random clarification message instead of
            # always showing the same one (more natural, less robotic).
            # (Original: "Nothing recognised: gently clarify instead of failing silently.")
            return random.choice(_CONFUSED_REPLIES)                         # RESPONSE TYPE 9
        nxt = self._next_question()
        if nxt is None:
            # We have enough - recommend right away (ack becomes the lead-in).
            return self._lead(ack, self._format_recommendation())
        return ack + "\n" + nxt


def render_response_text(reply):
    # Convert ANY respond() return value into plain text. Most replies are
    # already strings and pass straight through; a structured recommendation
    # payload (dict) is flattened back into the same readable layout the CLI
    # always used. This lets main.py stay a simple `print(...)` loop while the
    # GUI consumes the rich dict directly.
    if not isinstance(reply, dict) or reply.get("type") != "recommendations":
        return reply
    lines = []
    # Optional conversational lead-in (skip-all / reject / ack), if present.
    if reply.get("lead"):
        lines.append(reply["lead"])
    # Forward-chaining advisories first, mirroring the old output.
    for msg in reply.get("advisories", []):
        lines.append("Note: " + msg)
    lines.append(reply["header"])
    for r in reply["results"]:
        lines.append("#" + str(r["rank"]) + " " + r["city"] + ", " + r["country"]
                     + " (" + r["region"] + ", " + r["temp"] + ", " + r["budget"] + ")")
        strong = (" | strong on " + ", ".join(r["strong_on"])) if r["strong_on"] else ""
        lines.append("   confidence " + str(r["confidence"]) + " | " + r["explanation"] + strong)
        lines.append("   " + r["description"])
    lines.append(reply["footer"])
    return "\n".join(lines)