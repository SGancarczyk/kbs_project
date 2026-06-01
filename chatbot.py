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

# Words that mean "I have no preference for the thing you just asked" - they let
# the user SKIP a slot instead of being forced to answer (un-menu-like).
SKIP_WORDS = {"any", "anywhere", "whatever", "skip", "none", "no", "nope", "idk", "anything"}
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
        "That's fine. Even a small detail helps — like whether you prefer beaches, cities, or mountains.",
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
     "own words — for example, what you enjoy doing, roughly where you want to go, "
     "or how long you have. Or type 'go' to see picks with what I have so far."),
    ("I'm not sure I followed that. The more you describe about the trip — even "
     "just the kind of atmosphere or activities you enjoy — the better I can match "
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
            "climate": [],
            "climate_exclude": [],
            "budget": [],
            "budget_exclude": [],
            "duration": [],
            "duration_exclude": [],
            "lifestyle": {},
            "lifestyle_exclude": [],
            # NEW: travel_months stores the month numbers (1-12) extracted from
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
            ("continent", "Do you already have a region in mind? You can mention somewhere specific like Europe or Southeast Asia, or just say you're open to anything."),
            ("season",    "When are you hoping to travel? Even a rough idea like a season or month is helpful."),
            ("duration",  "How long are you thinking of going for?"),
            ("budget",    "What kind of budget are you working with? For example, are you looking to keep costs down, or is it more of a treat yourself trip?"),
            ("climate",   "Do you have a preference for the weather? Warm and sunny, cooler, or somewhere in between?"),
            ("lifestyle", "What usually makes a trip memorable for you? Things like food, culture, beaches, hiking, nightlife — or anything you'd rather avoid."),
        ]

    def _slot_filled(self, slot):
        # Has this slot already received a value in the query, either as a wish
        # or as an explicit exclusion?
        if slot == "continent":
            return bool(self.query["regions_include"]) or bool(self.query["regions_exclude"])
        # The season slot is filled when at least one travel month was extracted.
        if slot == "season":
            return bool(self.query["travel_months"])
        if slot == "lifestyle":
            return bool(self.query["lifestyle"]) or bool(self.query["lifestyle_exclude"])
        if slot == "climate":
            return bool(self.query["climate"]) or bool(self.query["climate_exclude"])
        if slot == "budget":
            return bool(self.query["budget"]) or bool(self.query["budget_exclude"])
        if slot == "duration":
            return bool(self.query["duration"]) or bool(self.query["duration_exclude"])
        return bool(self.query[slot])

    def greeting(self):
        # The opening message (RESPONSE TYPE 1).
        return ("Hi! I'm your travel assistant. Tell me a bit about the kind of trip "
                "you're looking for and I'll help you find some great destinations. You "
                "can mention a place, budget, trip length, climate, or what you enjoy — "
                "all at once or bit by bit.\n"
                "You can say 'skip' to move past any question, 'skip all' to jump "
                "straight to recommendations, 'go' when you're ready, 'why' to see my "
                "reasoning, 'restart' to start over, or 'exit' to quit.\n"
                + self._next_question())

    def help_text(self):
        # The help message (RESPONSE TYPE 2).
        return ("I match you to travel destinations from a database of 560 cities. "
                "Just describe what you want in your own words — for example: "
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
        # Run the SAME informal-shorthand normalization the NLP layer uses BEFORE
        # matching, so casual variants are folded into their plain-English form
        # (e.g. "dont care" stays caught, and any shorthand the table expands no
        # longer slips past the skip check). This keeps skip detection in sync
        # with how preferences are parsed everywhere else.
        t = nlp._normalize_common_phrases(t)
        if t in SKIP_PHRASES:
            return True
        tokens = t.split()
        return len(tokens) > 0 and all(tok in SKIP_WORDS for tok in tokens)

    def _is_skip_all(self, text):
        # True if the user wants to skip ALL remaining questions and just get
        # picks now (e.g. "skip all", "skip the rest", "skip everything").
        t = text.lower().strip()
        if t in {"skip all", "skip everything", "skip the rest", "skip rest", "just recommend"}:
            return True
        return "skip" in t and ("all" in t or "rest" in t or "everything" in t)

    def _allow_spellcheck_for(self, slot):
        # The spell corrector is now conservative enough to run across slots: it
        # requires close normalized edit distance, avoids tiny target words, and
        # keeps first/last letters for normal corrections. This lets all-in-one
        # sentences with typos, such as "chaep asia cuisine", still fill budget.
        return True

    def _blocking_vocabulary_for(self, keyword_map):
        # All vocabulary keys except the current slot. Used by nlp.extract so a
        # negation aimed at another category does not leak into this category.
        # SEASON_SYNONYMS is included so e.g. "not summer" never bleeds into
        # a continent or lifestyle extraction.
        all_maps = [
            kb.CONTINENT_SYNONYMS,
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

    def _has_meaningful_preferences(self):
        # Do not recommend from an empty query, because that only returns dataset
        # order with zero evidence. At least one include, exclude, or interest is
        # needed for a meaningful recommendation.
        return any([
            self.query["regions_include"], self.query["regions_exclude"],
            self.query["climate"], self.query["climate_exclude"],
            self.query["budget"], self.query["budget_exclude"],
            self.query["duration"], self.query["duration_exclude"],
            self.query["lifestyle"], self.query["lifestyle_exclude"],
            # travel_months alone counts as a meaningful preference because the
            # inference engine will use it to adjust climate scoring.
            self.query["travel_months"],
        ])

    def _extract_season(self, text):
        # Run the NLP extractor over the SEASON_SYNONYMS vocabulary and convert
        # every matched key ("july", "summer", ...) into a list of month numbers
        # via knowledge_base.season_to_months(). No spellcheck: month/season
        # names are short and exact matches are safer here.
        result = nlp.extract(
            text, kb.SEASON_SYNONYMS,
            allow_spellcheck=False,
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
            "climate": [], "climate_exclude": [],
            "budget": [], "budget_exclude": [],
            "duration": [], "duration_exclude": [],
            "lifestyle": [], "lifestyle_exclude": [],
            # NEW: tracks which travel months were newly added this turn so
            # _acknowledge() can confirm them back to the user.
            "travel_months": [],
        }

        cont = nlp.extract(text, kb.CONTINENT_SYNONYMS, allow_spellcheck=self._allow_spellcheck_for("continent"), blocking_vocabulary=self._blocking_vocabulary_for(kb.CONTINENT_SYNONYMS))
        for v in cont["include"]:
            if v not in self.query["regions_include"]:
                self.query["regions_include"].append(v)
                changed["regions_include"].append(v)
        for v in cont["exclude"]:
            if v not in self.query["regions_exclude"]:
                self.query["regions_exclude"].append(v)
                changed["regions_exclude"].append(v)

        clim = nlp.extract(text, kb.CLIMATE_SYNONYMS, allow_spellcheck=self._allow_spellcheck_for("climate"), blocking_vocabulary=self._blocking_vocabulary_for(kb.CLIMATE_SYNONYMS))
        for v in clim["include"]:
            if v not in self.query["climate"]:
                self.query["climate"].append(v)
                changed["climate"].append(v)
        for v in clim["exclude"]:
            if v not in self.query["climate_exclude"]:
                self.query["climate_exclude"].append(v)
                changed["climate_exclude"].append(v)

        bud = nlp.extract(text, kb.BUDGET_SYNONYMS, allow_spellcheck=self._allow_spellcheck_for("budget"), blocking_vocabulary=self._blocking_vocabulary_for(kb.BUDGET_SYNONYMS))
        for v in bud["include"]:
            if v not in self.query["budget"]:
                self.query["budget"].append(v)
                changed["budget"].append(v)
        for v in bud["exclude"]:
            if v not in self.query["budget_exclude"]:
                self.query["budget_exclude"].append(v)
                changed["budget_exclude"].append(v)

        # DURATION is the one slot we never fill from a spell-corrected guess.
        # Duration keywords are short ("day", "week", "long", "short") and prone
        # to false positives (e.g. "sports" -> "short", "do" -> "day"), and the
        # bot should ASK about trip length rather than infer it from an unrelated
        # word. So we force allow_spellcheck=False: duration is only filled when
        # the user writes an actual duration word, exactly matched.
        dur = nlp.extract(text, kb.DURATION_SYNONYMS, allow_spellcheck=False, blocking_vocabulary=self._blocking_vocabulary_for(kb.DURATION_SYNONYMS))
        for v in dur["include"]:
            if v not in self.query["duration"]:
                self.query["duration"].append(v)
                changed["duration"].append(v)
        for v in dur["exclude"]:
            if v not in self.query["duration_exclude"]:
                self.query["duration_exclude"].append(v)
                changed["duration_exclude"].append(v)

        life = nlp.extract(text, kb.LIFESTYLE_SYNONYMS, allow_spellcheck=self._allow_spellcheck_for("lifestyle"), blocking_vocabulary=self._blocking_vocabulary_for(kb.LIFESTYLE_SYNONYMS))
        for v in life["include"]:
            if v not in self.query["lifestyle"]:
                self.query["lifestyle"][v] = 5
                changed["lifestyle"].append(v)
        for v in life["exclude"]:
            if v not in self.query["lifestyle_exclude"]:
                self.query["lifestyle_exclude"].append(v)
                changed["lifestyle_exclude"].append(v)

        # SEASON: no spellcheck (see _extract_season). Month numbers are stored
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
        if changed["regions_include"]:
            # Proper-case the regions so they read like place names.
            clauses.append(_join_natural([_pretty_region(r) for r in changed["regions_include"]]))
        if changed["budget"]:
            # "Budget" -> "on a budget", "Luxury" -> "luxury", etc.
            clauses.append(_join_natural([_BUDGET_PHRASES.get(b, b.lower()) for b in changed["budget"]]))
        if changed["climate"]:
            # "warm" -> "warm climate" so the slot is unambiguous.
            clauses.append(_join_natural([c + " climate" for c in changed["climate"]]))
        if changed["duration"]:
            clauses.append(_join_natural([_DURATION_PHRASES.get(d, d.lower()) for d in changed["duration"]]))
        if changed["lifestyle"]:
            # Interests are already plain words ("culture", "food"-ish).
            clauses.append("you enjoy " + _join_natural(changed["lifestyle"]))
        # NEW: confirm travel months back to the user in human-readable form.
        if changed["travel_months"]:
            clauses.append("travelling in " + self._season_label(changed["travel_months"]))
        # --- exclusions (what the user does NOT want) ---
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