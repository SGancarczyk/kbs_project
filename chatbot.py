import random
import knowledge_base as kb
import nlp
import inference as inf
def _lifestyle_pick_question():
    dims = kb.LIFESTYLE_DIMENSIONS
    rows = []
    for i in range(0, len(dims), 3):
        row = [f"{i+j+1}. {dims[i+j].capitalize():<12}"
               for j in range(3) if i + j < len(dims)]
        rows.append("  " + "  ".join(row))
    grid = "\n".join(rows)
    return (
        "What kind of experiences are you looking for on this trip?\n"
        "Here are some categories I can use to personalise recommendations:\n\n"
        + grid + "\n\n"
        "Just type the numbers that appeal to you (for example, \"1 4 6\").\n"
        "You can also type \"all\" if everything sounds interesting, "
        "or \"skip\" if you\'d rather move on."
    )
_ALL_LIFESTYLE_PHRASES = {
    "all", "everything", "all of them", "all of it", "all please",
    "each", "each one", "every one", "select all", "yes all", "all dimensions",
}
def _parse_lifestyle_selection(raw):
    dims = kb.LIFESTYLE_DIMENSIONS
    t = raw.lower().strip()
    if t in _ALL_LIFESTYLE_PHRASES:
        return list(enumerate(dims))
    parts = raw.replace(",", " ").split()
    selected, seen = [], set()
    for p in parts:
        if p.isdigit() and 1 <= int(p) <= len(dims):
            idx = int(p) - 1
            if idx not in seen:
                seen.add(idx)
                selected.append((idx, dims[idx]))
    return selected
def _parse_lifestyle_ratings(raw, selected_dims):
    import re as _re
    t = raw.lower().strip()
    ratings = {}
    events = []
    for dim in selected_dims:
        for m in _re.finditer(r"\b" + _re.escape(dim) + r"\b", t):
            events.append((m.start(), "dim", dim))
    for m in _re.finditer(r"\b([1-5])\b", t):
        events.append((m.start(), "digit", int(m.group(1))))
    events.sort(key=lambda e: e[0])
    current_dim = None
    orphan_digit = None
    for _, etype, evalue in events:
        if etype == "dim":
            current_dim = evalue
            if orphan_digit is not None and current_dim not in ratings:
                ratings[current_dim] = orphan_digit
                orphan_digit = None
        else:
            if current_dim is not None and current_dim not in ratings:
                ratings[current_dim] = evalue
                current_dim = None
            elif current_dim is not None and current_dim in ratings:
                orphan_digit = evalue
                current_dim = None
            else:
                orphan_digit = evalue
    if not ratings:
        numbers = _re.findall(r"\b([1-5])\b", t)
        for i, dim in enumerate(selected_dims):
            if i < len(numbers):
                ratings[dim] = int(numbers[i])
    return ratings
_AFFIRM_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please",
                 "go", "go ahead", "do it", "yes please", "sounds good", "why not",
                 "please do", "definitely", "absolutely", "that works", "perfect", "yes pls"}
_DECLINE_WORDS = {"no", "nope", "nah", "not really", "no thanks", "not quite", "not exactly"}
SKIP_WORDS = {"any", "anywhere", "whatever", "skip", "none", "no", "nope", "idk", "anything"}
SKIP_PHRASES = {"doesn't matter", "does not matter", "dont care", "don't care", "do not care", "no preference", "not sure",
                "doesnt matter", "doesnt", "dnt care", "dnt matter",
                "idgaf", "idc", "no pref", "np",
                "whatever works", "no idea", "surprise me", "up to you",
                "not bothered", "not fussed", "doesn't bother me",
                "not really", "not really sure"}
REJECTION_CUES = ("not ", "no ", "remove", "don't want", "dont want",
                  "do not want", "without", "exclude", "skip", "drop",
                  "rather not", "get rid", "anything but", "take out",
                  "i hate", "lose the")
_POSITION_WORDS = {
    "first": 0, "top": 0, "1st": 0, "one": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
}
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
        "",
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
    return region.replace("_", " ").title()
def _join_natural(items):
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
        self.destinations = destinations
        self._city_index = {d["city"].lower(): d for d in destinations}
        self.reset()
    def reset(self):
        self.query = {
            "regions_include": [],
            "regions_exclude": [],
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
            "travel_months": [],
            "importance": {},
        }
        self.resolved = set()
        self.pending_slot = None
        self.last_results = None
        self.finished = False
        self.rejected_cities = set()
        self._suggested_city = None
        self._skipped = {}
        self._ack_index = 0
    def _slots(self):
        return [
            ("continent", "Do you already have a region in mind? You can mention somewhere specific like Europe or Southeast Asia, or just say you're open to anything."),
            ("season", "When are you hoping to travel? Even a rough idea like a season or month is helpful."),
            ("duration", "How long are you thinking of going for?"),
            ("budget", "What kind of budget are you working with? For example, are you looking to keep costs down, or is it more of a treat yourself trip?"),
            ("climate", "Do you have a preference for the weather? Warm and sunny, cooler, or somewhere in between?"),
            ("lifestyle_pick", _lifestyle_pick_question()),
            ("lifestyle_rate", ""),
        ]
    def _slot_filled(self, slot):
        if slot == "continent":
            return (bool(self.query["regions_include"]) or bool(self.query["regions_exclude"])
                    or bool(self.query["countries_include"]) or bool(self.query["countries_exclude"]))
        if slot == "season":
            return bool(self.query["travel_months"])
        if slot == "lifestyle_pick":
            return (bool(self.query["lifestyle"]) or
                    self._skipped.get("lifestyle_pick", False))
        if slot == "lifestyle_rate":
            pending = self._lifestyle_pending_dims()
            return not pending or self._skipped.get("lifestyle_rate", False)
        if slot == "climate":
            return (bool(self.query["climate"]) or bool(self.query["climate_exclude"])
                    or self.query["importance"].get("climate") == 0.6)
        if slot == "budget":
            return (bool(self.query["budget"]) or bool(self.query["budget_exclude"])
                    or self.query["importance"].get("budget") == 0.6)
        if slot == "duration":
            return (bool(self.query["duration"]) or bool(self.query["duration_exclude"])
                    or self.query["importance"].get("duration") == 0.6)
        return bool(self.query[slot])
    def greeting(self):
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
        return ("I match you to travel destinations from a database of 560 cities. "
                "Just describe what you want in your own words — for example: "
                "'a cheap warm week in Asia, I love food and culture'. "
                "I understand full sentences, fix small typos, and handle negation "
                "like 'not Africa' or 'avoid nightlife'.\n"
                "You can also ask about a place — 'tell me about Kyoto' — or say "
                "'more like Lisbon' to find similar destinations.\n"
                "Commands: 'go' or 'skip all' (get picks now), 'why' (see reasoning), "
                "'skip' (skip one question), 'restart', 'exit'.")
    def _detect_intent(self, text):
        t = text.lower().strip()
        if t in {"exit", "quit", "bye", "goodbye", "stop"} or t.startswith("exit"):
            return "exit"
        if t in {"help", "?"} or "how do you work" in t or "what can you do" in t:
            return "help"
        if "restart" in t or "start over" in t or "reset" in t or "new search" in t or "search again" in t or "start again" in t:
            return "restart"
        if t.startswith("why") or "explain" in t or "reason" in t:
            return "why"
        _is_short = len(t.split()) <= 8
        if _is_short and (t in {"go", "done", "finish", "enough", "recommend",
                                "show me", "results", "go ahead", "show results",
                                "show me results", "give me results"} or
                          "that's all" in t or "thats all" in t):
            return "recommend"
        if t in {"hi", "hello", "hey", "selam", "yo"}:
            return "greet"
        if self._detect_small_talk(t) is not None:
            return "small_talk"
        if self._find_city_in_text(text) is not None:
            if any(cue in t for cue in self._MORE_LIKE_CUES):
                return "more_like"
            if any(cue in t for cue in self._DESCRIBE_CUES):
                return "describe_city"
        if self.last_results and self.last_results.get("results"):
            if self._has_rejection_cue(t) and self._cities_to_reject(text):
                return "reject_city"
        return "provide"
    def _detect_small_talk(self, lowered):
        for phrase_set, replies in _SMALL_TALK.items():
            if lowered in phrase_set:
                return replies
        for phrase_set, replies in _SMALL_TALK.items():
            for phrase in phrase_set:
                if len(phrase) > 4 and phrase in lowered:
                    return replies
        return None
    def _has_rejection_cue(self, t):
        return any(cue in t for cue in REJECTION_CUES)
    def _cities_to_reject(self, text):
        if not self.last_results or not self.last_results.get("results"):
            return []
        t = text.lower()
        shown = [dest["city"] for _, dest, _ in self.last_results["results"]]
        to_reject = []
        if "those" in t or "these" in t or "all of them" in t or "all of these" in t:
            return list(shown)
        for word, idx in _POSITION_WORDS.items():
            if word in t and idx < len(shown) and shown[idx] not in to_reject:
                to_reject.append(shown[idx])
        if "last" in t and shown and shown[-1] not in to_reject:
            to_reject.append(shown[-1])
        for city in shown:
            if city.lower() in t and city not in to_reject:
                to_reject.append(city)
        return to_reject
    def _find_city_in_text(self, text):
        import re as _re
        t = text.lower()
        for name in sorted(self._city_index, key=len, reverse=True):
            if _re.search(r"\b" + _re.escape(name) + r"\b", t):
                return self._city_index[name]
        return None
    _DESCRIBE_CUES = ("tell me about", "what about", "what can you tell", "describe",
                      "info on", "information on", "anything on", "how is", "how's",
                      "hows ", "what is", "whats ", "what's ", "tell me more about")
    _MORE_LIKE_CUES = ("more like", "similar to", "something like", "sort of like",
                       "places like", "somewhere like", "anything like")
    def _is_skip(self, text):
        t = text.lower().strip()
        t = nlp._normalize_common_phrases(t)
        if t in SKIP_PHRASES:
            return True
        tokens = t.split()
        return len(tokens) > 0 and all(tok in SKIP_WORDS for tok in tokens)
    def _is_skip_all(self, text):
        t = text.lower().strip()
        if t in {"skip all", "skip everything", "skip the rest", "skip rest", "just recommend"}:
            return True
        return "skip" in t and ("all" in t or "rest" in t or "everything" in t)
    def _allow_spellcheck_for(self, slot):
        return True
    def _blocking_vocabulary_for(self, keyword_map):
        all_maps = [
            kb.CONTINENT_SYNONYMS,
            kb.COUNTRY_SYNONYMS,
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
    def _slot_index(self, slot_name):
        for i, (slot, _) in enumerate(self._slots()):
            if slot == slot_name:
                return i
        return 0
    def _lifestyle_pending_dims(self):
        return [d for d, w in self.query["lifestyle"].items() if w == -1]
    def _lifestyle_rate_question(self):
        pending = self._lifestyle_pending_dims()
        if not pending:
            return None
        dims_str = _join_natural(pending)
        if len(pending) == 1:
            dim = pending[0]
            return (f"How important is {dim} to you on this trip?\n"
                    f"Rate it 1 to 5, where 1 = nice to have and 5 = absolutely essential.")
        return (
            f"Nice mix. I see {dims_str} on your list.\n"
            f"How important is each one to you?\n\n"
            f"Rate them from 1 to 5, where:\n"
            f"  1 = nice to have\n"
            f"  5 = absolutely essential\n\n"
            f"You can reply with numbers in order (\"" +
            " ".join(["5"] * len(pending)) + f"\") "
            f"or by name (e.g. \"{pending[0]} 5, {pending[-1]} 3\")."
        )
    def _has_meaningful_preferences(self):
        return any([
            self.query["regions_include"], self.query["regions_exclude"],
            self.query["climate"], self.query["climate_exclude"],
            self.query["budget"], self.query["budget_exclude"],
            self.query["duration"], self.query["duration_exclude"],
            self.query["lifestyle"], self.query["lifestyle_exclude"],
            self.query["travel_months"],
        ])
    def _extract_season(self, text):
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
    _EMPHASIS_CUES = ("most important", "matters most", "matter most", "care most",
                      "care the most", "top priority", "biggest priority",
                      "main priority", "number one", "is a must", "are a must",
                      "is a priority", "are a priority", "priority for me",
                      "really matters", "what matters most", "what i care about most",
                      "is very important", "are very important", "is really important",
                      "are really important", "is important", "are important")
    _DEEMPHASIS_CUES = ("don't care", "dont care", "doesnt care", "do not care",
                        "don't really care", "dont really care", "not fussed",
                        "not bothered", "care less", "least important",
                        "not that important", "not really important", "not important",
                        "doesn't matter", "doesnt matter", "does not matter",
                        "doesnt really matter", "dont matter", "not matter",
                        "not a priority", "less important", "not crucial")
    _CRIT_ANCHORS = {"budget": "budget", "budgets": "budget", "price": "budget",
                     "prices": "budget", "cost": "budget", "costs": "budget",
                     "money": "budget", "spend": "budget", "spending": "budget",
                     "expense": "budget", "expenses": "budget", "climate": "climate",
                     "weather": "climate", "temperature": "climate",
                     "duration": "duration", "length": "duration",
                     "interest": "lifestyle", "interests": "lifestyle",
                     "activities": "lifestyle", "activity": "lifestyle",
                     "lifestyle": "lifestyle"}
    _SLOT_TO_CRIT = {"budget": "budget", "climate": "climate", "duration": "duration",
                     "lifestyle_pick": "lifestyle", "lifestyle_rate": "lifestyle"}
    _CRIT_LABEL = {"climate": "the climate", "budget": "budget",
                   "duration": "the trip length", "lifestyle": "your interests"}
    def _semantic_layer(self, text):
        import re as _re
        t = text.lower()
        life_vocab = kb.LIFESTYLE_SYNONYMS
        crit_vocabs = (("climate", kb.CLIMATE_SYNONYMS), ("budget", kb.BUDGET_SYNONYMS),
                       ("duration", kb.DURATION_SYNONYMS))
        positions = []
        for tok in _re.finditer(r"[a-z']+", t):
            w = tok.group()
            if w in self._CRIT_ANCHORS:
                positions.append((tok.start(), tok.end(), self._CRIT_ANCHORS[w], None))
                continue
            hit = None
            for crit, vocab in crit_vocabs:
                if w in vocab:
                    hit = (crit, None)
                    break
            if hit is None and w in life_vocab:
                hit = ("lifestyle_dim", life_vocab[w])
            if hit is not None:
                positions.append((tok.start(), tok.end(), hit[0], hit[1]))
        importance, high_dims, low_dims, strip = {}, set(), set(), []
        pending_crit = self._SLOT_TO_CRIT.get(self.pending_slot)
        def attach(cue_start, cue_end):
            after_snip = t[cue_end:cue_end + 50].lstrip()
            if any(after_snip.startswith(p) for p in ("about ", "for ", "on ", "regarding ", "with ")):
                aft = [p for p in positions if p[0] >= cue_end]
                if aft:
                    return min(aft, key=lambda p: p[0] - cue_end)
            before = [p for p in positions if p[0] < cue_start]
            if before:
                cl = max(before, key=lambda p: p[0])
                if cue_start - cl[0] <= 30:
                    return cl
            aft = [p for p in positions if p[0] >= cue_end]
            if aft:
                na = min(aft, key=lambda p: p[0] - cue_end)
                if na[0] - cue_end <= 50:
                    return na
            return min(positions, key=lambda p: abs(p[0] - cue_start))
        for cue in self._EMPHASIS_CUES:
            for m in _re.finditer(_re.escape(cue), t):
                if positions:
                    p = attach(m.start(), m.end())
                    if p[2] == "lifestyle_dim":
                        high_dims.add(p[3])
                    else:
                        importance.setdefault(p[2], 1.3)
                elif pending_crit:
                    importance.setdefault(pending_crit, 1.3)
        for cue in self._DEEMPHASIS_CUES:
            for m in _re.finditer(_re.escape(cue), t):
                if positions:
                    p = attach(m.start(), m.end())
                    strip.append((min(m.start(), p[0]), max(m.end(), p[1])))
                    if p[2] == "lifestyle_dim":
                        low_dims.add(p[3])
                    else:
                        importance[p[2]] = 0.6
                elif pending_crit:
                    strip.append((m.start(), m.end()))
                    importance[pending_crit] = 0.6
        high_dims -= low_dims
        if strip:
            chars = list(text)
            for s, e in strip:
                for k in range(s, min(e, len(chars))):
                    chars[k] = " "
            cleaned = "".join(chars)
        else:
            cleaned = text
        return {"importance": importance, "high_dims": high_dims,
                "low_dims": low_dims, "cleaned": cleaned}
    def _is_dont_care(self, text):
        t = text.lower()
        return any(cue in t for cue in self._DEEMPHASIS_CUES)
    def _importance_note(self, imp_changed):
        if not imp_changed:
            return ""
        ups = [self._CRIT_LABEL[c] for c, f in imp_changed.items() if f > 1]
        downs = [self._CRIT_LABEL[c] for c, f in imp_changed.items() if f < 1]
        notes = []
        if ups:
            notes.append("I'll weight " + _join_natural(ups) + " more heavily")
        if downs:
            notes.append("I'll go easier on " + _join_natural(downs))
        return ("; ".join(notes) + ".") if notes else ""
    def _update_slots(self, text):
        changed = {
            "regions_include": [], "regions_exclude": [],
            "countries_include": [], "countries_exclude": [],
            "climate": [], "climate_exclude": [],
            "budget": [], "budget_exclude": [],
            "duration": [], "duration_exclude": [],
            "lifestyle": [], "lifestyle_exclude": [],
            "travel_months": [],
            "importance": {},
        }
        sem = self._semantic_layer(text)
        text = sem["cleaned"]
        cont = nlp.extract(text, kb.CONTINENT_SYNONYMS, allow_spellcheck=self._allow_spellcheck_for("continent"), blocking_vocabulary=self._blocking_vocabulary_for(kb.CONTINENT_SYNONYMS))
        for v in cont["include"]:
            if v not in self.query["regions_include"]:
                self.query["regions_include"].append(v)
                changed["regions_include"].append(v)
        for v in cont["exclude"]:
            if v not in self.query["regions_exclude"]:
                self.query["regions_exclude"].append(v)
                changed["regions_exclude"].append(v)
        ctry = nlp.extract(text, kb.COUNTRY_SYNONYMS, allow_spellcheck=False,
                           blocking_vocabulary=self._blocking_vocabulary_for(kb.COUNTRY_SYNONYMS))
        for v in ctry["include"]:
            if v not in self.query["countries_include"]:
                self.query["countries_include"].append(v)
                changed["countries_include"].append(v)
        for v in ctry["exclude"]:
            if v not in self.query["countries_exclude"]:
                self.query["countries_exclude"].append(v)
                changed["countries_exclude"].append(v)
        if changed["countries_include"] and changed["regions_include"]:
            if " or " not in text.lower():
                country_containing_regions = {
                    kb.country_to_region(c) for c in changed["countries_include"]
                    if kb.country_to_region(c)
                }
                for r in list(changed["regions_include"]):
                    if r in country_containing_regions:
                        changed["regions_include"].remove(r)
                        if r in self.query["regions_include"]:
                            self.query["regions_include"].remove(r)
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
        dur = nlp.extract(text, kb.DURATION_SYNONYMS, allow_spellcheck=False, blocking_vocabulary=self._blocking_vocabulary_for(kb.DURATION_SYNONYMS))
        for v in dur["include"]:
            if v not in self.query["duration"]:
                self.query["duration"].append(v)
                changed["duration"].append(v)
        for v in dur["exclude"]:
            if v not in self.query["duration_exclude"]:
                self.query["duration_exclude"].append(v)
                changed["duration_exclude"].append(v)
        life_any = nlp.extract(text, kb.LIFESTYLE_SYNONYMS, allow_spellcheck=False,
                               blocking_vocabulary=self._blocking_vocabulary_for(kb.LIFESTYLE_SYNONYMS))
        for v in life_any["exclude"]:
            if v not in self.query["lifestyle_exclude"]:
                self.query["lifestyle_exclude"].append(v)
                changed["lifestyle_exclude"].append(v)
        if self.pending_slot != "lifestyle_rate":
            for v in life_any["include"]:
                if v not in self.query["lifestyle"] or self.query["lifestyle"].get(v) == -1:
                    self.query["lifestyle"][v] = 4
                    if v not in changed["lifestyle"]:
                        changed["lifestyle"].append(v)
        if self.pending_slot == "lifestyle_pick":
            selected = _parse_lifestyle_selection(text)
            if not selected and text.strip() == "":
                selected = list(enumerate(kb.LIFESTYLE_DIMENSIONS))
            for _, dim in selected:
                if dim not in self.query["lifestyle"]:
                    self.query["lifestyle"][dim] = -1
                    changed["lifestyle"].append(dim)
        elif self.pending_slot == "lifestyle_rate":
            pending = self._lifestyle_pending_dims()
            ratings = _parse_lifestyle_ratings(text, pending)
            for dim, weight in ratings.items():
                self.query["lifestyle"][dim] = weight
                if dim not in changed["lifestyle"]:
                    changed["lifestyle"].append(dim)
            for dim in pending:
                if dim not in ratings:
                    self.query["lifestyle"][dim] = 3
                    if dim not in changed["lifestyle"]:
                        changed["lifestyle"].append(dim)
        months = self._extract_season(text)
        for m in months:
            if m not in self.query["travel_months"]:
                self.query["travel_months"].append(m)
                changed["travel_months"].append(m)
        if self.pending_slot != "lifestyle_rate":
            for dim in sem["high_dims"]:
                if self.query["lifestyle"].get(dim) != 5:
                    self.query["lifestyle"][dim] = 5
                    if dim not in changed["lifestyle"]:
                        changed["lifestyle"].append(dim)
            for dim in sem["low_dims"]:
                cur = self.query["lifestyle"].get(dim)
                if cur in (None, -1) or cur > 1:
                    self.query["lifestyle"][dim] = 1
                    if dim not in changed["lifestyle"]:
                        changed["lifestyle"].append(dim)
        for crit, factor in sem["importance"].items():
            if self.query["importance"].get(crit) != factor:
                self.query["importance"][crit] = factor
                changed["importance"][crit] = factor
        return changed
    def _next_opener(self):
        opener = _ACK_OPENERS[self._ack_index % len(_ACK_OPENERS)]
        self._ack_index += 1
        return opener
    def _season_label(self, months):
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December",
        }
        season_map = {
            frozenset([12, 1, 2]): "winter",
            frozenset([3, 4, 5]): "spring",
            frozenset([6, 7, 8]): "summer",
            frozenset([9, 10, 11]): "autumn",
        }
        key = frozenset(months)
        if key in season_map:
            return season_map[key]
        if len(months) == 1:
            return month_names.get(months[0], str(months[0]))
        return _join_natural([month_names.get(m, str(m)) for m in sorted(months)])
    def _acknowledge(self, changed):
        clauses = []
        if changed["countries_include"]:
            clauses.append(_join_natural(changed["countries_include"]))
        if changed["regions_include"]:
            clauses.append(_join_natural([_pretty_region(r) for r in changed["regions_include"]]))
        if changed["budget"]:
            clauses.append(_join_natural([_BUDGET_PHRASES.get(b, b.lower()) for b in changed["budget"]]))
        if changed["climate"]:
            clauses.append(_join_natural([c + " climate" for c in changed["climate"]]))
        if changed["duration"]:
            clauses.append(_join_natural([_DURATION_PHRASES.get(d, d.lower()) for d in changed["duration"]]))
        if changed["lifestyle"]:
            confirmed = [d for d in changed["lifestyle"]
                         if self.query["lifestyle"].get(d, -1) > 0]
            pending = [d for d in changed["lifestyle"]
                         if self.query["lifestyle"].get(d) == -1]
            if confirmed:
                _weight_phrases = {
                    5: "a must-have",
                    4: "very important",
                    3: "a nice bonus",
                    2: "not a priority",
                    1: "not really important",
                }
                weight_parts = [
                    f"{d} {'are' if d in ('beaches','adventures') else 'is'} "
                    f"{_weight_phrases.get(self.query['lifestyle'].get(d, 3), 'noted')}"
                    for d in confirmed
                ]
                clauses.append(_join_natural(weight_parts))
            if pending:
                clauses.append("picked: " + _join_natural(pending))
        if changed["travel_months"]:
            clauses.append("travelling in " + self._season_label(changed["travel_months"]))
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
        imp_note = self._importance_note(changed.get("importance") or {})
        if not clauses and not imp_note:
            return None
        opener = self._next_opener()
        ack_text = (opener + " " + _join_natural(clauses) + ".") if clauses else opener
        if imp_note:
            ack_text += " " + imp_note
        if any(self.query["lifestyle"].get(d, -1) > 0
               for d in changed.get("lifestyle", [])):
            top = [d for d in self.query["lifestyle"]
                    if self.query["lifestyle"].get(d, 0) >= 4]
            mid = [d for d in self.query["lifestyle"]
                    if self.query["lifestyle"].get(d, 0) == 3]
            if top:
                top_str = _join_natural(top)
                mid_str = (" while also considering " + _join_natural(mid)) if mid else ""
                ack_text += (f"\nI'll prioritise destinations that are strong on "
                             f"{top_str}{mid_str}.")
        return ack_text
    def _next_question(self):
        for slot, question in self._slots():
            if slot in self.resolved or self._slot_filled(slot):
                continue
            self.pending_slot = slot
            if slot == "lifestyle_rate":
                return self._lifestyle_rate_question()
            return question
        self.pending_slot = None
        return None
    def _sanitize_lifestyle_weights(self):
        for dim, w in list(self.query["lifestyle"].items()):
            if w == -1:
                self.query["lifestyle"][dim] = 3
    def _format_recommendation(self):
        if not self._has_meaningful_preferences():
            self.last_results = None
            return ("I need at least one preference before I can make a meaningful recommendation. "
                    "Tell me a region, budget, climate, trip length, interest, or something to avoid.")
        self._sanitize_lifestyle_weights()
        wanted = 5
        out = inf.recommend(self.destinations, self.query, top_n=wanted + len(self.rejected_cities))
        results = [
            (conf, dest, score)
            for conf, dest, score in out["results"]
            if dest["city"] not in self.rejected_cities
        ][:wanted]
        self.last_results = dict(out)
        self.last_results["results"] = results
        self.last_results["query"] = {k: (dict(v) if isinstance(v, dict) else list(v))
                                      for k, v in self.query.items()}
        if not results:
            lines = []
            for msg in out["advisories"]:
                lines.append("Note: " + msg)
            lines.append("I couldn't find any destinations that match all of those preferences "
                         "at once. If you're open to being a little more flexible on one or two "
                         "criteria, I can try again. Type 'restart' to start fresh.")
            return "\n".join(lines)
        query_snapshot = {k: (dict(v) if isinstance(v, dict) else list(v))
                          for k, v in self.query.items()}
        cards = []
        rank = 1
        for conf, dest, score in results:
            reasons = inf.explain_human(dest, score, query_snapshot)
            strong_on = [dim for dim in kb.LIFESTYLE_DIMENSIONS if dest.get(dim, 0) >= 4]
            cards.append({
                "rank": rank,
                "city": dest["city"],
                "country": dest["country"],
                "region": dest["region"],
                "temp": str(dest["avg_temp_yearly"]) + "C",
                "budget": dest["budget_level"],
                "match": inf.match_label(conf),
                "reasons": reasons,
                "description": dest["short_description"],
                "strong_on": strong_on,
                "dest": dest,
                "score": score,
            })
            rank += 1
        scope_parts = []
        if self.query["regions_include"]:
            scope_parts.extend([_pretty_region(r) for r in self.query["regions_include"]])
        if self.query["countries_include"]:
            scope_parts.extend(self.query["countries_include"])
        scope = (" in " + _join_natural(scope_parts)) if scope_parts else ""
        if out.get("broadened"):
            summary = ("Nothing matched that exact region, so I widened the search to "
                       + str(out["pool_size"]) + " destinations. Closest matches:")
        else:
            summary = ("I compared " + str(out["pool_size"]) + " destination"
                       + ("s" if out["pool_size"] != 1 else "") + scope
                       + " and ranked the best fits:")
        return {
            "type": "recommendations",
            "advisories": list(out["advisories"]),
            "summary": summary,
            "header": "Here are a few destinations that fit what you're looking for:",
            "results": cards,
            "footer": "Type 'why' for the reasoning, 'restart' to search again, or 'exit' to leave.",
            "query": {k: (dict(v) if isinstance(v, dict) else list(v))
                      for k, v in self.query.items()},
        }
    def _explain_last(self):
        if not self.last_results or not self.last_results["results"]:
            return "I haven't made any recommendations yet. Tell me what you're looking for first."
        stored_query = self.last_results.get("query") or self.query
        lines = ["Here's why I picked these:"]
        for conf, dest, score in self.last_results["results"]:
            lines.append("")
            lines.append(dest["city"] + ", " + dest["country"]
                         + " — " + inf.match_label(conf))
            for reason in inf.explain_human(dest, score, stored_query):
                lines.append("  - " + reason)
        return "\n".join(lines)
    def _describe_city(self, dest):
        lines = []
        lines.append(dest["city"] + ", " + dest["country"]
                     + " (" + _pretty_region(dest["region"]) + ")")
        lines.append(dest["short_description"])
        lines.append("Climate: " + dest["climate"] + ", about "
                     + str(dest["avg_temp_yearly"]) + "°C on average across the year.")
        comfy = kb.comfortable_months(dest["avg_temp_monthly"])
        if comfy:
            lines.append("Best time for warm, pleasant weather: "
                         + kb.format_month_ranges(comfy) + ".")
        else:
            wm = kb.warmest_month(dest["avg_temp_monthly"])
            if wm:
                lines.append("It stays cool most of the year; the mildest spell is "
                             "around " + kb._MONTH_ABBR[wm[0]]
                             + " (~" + str(round(wm[1])) + "°C).")
        wm = kb.warmest_month(dest["avg_temp_monthly"])
        if wm:
            hot = dest["avg_temp_monthly"][str(wm[0])]
            lines.append("In " + kb._MONTH_ABBR[wm[0]] + " expect roughly "
                         + str(round(hot["max"])) + "°C days and "
                         + str(round(hot["min"])) + "°C nights.")
        if kb.hemisphere(dest.get("latitude")) == "southern":
            lines.append("It's in the southern hemisphere, so its summer falls "
                         "around December–February.")
        strong = [dim for dim in kb.LIFESTYLE_DIMENSIONS if dest.get(dim, 0) >= 4]
        if strong:
            lines.append("Great for: " + _join_natural(strong) + ".")
        lines.append("Budget level: " + dest["budget_level"]
                     + ". Ideal for " + _join_natural([_DURATION_PHRASES.get(d, d.lower())
                                                       for d in dest["ideal_durations"]]) + ".")
        self._suggested_city = dest
        lines.append("Is this the kind of place you're after? I can find similar "
                     "destinations — just say \"yes\", or tell me what to change "
                     "(for example \"but warmer\", \"cheaper\", or \"more beaches\").")
        return "\n".join(lines)
    def _message_mentions_preference(self, text):
        for vocab in (kb.CLIMATE_SYNONYMS, kb.BUDGET_SYNONYMS, kb.DURATION_SYNONYMS,
                      kb.LIFESTYLE_SYNONYMS, kb.CONTINENT_SYNONYMS, kb.COUNTRY_SYNONYMS):
            r = nlp.extract(text, vocab, allow_spellcheck=False)
            if r["include"] or r["exclude"]:
                return True
        return False
    def _apply_more_like_tweaks(self, text):
        changes = []
        clim = nlp.extract(text, kb.CLIMATE_SYNONYMS, allow_spellcheck=True,
                           blocking_vocabulary=self._blocking_vocabulary_for(kb.CLIMATE_SYNONYMS))
        if clim["include"]:
            self.query["climate"] = list(dict.fromkeys(clim["include"]))
            changes.append(_join_natural(clim["include"]) + " climate")
        for v in clim["exclude"]:
            if v not in self.query["climate_exclude"]:
                self.query["climate_exclude"].append(v)
            changes.append("not " + v)
        bud = nlp.extract(text, kb.BUDGET_SYNONYMS, allow_spellcheck=True,
                          blocking_vocabulary=self._blocking_vocabulary_for(kb.BUDGET_SYNONYMS))
        if bud["include"]:
            self.query["budget"] = list(dict.fromkeys(bud["include"]))
            changes.append(_join_natural([_BUDGET_PHRASES.get(b, b.lower()) for b in bud["include"]]))
        for v in bud["exclude"]:
            if v not in self.query["budget_exclude"]:
                self.query["budget_exclude"].append(v)
            changes.append("nothing " + _BUDGET_PHRASES.get(v, v.lower()))
        dur = nlp.extract(text, kb.DURATION_SYNONYMS, allow_spellcheck=False,
                          blocking_vocabulary=self._blocking_vocabulary_for(kb.DURATION_SYNONYMS))
        if dur["include"]:
            self.query["duration"] = list(dict.fromkeys(dur["include"]))
            changes.append(_join_natural([_DURATION_PHRASES.get(d, d.lower()) for d in dur["include"]]))
        months = self._extract_season(text)
        if months:
            for m in months:
                if m not in self.query["travel_months"]:
                    self.query["travel_months"].append(m)
            changes.append("travelling in " + self._season_label(months))
        life = nlp.extract(text, kb.LIFESTYLE_SYNONYMS, allow_spellcheck=False,
                           blocking_vocabulary=self._blocking_vocabulary_for(kb.LIFESTYLE_SYNONYMS))
        for v in life["include"]:
            self.query["lifestyle"][v] = 5
            changes.append("more " + v)
        for v in life["exclude"]:
            if v not in self.query["lifestyle_exclude"]:
                self.query["lifestyle_exclude"].append(v)
            self.query["lifestyle"].pop(v, None)
            changes.append("less " + v)
        cont = nlp.extract(text, kb.CONTINENT_SYNONYMS, allow_spellcheck=True,
                           blocking_vocabulary=self._blocking_vocabulary_for(kb.CONTINENT_SYNONYMS))
        for v in cont["include"]:
            if v not in self.query["regions_include"]:
                self.query["regions_include"].append(v)
                changes.append("in " + _pretty_region(v))
        ctry = nlp.extract(text, kb.COUNTRY_SYNONYMS, allow_spellcheck=False)
        for v in ctry["include"]:
            if v not in self.query["countries_include"]:
                self.query["countries_include"].append(v)
                changes.append("in " + v)
        return changes
    def _more_like(self, dest, tweak_text=""):
        for key in ("regions_include", "regions_exclude", "countries_include",
                    "countries_exclude", "climate_exclude", "budget_exclude",
                    "lifestyle_exclude"):
            self.query[key] = []
        self.query["climate"] = [dest["climate"]]
        self.query["budget"] = [dest["budget_level"]]
        self.query["lifestyle"] = {dim: dest[dim] for dim in kb.LIFESTYLE_DIMENSIONS
                                   if dest.get(dim, 0) >= 4}
        import re as _re
        clean = _re.sub(r"\b" + _re.escape(dest["city"].lower()) + r"\b", " ",
                        (tweak_text or "").lower())
        changes = self._apply_more_like_tweaks(clean) if clean.strip() else []
        self.rejected_cities.add(dest["city"])
        strong = [d for d in kb.LIFESTYLE_DIMENSIONS if dest.get(d, 0) >= 4]
        template = (dest["climate"] + ", " + dest["budget_level"].lower()
                    + ", strong on " + _join_natural(strong))
        if changes:
            lead = ("Sure — starting from " + dest["city"] + " (" + template
                    + ") with your changes (" + _join_natural(changes)
                    + "), here are some options:")
        else:
            lead = ("Sure — using " + dest["city"] + " as a template (" + template
                    + "), here are places with a similar vibe:")
        return self._lead(lead, self._format_recommendation())
    def _lead(self, message, payload):
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["lead"] = message
            return payload
        return message + "\n" + payload
    def respond(self, text):
        if not text or not text.strip():
            return ("Tell me a bit about the trip you're looking for. The more "
                    "details you share, the better the recommendations will be.")
        if self._suggested_city is not None:
            seed = self._suggested_city
            t = text.lower().strip()
            if t in _DECLINE_WORDS or t.startswith("no "):
                self._suggested_city = None
                return ("No problem. Tell me what you'd prefer and I'll suggest "
                        "something else.")
            if (self._detect_intent(text) in {"exit", "restart", "help", "why",
                                              "describe_city", "more_like"}
                    or self._find_city_in_text(text) is not None):
                self._suggested_city = None
            elif (t in _AFFIRM_WORDS or any(t.startswith(a) for a in _AFFIRM_WORDS)
                  or self._message_mentions_preference(text)):
                self._suggested_city = None
                has_pref = self._message_mentions_preference(text)
                return self._more_like(seed, text if has_pref else "")
            else:
                self._suggested_city = None
        intent = self._detect_intent(text)
        if intent == "exit":
            self.finished = True
            return "Thanks for stopping by. Have a great trip, and feel free to come back anytime you want more travel ideas."
        if intent == "help":
            return self.help_text()
        if intent == "restart":
            self.reset()
            return "Sure, let's start over.\n" + self._next_question()
        if intent == "why":
            return self._explain_last()
        if intent == "describe_city":
            city = self._find_city_in_text(text)
            if city is not None:
                return self._describe_city(city)
        if intent == "more_like":
            city = self._find_city_in_text(text)
            if city is not None:
                return self._more_like(city, text)
        if intent == "recommend":
            if self._has_meaningful_preferences():
                lifestyle_untouched = (
                    not self.query["lifestyle"] and
                    not self.query["lifestyle_exclude"] and
                    "lifestyle_pick" not in self.resolved and
                    not self._skipped.get("lifestyle_pick", False)
                )
                if lifestyle_untouched:
                    self.pending_slot = "lifestyle_pick"
                    return ("Before I show results, one quick question — it helps me rank destinations.\n"
                            + self._slots()[self._slot_index("lifestyle_pick")][1])
            return self._format_recommendation()
        if intent == "greet":
            changed = self._update_slots(text)
            if self.pending_slot is not None and self._slot_filled(self.pending_slot):
                self.resolved.add(self.pending_slot)
            ack = self._acknowledge(changed)
            nxt = self._next_question() or "Tell me about the kind of trip you have in mind and I'll suggest some destinations."
            if ack:
                return "Hello! " + ack + "\n" + nxt
            return "Hello! " + nxt
        if intent == "small_talk":
            replies = self._detect_small_talk(text.lower().strip())
            if replies:
                chosen = random.choice(replies)
                if chosen:
                    changed = self._update_slots(text)
                    if self.pending_slot is not None and self._slot_filled(self.pending_slot):
                        self.resolved.add(self.pending_slot)
                    elif self.pending_slot is not None:
                        crit = self._SLOT_TO_CRIT.get(self.pending_slot)
                        if crit and self.query["importance"].get(crit) == 0.6:
                            self._skipped[self.pending_slot] = True
                            self.resolved.add(self.pending_slot)
                    ack = self._acknowledge(changed)
                    nxt = self._next_question()
                    if (nxt is None and self._has_meaningful_preferences()
                            and (self.last_results is None or ack)):
                        return self._lead(ack if ack else chosen, self._format_recommendation())
                    if ack:
                        return chosen + "\n" + ack + ("\n" + nxt if nxt else "")
                    return chosen + ("\n" + nxt if nxt else "")
        if intent == "reject_city":
            rejected = self._cities_to_reject(text)
            for city in rejected:
                self.rejected_cities.add(city)
            removed = _join_natural(rejected)
            return self._lead("Okay, I've removed " + removed + " from the list. Here are some updated options:",
                              self._format_recommendation())
        if self._is_skip_all(text):
            for slot, _q in self._slots():
                self.resolved.add(slot)
            self.pending_slot = None
            return self._lead("Sure, I'll work with what you've given me so far.", self._format_recommendation())
        if self._is_skip(text) and self.pending_slot is not None:
            self._skipped[self.pending_slot] = True
            self.resolved.add(self.pending_slot)
            nxt = self._next_question()
            if nxt is None:
                return self._lead("No problem.", self._format_recommendation())
            return "No problem, we can skip that. (say 'skip all' to skip the rest)\n" + nxt
        changed = self._update_slots(text)
        ack = self._acknowledge(changed)
        if self.pending_slot is not None and self._slot_filled(self.pending_slot):
            self.resolved.add(self.pending_slot)
        elif self.pending_slot is not None:
            crit = self._SLOT_TO_CRIT.get(self.pending_slot)
            if crit and self.query["importance"].get(crit) == 0.6:
                self._skipped[self.pending_slot] = True
                self.resolved.add(self.pending_slot)
        if ack is None:
            nxt = self._next_question()
            if (nxt is None and self.last_results is None
                    and self._has_meaningful_preferences()):
                return self._lead("Alright — here's what I'd suggest:",
                                  self._format_recommendation())
            if self.pending_slot is not None and self._is_dont_care(text):
                self._skipped[self.pending_slot] = True
                self.resolved.add(self.pending_slot)
                nxt = self._next_question()
                if nxt is None:
                    return self._lead("No problem, we'll leave that open.", self._format_recommendation())
                return "No problem, we can leave that open.\n" + nxt
            if self.last_results is not None:
                return ("Those are my picks for now. You can refine them — try "
                        "'cheaper', 'warmer', 'more culture', 'more like #1', "
                        "'not <city>', or 'restart' to begin again.")
            return random.choice(_CONFUSED_REPLIES)
        nxt = self._next_question()
        if nxt is None:
            return self._lead(ack, self._format_recommendation())
        return ack + "\n" + nxt
def render_response_text(reply):
    if not isinstance(reply, dict) or reply.get("type") != "recommendations":
        return reply
    lines = []
    if reply.get("lead"):
        lines.append(reply["lead"])
    for msg in reply.get("advisories", []):
        lines.append("Note: " + msg)
    if reply.get("summary"):
        lines.append(reply["summary"])
    lines.append(reply["header"])
    for r in reply["results"]:
        region_pretty = r["region"].replace("_", " ").title()
        lines.append("#" + str(r["rank"]) + " " + r["city"] + ", " + r["country"]
                     + " (" + region_pretty + ", " + r["temp"] + ", " + r["budget"]
                     + ")  [" + r["match"] + "]")
        lines.append("   " + r["description"])
        for reason in r.get("reasons", []):
            lines.append("   - " + reason)
        if r["strong_on"]:
            lines.append("   Strong on: " + ", ".join(r["strong_on"]))
    lines.append(reply["footer"])
    return "\n".join(lines)
