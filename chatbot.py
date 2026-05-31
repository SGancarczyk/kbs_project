# chatbot.py
# ----------------------------------------------------------------------------
# THE CHATBOT - the user interface of the Knowledge-Based System. Its job is to
# hold a real conversation: understand free-typed sentences (via nlp.py), fill
# in the user's wishes one piece ("slot") at a time, and hand the finished
# request to the inference engine, then explain the answer. There are NO
# numbered menus - the project rules forbid that. The bot also recognises
# several "intents" (greet, help, why, restart, exit) and gives many different
# responses, satisfying the "at least 15 different responses" requirement.
# ----------------------------------------------------------------------------
import knowledge_base as kb
import nlp
import inference as inf
# Words that mean "I have no preference for the thing you just asked" - they let
# the user SKIP a slot instead of being forced to answer (un-menu-like).
SKIP_WORDS = {"any", "anywhere", "whatever", "skip", "none", "no", "nope", "idk", "anything"}
SKIP_PHRASES = {"doesn't matter", "does not matter", "dont care", "don't care", "do not care", "no preference", "not sure"}

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
_ACK_OPENERS = ["Perfect!", "Got it!", "Sure thing —", "Noted —",
                "Great choice —", "Awesome —", "Love it —"]


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
    def _slots(self):
        return [
            ("continent", "Where would you like to go? (e.g. Europe, Asia, South America - or say 'anywhere')"),
            ("duration", "How long is the trip? (day trip, weekend, short trip, one week, or long trip)"),
            ("budget", "What's your budget like? (cheap, mid-range, or luxury)"),
            ("climate", "Any climate preference? (warm, mild, cold - or say 'no')"),
            ("lifestyle", "What do you enjoy most? You can also say what to avoid (e.g. culture, food, nature, beaches, avoid nightlife)."),
        ]
    def _slot_filled(self, slot):
        # Has this slot already received a value in the query, either as a wish
        # or as an explicit exclusion?
        if slot == "continent":
            return bool(self.query["regions_include"]) or bool(self.query["regions_exclude"])
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
        return ("Hi! I'm your Travel Destination Assistant. Tell me about your ideal trip in your own words - "
                "you can mention a place, budget, trip length, climate, or what you enjoy, all at once or bit by bit.\n"
                "Tips: say 'anywhere' / 'no' to skip a question, or 'skip all' to skip the rest and see picks now; "
                "'go' when you want my picks, 'why' to see my reasoning, 'restart' to start over, or 'exit' to quit.\n" + self._next_question())
    def help_text(self):
        # The help message (RESPONSE TYPE 2).
        return ("I match you to travel destinations from a database of 560 cities. Just describe what you want - "
                "for example: 'a cheap warm week in Asia, I love food and culture'. I understand full sentences, "
                "fix small typos, and handle 'not' (e.g. 'not Africa'). Commands: 'go' or 'skip all' (get picks now), "
                "'why' (reasoning), 'skip'/'anywhere'/'no' (skip one question), 'restart', 'exit'.")
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
        # REJECTION: only possible once we have actually shown some picks. We
        # require a rejection cue ("not", "remove", ...) AND that it points at a
        # city/position currently on screen, so an ordinary preference such as
        # "warm" is never misread as "remove a city".
        if self.last_results and self.last_results.get("results"):
            if self._has_rejection_cue(t) and self._cities_to_reject(text):
                return "reject_city"
        # Anything else is treated as the user giving us preferences.
        return "provide"
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
        all_maps = [
            kb.CONTINENT_SYNONYMS,
            kb.CLIMATE_SYNONYMS,
            kb.BUDGET_SYNONYMS,
            kb.DURATION_SYNONYMS,
            kb.LIFESTYLE_SYNONYMS,
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
        ])
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

        dur = nlp.extract(text, kb.DURATION_SYNONYMS, allow_spellcheck=self._allow_spellcheck_for("duration"), blocking_vocabulary=self._blocking_vocabulary_for(kb.DURATION_SYNONYMS))
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
        return changed
    def _next_opener(self):
        # Pick the next rotating opener (e.g. "Perfect!", "Noted —") and advance
        # the counter, wrapping around. Rotating instead of random keeps the
        # behaviour deterministic (nice for tests) while still feeling varied.
        opener = _ACK_OPENERS[self._ack_index % len(_ACK_OPENERS)]
        self._ack_index += 1
        return opener
    def _acknowledge(self, changed):
        # Build a SHORT, NATURAL confirmation of what we just understood, e.g.
        # "Perfect! Europe, on a budget, and you enjoy culture and food."
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
        # "Noted — mid-range, warm climate, and you enjoy culture and food."
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
        lines = []
        # Surface the forward-chaining advisories first, if any fired.
        for msg in out["advisories"]:
            lines.append("Heads up: " + msg)
        if not results:
            lines.append("I couldn't find anything matching all of that. Try relaxing one wish - "
                         "remove a region, widen the budget, or drop the climate. (type 'restart' to redo)")
            return "\n".join(lines)
        # Conversational header (RESPONSE TYPE 10), e.g.
        # "Here are your top picks — chosen from 312 destinations that match your region:"
        lines.append("Here are your top picks — chosen from " + str(out["pool_size"]) + " destinations that match your region:")
        rank = 1
        for conf, dest, score in results:
            header = ("#" + str(rank) + " " + dest["city"] + ", " + dest["country"] + " (" + dest["region"] + ", " + str(dest["avg_temp_yearly"]) + "C, " + dest["budget_level"] + ")")
            lines.append(header)
            lines.append("   " + inf.explain(dest, score, self.query))
            lines.append("   " + dest["short_description"])
            rank += 1
        lines.append("Type 'why' for the reasoning, 'restart' to search again, or 'exit' to leave.")
        return "\n".join(lines)
    def _explain_last(self):
        # The "why" intent: a deeper reasoning breakdown of the last picks
        # (RESPONSE TYPE 12). If we have not recommended yet, say so.
        if not self.last_results or not self.last_results["results"]:
            return "I haven't recommended anything yet - tell me what you're looking for first."
        lines = ["Here's my reasoning (each criterion becomes a certainty factor: CF>0 supports the city, CF<0 counts against it; all are combined MYCIN-style into the confidence):"]
        for conf, dest, score in self.last_results["results"]:
            comps = ", ".join(name + " m=" + str(round(m, 2)) + "/cf=" + ("+" if cf >= 0 else "") + str(round(cf, 2)) for name, (m, cf) in score["details"].items()) or "no criteria given"
            lines.append(dest["city"] + ": " + comps + " -> confidence " + str(round(conf, 2)))
        return "\n".join(lines)
    def respond(self, text):
        # THE MAIN ENTRY POINT. Given one user message, return the bot's reply as
        # a string. Keeping all logic here (and returning a string) makes the bot
        # easy to test automatically by feeding it a scripted conversation.
        if not text or not text.strip():
            return "Go ahead - tell me about your trip, or type 'help'."
        intent = self._detect_intent(text)
        if intent == "exit":
            self.finished = True
            return "Thanks for chatting - safe travels! Goodbye."          # RESPONSE TYPE 15
        if intent == "help":
            return self.help_text()
        if intent == "restart":
            self.reset()
            return "Okay, fresh start.\n" + self._next_question()           # RESPONSE TYPE 14
        if intent == "why":
            return self._explain_last()
        if intent == "recommend":
            return self._format_recommendation()
        if intent == "greet":
            return "Hello! " + (self._next_question() or "Tell me your preferences and I'll suggest destinations.")
        if intent == "reject_city":
            # Remember the rejected cities for the rest of the session, then
            # immediately recompute the picks. _format_recommendation() filters
            # out everything in self.rejected_cities, so the rejected places are
            # gone and fresh ones take their slots.
            rejected = self._cities_to_reject(text)
            for city in rejected:
                self.rejected_cities.add(city)
            removed = _join_natural(rejected)
            return ("Got it, I've removed " + removed + ". Here are updated picks:\n"
                    + self._format_recommendation())
        # intent == "provide": the user is giving preferences (the common case).
        # 'skip all' -> stop asking the rest and recommend with what we have.
        if self._is_skip_all(text):
            for slot, _q in self._slots():
                self.resolved.add(slot)
            self.pending_slot = None
            return "Sure - skipping the rest and using what I have so far.\n" + self._format_recommendation()  # RESPONSE TYPE 19 (skip-all)
        # If this message is a pure "skip" and we just asked a question, mark
        # that one slot resolved so we move on instead of nagging.
        if self._is_skip(text) and self.pending_slot is not None:
            self.resolved.add(self.pending_slot)
            nxt = self._next_question()
            if nxt is None:
                return "No problem.\n" + self._format_recommendation()
            return "No problem, skipping that. (say 'skip all' to skip the rest)\n" + nxt   # RESPONSE TYPE 18 (skip ack)
        # Otherwise, fill whatever slots the sentence contains.
        changed = self._update_slots(text)
        ack = self._acknowledge(changed)
        # If the slot we had asked about is now filled, mark it resolved.
        if self.pending_slot is not None and self._slot_filled(self.pending_slot):
            self.resolved.add(self.pending_slot)
        if ack is None:
            # Nothing recognised: gently clarify instead of failing silently.
            return ("I didn't catch a known option there. You can mention a continent, a budget "
                    "(cheap/mid/luxury), a trip length, a climate (warm/mild/cold), or interests like "
                    "food, nature, or nightlife. Or type 'go' for picks with what I have.")  # RESPONSE TYPE 9
        nxt = self._next_question()
        if nxt is None:
            # We have enough - recommend right away.
            return ack + "\n" + self._format_recommendation()
        return ack + "\n" + nxt
