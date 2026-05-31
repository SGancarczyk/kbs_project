# nlp.py
# ----------------------------------------------------------------------------
# THE NATURAL-LANGUAGE LAYER of the chatbot interface.
# It turns a messy user sentence into clean facts for the rest of the KBS:
# normalize, tokenize, lemmatize, match domain vocabulary, correct small typos,
# and handle negation such as "not asia" or "avoid nightlife". The whole module
# uses only Python's standard library, so the language understanding logic stays
# transparent and course compliant.
# ----------------------------------------------------------------------------
import re

NEGATION_WORDS = {"not", "no", "dont", "don't", "avoid", "without", "except", "never", "exclude", "hate", "skip"}
NEGATION_CONNECTORS = {"or", "and", "nor"}

# ----------------------------------------------------------------------------
# INFORMAL-PHRASE NORMALIZATION (runs BEFORE tokenization).
# The tokenizer + spell corrector are good at single mistyped words, but they
# cannot expand chat shorthand like "wanna" (one token that should become two
# words) or symbols like "&" / "w/". Those slip through because they are not
# typos of any vocabulary word - they are different tokens entirely. So we do a
# cheap regex pass on the RAW sentence first, rewriting common informal inputs
# into their plain-English form. Everything downstream (tokenize, lemmatize,
# vocabulary matching, negation) then benefits automatically.
# ----------------------------------------------------------------------------
# Word-shaped replacements: \b...\b makes sure we only swap WHOLE words, so
# "u" -> "you" never touches "blue", and "bc" never touches "abcdef".
_INFORMAL_WORD_REPLACEMENTS = [
    (r"\bwanna\b", "want to"),
    (r"\bgonna\b", "going to"),
    (r"\bgotta\b", "got to"),
    (r"\bsmth\b", "something"),
    (r"\bpls\b", "please"),
    (r"\bplz\b", "please"),
    (r"\bu\b", "you"),
    (r"\bbc\b", "because"),
    (r"\bcuz\b", "because"),
    (r"\bcoz\b", "because"),
]


def _normalize_common_phrases(text):
    # Expand chat shorthand into plain English. Order matters for the symbol
    # rules: "w/o" (without) must be handled BEFORE "w/" (with), otherwise the
    # "w/" rule would fire first and leave a stray "o" behind.
    result = text
    for pattern, replacement in _INFORMAL_WORD_REPLACEMENTS:
        # re.IGNORECASE so "Wanna" and "WANNA" are both caught.
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    # Symbol-based shorthand. These are plain substring swaps because regex word
    # boundaries (\b) do not behave intuitively next to "/" and "&".
    result = result.replace("w/o", "without").replace("W/o", "without").replace("W/O", "without")
    result = result.replace("w/", "with").replace("W/", "with")
    result = result.replace("&", " and ")
    return result


def normalize(text):
    # Lowercase and replace punctuation with spaces so "South-America" and
    # "south america" are handled the same way.
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def tokenize(text):
    return normalize(text).split()


def lemmatize(token):
    # Small rule-based plural lemmatizer focused on the project vocabulary.
    t = token
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    for suffix in ("ches", "shes", "ses", "xes", "zes"):
        if len(t) > 4 and t.endswith(suffix):
            return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def edit_distance(a, b):
    # Standard Levenshtein distance, implemented directly so no external NLP
    # package is needed.
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def _one_adjacent_transposition_away(token, candidate):
    # Handles common typos like "chaep" -> "cheap" and "asai" -> "asia".
    if len(token) != len(candidate):
        return False
    diffs = [i for i, (a, b) in enumerate(zip(token, candidate)) if a != b]
    if len(diffs) != 2:
        return False
    i, j = diffs
    return j == i + 1 and token[i] == candidate[j] and token[j] == candidate[i]


def _spell_correct(token, vocabulary, max_ratio=0.26):
    # Conservative spell correction. It fixes genuine typos but avoids rewriting
    # unrelated words across categories, for example cheap -> heat, mild -> mid,
    # south -> short, or range -> france.
    if token in vocabulary:
        return token
    if len(token) <= 3:
        return token

    best_word = None
    best_distance = None
    for candidate in vocabulary:
        # Do not correct into very short keywords such as "mid". These are too
        # easy to hit accidentally from unrelated words.
        if len(candidate) <= 3:
            continue
        if _one_adjacent_transposition_away(token, candidate):
            return candidate
        # Normal edit-distance corrections must preserve the first and last
        # letter. This keeps useful fixes such as "cuisne" -> "cuisine" but
        # blocks semantic false positives such as "short" -> "shore".
        if token[0] != candidate[0] or token[-1] != candidate[-1]:
            continue
        d = edit_distance(token, candidate)
        ratio = d / max(len(token), len(candidate))
        if ratio <= max_ratio and (best_distance is None or d < best_distance):
            best_distance = d
            best_word = candidate
    return best_word if best_word is not None else token


def _add_unique(target, value):
    if value not in target:
        target.append(value)


def _match_exact_at(raw_tokens, lemma_tokens, start, keyword_map):
    # Prefer multi-word phrases before single tokens. The vocabulary stores some
    # phrases without spaces, such as southamerica, middleeast, unitedstates.
    max_len = min(3, len(raw_tokens) - start)
    for size in range(max_len, 0, -1):
        raw_piece = raw_tokens[start:start + size]
        lemma_piece = lemma_tokens[start:start + size]
        candidates = [
            "".join(raw_piece),
            "".join(lemma_piece),
            " ".join(raw_piece),
            " ".join(lemma_piece),
        ]
        for key in candidates:
            if key in keyword_map:
                return key, size
    return None, 0


def extract(text, keyword_map, allow_spellcheck=True, blocking_vocabulary=None):
    # Returns canonical values the user wants to include and avoid:
    # {"include": [...], "exclude": [...]}. Exact phrase matches always run.
    # Spell correction can be switched off by the chatbot. blocking_vocabulary
    # contains keywords from all other slots; it prevents negation from leaking
    # across categories, e.g. "not expensive, South America" should not exclude
    # South America when we are extracting regions.
    #
    # FIRST clean up informal shorthand ("wanna", "&", "w/o", ...) on the raw
    # sentence, so the tokenizer and every matcher below see plain English. We do
    # this here (not in the chatbot) so EVERY call to extract() benefits, no
    # matter which slot/vocabulary it is run for.
    text = _normalize_common_phrases(text)
    raw_tokens = tokenize(text)
    lemma_tokens = [lemmatize(t) for t in raw_tokens]
    vocabulary = set(keyword_map.keys())
    blocking_vocabulary = set(blocking_vocabulary or [])
    include = []
    exclude = []

    pending_negate = False
    negate_ttl = 0
    negated_recently = False
    i = 0
    while i < len(raw_tokens):
        raw = raw_tokens[i]
        lemma = lemma_tokens[i]

        if raw in NEGATION_CONNECTORS and negated_recently:
            # "not africa or asia" means both values should be excluded.
            pending_negate = True
            negate_ttl = 3
            negated_recently = False
            i += 1
            continue

        if raw not in NEGATION_CONNECTORS:
            negated_recently = False

        if lemma in NEGATION_WORDS or raw in NEGATION_WORDS:
            pending_negate = True
            negate_ttl = 4
            i += 1
            continue

        matched, consumed = _match_exact_at(raw_tokens, lemma_tokens, i, keyword_map)

        # If a pending negation reaches a known keyword that belongs to another
        # slot, consume the negation there instead of letting it leak forward to
        # the next keyword in this vocabulary. Example: in region extraction,
        # "not expensive, South America" sees "expensive" as a blocker, so
        # South America remains an included region.
        if matched is None and pending_negate and blocking_vocabulary:
            blocked, blocked_consumed = _match_exact_at(raw_tokens, lemma_tokens, i, {k: k for k in blocking_vocabulary})
            if blocked is not None and blocked not in keyword_map:
                pending_negate = False
                negate_ttl = 0
                negated_recently = True
                i += blocked_consumed
                continue

        if matched is None and allow_spellcheck:
            # Spell correction is only done for a single token, never for whole
            # phrases, because phrase matching is exact and safer.
            corrected = _spell_correct(lemma, vocabulary)
            if corrected in keyword_map:
                matched = corrected
                consumed = 1
            else:
                corrected_raw = _spell_correct(raw, vocabulary)
                if corrected_raw in keyword_map:
                    matched = corrected_raw
                    consumed = 1

        if matched is not None:
            value = keyword_map[matched]
            if pending_negate:
                _add_unique(exclude, value)
                pending_negate = False
                negate_ttl = 0
                negated_recently = True
            else:
                _add_unique(include, value)
            i += consumed
            continue

        if pending_negate:
            negate_ttl -= 1
            if negate_ttl <= 0:
                pending_negate = False

        i += 1

    return {"include": include, "exclude": exclude}
