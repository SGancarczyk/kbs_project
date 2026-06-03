import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
_REQUIRED_NLTK = ("punkt", "punkt_tab", "averaged_perceptron_tagger",
                  "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4")
def ensure_nltk_data():
    for pkg in _REQUIRED_NLTK:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    try:
        word_tokenize("warm weather")
        nltk.pos_tag(["warm", "weather"])
        WordNetLemmatizer().lemmatize("beaches")
    except LookupError as exc:
        raise RuntimeError(
            "Required NLTK data is missing and could not be downloaded "
            "automatically. Run this once with internet access:\n"
            "    python -c \"import nltk; "
            + "; ".join("nltk.download('" + p + "')" for p in _REQUIRED_NLTK)
            + "\"\nOriginal error: " + str(exc)
        ) from exc
ensure_nltk_data()
_lemmatizer = WordNetLemmatizer()
NEGATION_WORDS = {"not", "no", "dont", "don't", "n't", "avoid", "without", "except", "never", "exclude", "hate", "skip"}
NEGATION_CONNECTORS = {"or", "and", "nor"}
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
    result = text
    for pattern, replacement in _INFORMAL_WORD_REPLACEMENTS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = result.replace("w/o", "without").replace("W/o", "without").replace("W/O", "without")
    result = result.replace("w/", "with").replace("W/", "with")
    result = result.replace("&", " and ")
    return result
def _pos_to_wordnet(tag):
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return "n"
def tokenize(text):
    return [t.lower() for t in word_tokenize(text.lower())]
def lemmatize(token, pos_tag="NN"):
    wn_pos = _pos_to_wordnet(pos_tag)
    return _lemmatizer.lemmatize(token, pos=wn_pos)
def _one_adjacent_transposition_away(token, candidate):
    if len(token) != len(candidate):
        return False
    diffs = [i for i, (a, b) in enumerate(zip(token, candidate)) if a != b]
    if len(diffs) != 2:
        return False
    i, j = diffs
    return j == i + 1 and token[i] == candidate[j] and token[j] == candidate[i]
def edit_distance(a, b):
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,
                         cur[j - 1] + 1,
                         prev[j - 1] + cost)
        prev = cur
    return prev[n]
_SPELLCHECK_BLOCKLIST = {
    "like", "lake", "want", "wants", "good", "great", "place", "places",
    "there", "where", "these", "those", "thing", "things", "think", "really",
    "about", "around", "would", "could", "should", "going", "visit", "trip",
    "trips", "love", "need", "very", "some", "more", "make", "take", "time",
}
def _spell_correct(token, vocabulary):
    if len(token) < 5 or token in _SPELLCHECK_BLOCKLIST:
        return None
    for candidate in vocabulary:
        if " " in candidate or len(candidate) < 5:
            continue
        if candidate[0] != token[0] or candidate[-1] != token[-1]:
            continue
        if _one_adjacent_transposition_away(token, candidate):
            return candidate
    return None
def _add_unique(target, value):
    if value not in target:
        target.append(value)
def _match_exact_at(raw_tokens, lemma_tokens, start, keyword_map):
    max_len = min(5, len(raw_tokens) - start)
    for size in range(max_len, 0, -1):
        raw_piece = raw_tokens[start:start + size]
        lemma_piece = lemma_tokens[start:start + size]
        candidates = [
            " ".join(raw_piece),
            " ".join(lemma_piece),
        ]
        for key in candidates:
            if key in keyword_map:
                return key, size
    return None, 0
def extract(text, keyword_map, allow_spellcheck=True, blocking_vocabulary=None):
    text = _normalize_common_phrases(text)
    raw_tokens = tokenize(text)
    pos_tags = nltk.pos_tag(raw_tokens)
    lemma_tokens = [lemmatize(tok, tag) for tok, tag in pos_tags]
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
        if raw == "may" and pos_tags[i][1] == "MD":
            i += 1
            continue
        if raw in NEGATION_CONNECTORS and negated_recently:
            pending_negate = True
            negate_ttl = 3
            negated_recently = False
            i += 1
            continue
        if raw not in NEGATION_CONNECTORS:
            negated_recently = False
        if raw in {".", "!", "?", ";"}:
            pending_negate = False
            negate_ttl = 0
        if lemma in NEGATION_WORDS or raw in NEGATION_WORDS:
            phrase_matched, phrase_consumed = _match_exact_at(raw_tokens, lemma_tokens, i, keyword_map)
            if phrase_matched is not None and phrase_consumed > 1:
                value = keyword_map[phrase_matched]
                if pending_negate:
                    _add_unique(exclude, value)
                    pending_negate = False
                    negate_ttl = 0
                    negated_recently = True
                else:
                    _add_unique(include, value)
                i += phrase_consumed
                continue
            pending_negate = True
            negate_ttl = 20
            i += 1
            continue
        matched, consumed = _match_exact_at(raw_tokens, lemma_tokens, i, keyword_map)
        if matched is None and allow_spellcheck:
            corrected = _spell_correct(raw, vocabulary) or _spell_correct(lemma, vocabulary)
            if corrected is not None:
                matched, consumed = corrected, 1
        if matched is None and pending_negate and blocking_vocabulary:
            blocked, blocked_consumed = _match_exact_at(
                raw_tokens, lemma_tokens, i, {k: k for k in blocking_vocabulary}
            )
            if blocked is not None and blocked not in keyword_map:
                pending_negate = False
                negate_ttl = 0
                negated_recently = True
                i += blocked_consumed
                continue
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
