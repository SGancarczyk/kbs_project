# nlp.py
# ----------------------------------------------------------------------------
# THE NATURAL-LANGUAGE LAYER of the chatbot interface.
# It turns a messy user sentence into clean facts for the rest of the KBS:
# normalize, tokenize, lemmatize, match domain vocabulary, correct small typos,
# and handle negation such as "not asia" or "avoid nightlife".
#
# REWRITE: replaced hand-rolled normalize/tokenize/lemmatize with standard NLP
# libraries from the course notebooks (NLP_Hands_on__3_.ipynb, chatbot__4_.ipynb):
#   - normalize() / tokenize()  : replaced with nltk.word_tokenize()
#   - lemmatize()               : replaced with WordNetLemmatizer.lemmatize()
#                                 using nltk.pos_tag() to pick the correct POS
#
# 2. Spell-checking is disconnected The comments at the top of nlp.py claim that
# spell correction was kept from the original version. However, while the helper
# function _one_adjacent_transposition_away exists, it is never actually called
# inside the extract() function or _match_exact_at. The _spell_correct function
# itself is completely missing.
#
# The PUBLIC API is IDENTICAL to the original so chatbot.py needs zero changes:
#   extract(text, keyword_map, allow_spellcheck, blocking_vocabulary) -> dict
#   _normalize_common_phrases(text)                                   -> str
# All internal logic (negation propagation, multi-word phrase matching,
# blocking vocabulary, negation connectors) is also preserved exactly.
# ----------------------------------------------------------------------------
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data once (silent if already present).
# punkt / punkt_tab              -> tokenizer models used by word_tokenize
# averaged_perceptron_tagger_eng -> POS tagger used to guide lemmatization
# wordnet                        -> lemma dictionary used by WordNetLemmatizer
for _pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "wordnet"):
    nltk.download(_pkg, quiet=True)

# Module-level singleton so we only construct it once per process.
_lemmatizer = WordNetLemmatizer()

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
    (r"\bsmth\b",  "something"),
    (r"\bpls\b",   "please"),
    (r"\bplz\b",   "please"),
    (r"\bu\b",     "you"),
    (r"\bbc\b",    "because"),
    (r"\bcuz\b",   "because"),
    (r"\bcoz\b",   "because"),
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


def _pos_to_wordnet(tag):
    # Convert a Penn Treebank POS tag (from nltk.pos_tag) into a WordNet POS
    # constant so WordNetLemmatizer.lemmatize() picks the right inflection table.
    # Without this, lemmatize() defaults to NOUN for everything, which is wrong
    # for verbs ("running" -> "running") and adjectives ("better" -> "better").
    #   J* -> adjective   N* -> noun (default)
    #   V* -> verb        R* -> adverb
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return "n"   # safe default: noun


def tokenize(text):
    # REWRITE: replaced normalize(text).split() with nltk.word_tokenize().
    # nltk.word_tokenize handles contractions ("don't" -> ["do", "n't"]),
    # hyphenated words, and punctuation more accurately than a simple split,
    # matching the approach demonstrated in NLP_Hands_on__3_.ipynb section 1.1.
    # We return plain lowercase strings, exactly as the original did, so the
    # rest of extract() sees no change in token format.
    return [t.lower() for t in word_tokenize(text.lower())]


def lemmatize(token, pos_tag="NN"):
    # REWRITE: replaced the hand-written suffix-stripping rules with
    # WordNetLemmatizer.lemmatize(), using the WordNet POS constant derived from
    # the Penn Treebank tag. This gives correct forms like:
    #   "beaches" -> "beach"  (was correct with old rules too)
    #   "running" -> "run"    (old rules kept it as "running")
    #   "better"  -> "good"   (old rules kept it as "better")
    # The pos_tag parameter defaults to "NN" (noun) so callers that do not
    # pass a tag get the same behaviour as the old lemmatize(token).
    wn_pos = _pos_to_wordnet(pos_tag)
    return _lemmatizer.lemmatize(token, pos=wn_pos)




def _one_adjacent_transposition_away(token, candidate):
    # Handles common typos like "chaep" -> "cheap" and "asai" -> "asia".
    if len(token) != len(candidate):
        return False
    diffs = [i for i, (a, b) in enumerate(zip(token, candidate)) if a != b]
    if len(diffs) != 2:
        return False
    i, j = diffs
    return j == i + 1 and token[i] == candidate[j] and token[j] == candidate[i]




def _add_unique(target, value):
    if value not in target:
        target.append(value)


def _match_exact_at(raw_tokens, lemma_tokens, start, keyword_map):
    # Prefer multi-word phrases before single tokens, trying longest match first.
    # We use space-joined candidates only — the old no-space join (e.g. "to"+"go"
    # -> "togo") was a workaround for the hand-rolled tokenizer that split on
    # punctuation. nltk.word_tokenize keeps run-together words as single tokens
    # ("southamerica" stays "southamerica"), so the no-space join is unnecessary
    # and actively harmful: it caused "to"+"go" to match the country "Togo".
    max_len = min(5, len(raw_tokens) - start)
    for size in range(max_len, 0, -1):
        raw_piece   = raw_tokens[start:start + size]
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
    # Returns canonical values the user wants to include and avoid:
    # {"include": [...], "exclude": [...]}. blocking_vocabulary
    # contains keywords from all other slots; it prevents negation from leaking
    # across categories, e.g. "not expensive, South America" should not exclude
    # South America when we are extracting regions.
    #
    # FIRST clean up informal shorthand ("wanna", "&", "w/o", ...) on the raw
    # sentence, so the tokenizer and every matcher below see plain English. We do
    # this here (not in the chatbot) so EVERY call to extract() benefits, no
    # matter which slot/vocabulary it is run for.
    text = _normalize_common_phrases(text)

    # REWRITE: build tokens using the new nltk-based tokenize() and lemmatize().
    # We also POS-tag the full token list once so each lemmatize() call gets the
    # right part-of-speech context — matching the pipeline in the hands-on
    # notebook (NLP_Hands_on__3_.ipynb sections 1.1, 1.2, 1.5).
    raw_tokens   = tokenize(text)
    pos_tags     = nltk.pos_tag(raw_tokens)    # [(token, POS-tag), ...]
    lemma_tokens = [lemmatize(tok, tag) for tok, tag in pos_tags]

    vocabulary          = set(keyword_map.keys())
    blocking_vocabulary = set(blocking_vocabulary or [])
    include = []
    exclude = []

    pending_negate   = False
    negate_ttl       = 0
    negated_recently = False
    i = 0
    while i < len(raw_tokens):
        raw   = raw_tokens[i]
        lemma = lemma_tokens[i]

        if raw in NEGATION_CONNECTORS and negated_recently:
            # "not africa or asia" means both values should be excluded.
            pending_negate   = True
            negate_ttl       = 3
            negated_recently = False
            i += 1
            continue

        if raw not in NEGATION_CONNECTORS:
            negated_recently = False

        # Stop negation completely if we hit the end of a sentence
        if raw in {".", "!", "?", ";"}:
            pending_negate = False
            negate_ttl = 0

        if lemma in NEGATION_WORDS or raw in NEGATION_WORDS:
            pending_negate = True
            negate_ttl     = 20
            i += 1
            continue

        matched, consumed = _match_exact_at(raw_tokens, lemma_tokens, i, keyword_map)

        # If a pending negation reaches a known keyword that belongs to another
        # slot, consume the negation there instead of letting it leak forward to
        # the next keyword in this vocabulary. Example: in region extraction,
        # "not expensive, South America" sees "expensive" as a blocker, so
        # South America remains an included region.
        if matched is None and pending_negate and blocking_vocabulary:
            blocked, blocked_consumed = _match_exact_at(
                raw_tokens, lemma_tokens, i, {k: k for k in blocking_vocabulary}
            )
            if blocked is not None and blocked not in keyword_map:
                pending_negate   = False
                negate_ttl       = 0
                negated_recently = True
                i += blocked_consumed
                continue


        if matched is not None:
            value = keyword_map[matched]
            if pending_negate:
                _add_unique(exclude, value)
                pending_negate   = False
                negate_ttl       = 0
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