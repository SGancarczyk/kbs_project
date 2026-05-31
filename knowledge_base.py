# knowledge_base.py
# ----------------------------------------------------------------------------
# THE KNOWLEDGE BASE of our Knowledge-Based System (KBS).
# In KBS theory (see course "Knowledge representation and reasoning" deck) the
# Knowledge Base is the part that STORES what the system knows. It does NOT
# reason; reasoning happens later in the inference engine. Here we keep two
# kinds of knowledge:
#   1) SIMPLE RELATIONAL KNOWLEDGE -> the table of travel cities (one row =
#      one destination, columns = its attributes). This is loaded from the CSV.
#   2) DOMAIN VOCABULARY -> dictionaries that map the words a human types
#      ("cheap", "warm", "beach") to the exact values stored in the dataset
#      ("Budget", "warm", "beaches"). The chatbot needs this to translate
#      natural language into facts it can match against the data.
# ----------------------------------------------------------------------------
import csv
import json
import os
# The nine "lifestyle" columns. Each city is scored 1..5 on each of these.
# We list them once here so every other file imports the SAME order (important
# later: the cosine-similarity ranking treats these scores as a vector, and a
# vector only makes sense if the order of its components is fixed).
LIFESTYLE_DIMENSIONS = ["culture", "adventure", "nature", "beaches", "nightlife", "cuisine", "wellness", "urban", "seclusion"]
# The exact text values the dataset uses, so we never guess/misspell them.
REGIONS = ["europe", "asia", "africa", "north_america", "south_america", "oceania", "middle_east"]
BUDGET_LEVELS = ["Budget", "Mid-range", "Luxury"]
DURATIONS = ["Day trip", "Weekend", "Short trip", "One week", "Long trip"]
CLIMATES = ["cold", "mild", "warm"]
def yearly_avg_temp(avg_temp_monthly):
    # CONCEPT: "derived knowledge". The dataset stores 12 monthly temperatures
    # but never states a single yearly temperature. We DERIVE it by averaging
    # the 12 monthly "avg" values. We keep this NUMBER (not only a label)
    # because the fuzzy-logic layer needs the real temperature to compute a
    # graded membership ("how warm is it?" rather than just "warm: yes/no").
    # avg_temp_monthly looks like {"1":{"avg":3.7,...}, ..., "12":{"avg":4.7,...}}
    monthly_avgs = [month["avg"] for month in avg_temp_monthly.values()]
    return sum(monthly_avgs) / len(monthly_avgs)
def climate_label(yearly):
    # A simple crisp label, handy for quick printing/reference. The real graded
    # reasoning about climate happens with fuzzy membership in fuzzy.py.
    if yearly < 15:
        return "cold"
    if yearly < 22:
        return "mild"
    return "warm"
def load_destinations(csv_path=None):
    # Reads the CSV file and turns every row into a clean Python dictionary.
    # We use Python's built-in `csv` module (not a heavy library) so you can see
    # exactly what happens to every field - the dataset rules say the system
    # must be ours, and reading our own data keeps it fully transparent.
    if csv_path is None:
        # Default to the dataset that sits next to this file, so `python main.py`
        # works no matter which folder the grader runs it from.
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, "Worldwide Travel Cities Dataset (Ratings and Climate).csv")
    destinations = []
    # newline="" + utf-8 is the standard safe way to open a CSV in Python.
    with open(csv_path, newline="", encoding="utf-8") as f:
        # DictReader reads the header row once and then gives every following
        # row as a dict {column_name: value}. Values arrive as strings, so we
        # convert the ones that should be numbers/lists below.
        reader = csv.DictReader(f)
        for row in reader:
            # Two columns are stored as JSON text inside the CSV, so we parse
            # them back into real Python objects (a dict and a list).
            row["avg_temp_monthly"] = json.loads(row["avg_temp_monthly"])
            row["ideal_durations"] = json.loads(row["ideal_durations"])
            # Convert each lifestyle score from text ("5") to an integer (5),
            # because we will do math (cosine similarity) on them later.
            for dim in LIFESTYLE_DIMENSIONS:
                row[dim] = int(row[dim])
            # Attach two brand-new derived facts on the row: the yearly average
            # temperature (a number, for fuzzy reasoning) and a crisp climate
            # label (for quick reference).
            yearly = yearly_avg_temp(row["avg_temp_monthly"])
            row["avg_temp_yearly"] = round(yearly, 1)
            row["climate"] = climate_label(yearly)
            destinations.append(row)
    return destinations
# ----------------------------------------------------------------------------
# DOMAIN VOCABULARY
# These dictionaries are pure knowledge: "if the user says the word on the LEFT,
# they mean the dataset value on the RIGHT". The chatbot's language layer
# (nlp.py) will look words up here. Keeping them in the Knowledge Base (not in
# the chatbot code) follows the KBS principle of separating WHAT we know from
# HOW we process it - the same idea your slides call "separation of knowledge
# from its processing".
# ----------------------------------------------------------------------------
# Words -> region. We add common variants/adjectives so real sentences match.
CONTINENT_SYNONYMS = {"europe": "europe", "european": "europe", "spain": "europe", "spanish": "europe", "france": "europe", "french": "europe", "italy": "europe", "italian": "europe", "greece": "europe", "greek": "europe", "portugal": "europe", "portuguese": "europe", "germany": "europe", "german": "europe", "netherlands": "europe", "switzerland": "europe", "austria": "europe", "croatia": "europe", "ireland": "europe", "scotland": "europe", "england": "europe", "britain": "europe", "british": "europe", "uk": "europe", "poland": "europe", "czech": "europe", "hungary": "europe", "sweden": "europe", "norway": "europe", "denmark": "europe", "finland": "europe", "scandinavia": "europe", "scandinavian": "europe", "nordic": "europe", "balkans": "europe", "balkan": "europe", "asia": "asia", "asian": "asia", "japan": "asia", "japanese": "asia", "china": "asia", "chinese": "asia", "thailand": "asia", "thai": "asia", "india": "asia", "indian": "asia", "vietnam": "asia", "indonesia": "asia", "bali": "asia", "korea": "asia", "korean": "asia", "malaysia": "asia", "singapore": "asia", "philippines": "asia", "cambodia": "asia", "nepal": "asia", "taiwan": "asia", "africa": "africa", "african": "africa", "egypt": "africa", "egyptian": "africa", "morocco": "africa", "moroccan": "africa", "kenya": "africa", "tanzania": "africa", "nigeria": "africa", "ethiopia": "africa", "ghana": "africa", "tunisia": "africa", "namibia": "africa", "senegal": "africa", "mauritius": "africa", "america": "north_america", "american": "north_america", "usa": "north_america", "unitedstates": "north_america", "canada": "north_america", "canadian": "north_america", "mexico": "north_america", "mexican": "north_america", "north": "north_america", "northamerica": "north_america", "south": "south_america", "southamerica": "south_america", "latin": "south_america", "brazil": "south_america", "brazilian": "south_america", "argentina": "south_america", "peru": "south_america", "chile": "south_america", "colombia": "south_america", "bolivia": "south_america", "ecuador": "south_america", "oceania": "oceania", "australia": "oceania", "australian": "oceania", "pacific": "oceania", "newzealand": "oceania", "fiji": "oceania", "middleeast": "middle_east", "middle": "middle_east", "arabian": "middle_east", "gulf": "middle_east", "dubai": "middle_east", "uae": "middle_east", "emirates": "middle_east", "qatar": "middle_east", "jordan": "middle_east", "oman": "middle_east", "lebanon": "middle_east"}
# Words -> budget level.
BUDGET_SYNONYMS = {"cheap": "Budget", "cheaper": "Budget", "cheapest": "Budget", "affordable": "Budget", "afford": "Budget", "budget": "Budget", "budgetfriendly": "Budget", "inexpensive": "Budget", "low": "Budget", "economical": "Budget", "economic": "Budget", "frugal": "Budget", "broke": "Budget", "tight": "Budget", "shoestring": "Budget", "backpacker": "Budget", "backpacking": "Budget", "moderate": "Mid-range", "mid": "Mid-range", "midrange": "Mid-range", "medium": "Mid-range", "average": "Mid-range", "normal": "Mid-range", "reasonable": "Mid-range", "standard": "Mid-range", "decent": "Mid-range", "luxury": "Luxury", "luxurious": "Luxury", "expensive": "Luxury", "premium": "Luxury", "high": "Luxury", "lavish": "Luxury", "fancy": "Luxury", "splurge": "Luxury", "upscale": "Luxury", "highend": "Luxury", "fivestar": "Luxury", "posh": "Luxury", "deluxe": "Luxury", "exclusive": "Luxury", "opulent": "Luxury"}
# Words -> trip duration.
DURATION_SYNONYMS = {"day": "Day trip", "daytrip": "Day trip", "overnight": "Weekend", "weekend": "Weekend", "getaway": "Weekend", "short": "Short trip", "few": "Short trip", "quick": "Short trip", "couple": "Short trip", "week": "One week", "weekly": "One week", "weeklong": "One week", "long": "Long trip", "extended": "Long trip", "month": "Long trip", "monthlong": "Long trip", "fortnight": "Long trip"}
# Words -> climate label (these map onto the "climate" fact we derived above).
CLIMATE_SYNONYMS = {"warm": "warm", "hot": "warm", "tropical": "warm", "sunny": "warm", "sun": "warm", "sunshine": "warm", "summer": "warm", "heat": "warm", "scorching": "warm", "boiling": "warm", "mild": "mild", "temperate": "mild", "spring": "mild", "autumn": "mild", "pleasant": "mild", "cold": "cold", "cool": "cold", "chilly": "cold", "snowy": "cold", "snow": "cold", "winter": "cold", "freezing": "cold", "frosty": "cold", "icy": "cold"}
# Words -> one of the nine lifestyle dimensions (these become ranking weights).
LIFESTYLE_SYNONYMS = {"culture": "culture", "cultural": "culture", "history": "culture", "historical": "culture", "historic": "culture", "heritage": "culture", "museum": "culture", "art": "culture", "arts": "culture", "architecture": "culture", "temple": "culture", "ruins": "culture", "monument": "culture", "tradition": "culture", "traditional": "culture", "ancient": "culture", "gallery": "culture", "adventure": "adventure", "adventurous": "adventure", "hiking": "adventure", "hike": "adventure", "trekking": "adventure", "trek": "adventure", "climbing": "adventure", "diving": "adventure", "surfing": "adventure", "surf": "adventure", "safari": "adventure", "adrenaline": "adventure", "outdoor": "adventure", "outdoors": "adventure", "extreme": "adventure", "rafting": "adventure", "exploring": "adventure", "nature": "nature", "natural": "nature", "mountain": "nature", "mountains": "nature", "forest": "nature", "scenery": "nature", "scenic": "nature", "landscape": "nature", "wildlife": "nature", "greenery": "nature", "lake": "nature", "river": "nature", "waterfall": "nature", "park": "nature", "parks": "nature", "countryside": "nature", "hills": "nature", "beach": "beaches", "beaches": "beaches", "coast": "beaches", "coastal": "beaches", "sea": "beaches", "seaside": "beaches", "ocean": "beaches", "shore": "beaches", "sand": "beaches", "sandy": "beaches", "island": "beaches", "islands": "beaches", "swimming": "beaches", "snorkeling": "beaches", "sunbathing": "beaches", "nightlife": "nightlife", "party": "nightlife", "parties": "nightlife", "partying": "nightlife", "club": "nightlife", "clubs": "nightlife", "clubbing": "nightlife", "bar": "nightlife", "bars": "nightlife", "pub": "nightlife", "pubs": "nightlife", "dancing": "nightlife", "drinks": "nightlife", "festival": "nightlife", "cuisine": "cuisine", "food": "cuisine", "foodie": "cuisine", "gastronomy": "cuisine", "culinary": "cuisine", "eating": "cuisine", "restaurant": "cuisine", "restaurants": "cuisine", "dishes": "cuisine", "streetfood": "cuisine", "wine": "cuisine", "dining": "cuisine", "tasting": "cuisine", "wellness": "wellness", "spa": "wellness", "relax": "wellness", "relaxation": "wellness", "relaxing": "wellness", "calm": "wellness", "yoga": "wellness", "retreat": "wellness", "massage": "wellness", "unwind": "wellness", "tranquil": "wellness", "healing": "wellness", "meditation": "wellness", "serene": "wellness", "soothing": "wellness", "urban": "urban", "city": "urban", "metropolitan": "urban", "metropolis": "urban", "modern": "urban", "shopping": "urban", "skyline": "urban", "skyscrapers": "urban", "downtown": "urban", "cosmopolitan": "urban", "business": "urban", "seclusion": "seclusion", "secluded": "seclusion", "quiet": "seclusion", "peaceful": "seclusion", "isolated": "seclusion", "remote": "seclusion", "solitude": "seclusion", "privacy": "seclusion", "private": "seclusion", "offgrid": "seclusion", "hidden": "seclusion", "untouched": "seclusion", "tranquility": "seclusion", "alone": "seclusion", "escape": "seclusion"}

# Multi-word phrase aliases used by nlp.extract() after normalization. They are
# kept here with the rest of the domain vocabulary so language knowledge remains
# separate from the chatbot and inference logic.
CONTINENT_SYNONYMS.update({
    "latinamerica": "south_america",
    "centralamerica": "north_america",
    "northamerica": "north_america",
    "southamerica": "south_america",
    "middleeast": "middle_east",
    "newzealand": "oceania",
    "unitedstates": "north_america",
})
