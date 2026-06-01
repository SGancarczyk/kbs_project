# knowledge_base.py
# ----------------------------------------------------------------------------
# THE KNOWLEDGE BASE of our Knowledge-Based System (KBS).
# Stores: (1) the travel city table loaded from CSV, (2) domain vocabulary
# dictionaries mapping human words to dataset values, (3) a new SEASON/MONTH
# vocabulary so the bot can match monthly temperature instead of yearly average.
# ----------------------------------------------------------------------------
import csv
import json
import os

LIFESTYLE_DIMENSIONS = ["culture", "adventure", "nature", "beaches", "nightlife", "cuisine", "wellness", "urban", "seclusion"]
REGIONS = ["europe", "asia", "africa", "north_america", "south_america", "oceania", "middle_east"]
BUDGET_LEVELS = ["Budget", "Mid-range", "Luxury"]
DURATIONS = ["Day trip", "Weekend", "Short trip", "One week", "Long trip"]
CLIMATES = ["cold", "mild", "warm"]

# Month numbers for each season / month name, used by the season slot.
SEASON_MONTHS = {
    "winter":    [12, 1, 2],
    "spring":    [3, 4, 5],
    "summer":    [6, 7, 8],
    "autumn":    [9, 10, 11],
    "fall":      [9, 10, 11],
}
MONTH_NUMBER = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def yearly_avg_temp(avg_temp_monthly):
    monthly_avgs = [month["avg"] for month in avg_temp_monthly.values()]
    return sum(monthly_avgs) / len(monthly_avgs)

def climate_label(yearly):
    if yearly < 15:
        return "cold"
    if yearly < 22:
        return "mild"
    return "warm"

def seasonal_avg_temp(avg_temp_monthly, months):
    """Return the average temperature across the given list of month numbers."""
    avgs = []
    for m in months:
        key = str(m)
        if key in avg_temp_monthly:
            avgs.append(avg_temp_monthly[key]["avg"])
    if not avgs:
        return None
    return sum(avgs) / len(avgs)

def load_destinations(csv_path=None):
    if csv_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, "Worldwide Travel Cities Dataset (Ratings and Climate).csv")
    destinations = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["avg_temp_monthly"] = json.loads(row["avg_temp_monthly"])
            row["ideal_durations"] = json.loads(row["ideal_durations"])
            for dim in LIFESTYLE_DIMENSIONS:
                row[dim] = int(row[dim])
            yearly = yearly_avg_temp(row["avg_temp_monthly"])
            row["avg_temp_yearly"] = round(yearly, 1)
            row["climate"] = climate_label(yearly)
            destinations.append(row)
    return destinations

# ----------------------------------------------------------------------------
# DOMAIN VOCABULARY
# ----------------------------------------------------------------------------

CONTINENT_SYNONYMS = {
    # Core continents / regions
    "europe": "europe", "european": "europe",
    # Western Europe
    "spain": "europe", "spanish": "europe", "madrid": "europe", "barcelona": "europe",
    "france": "europe", "french": "europe", "paris": "europe",
    "italy": "europe", "italian": "europe", "rome": "europe", "milan": "europe", "venice": "europe", "florence": "europe",
    "greece": "europe", "greek": "europe", "athens": "europe",
    "portugal": "europe", "portuguese": "europe", "lisbon": "europe",
    "germany": "europe", "german": "europe", "berlin": "europe", "munich": "europe",
    "netherlands": "europe", "dutch": "europe", "amsterdam": "europe",
    "switzerland": "europe", "swiss": "europe", "zurich": "europe",
    "austria": "europe", "vienna": "europe",
    "belgium": "europe", "belgian": "europe", "brussels": "europe",
    "ireland": "europe", "irish": "europe", "dublin": "europe",
    "scotland": "europe", "scottish": "europe", "edinburgh": "europe",
    "england": "europe", "english": "europe", "london": "europe",
    "britain": "europe", "british": "europe", "uk": "europe",
    "wales": "europe", "welsh": "europe",
    # Northern Europe
    "sweden": "europe", "swedish": "europe", "stockholm": "europe",
    "norway": "europe", "norwegian": "europe", "oslo": "europe",
    "denmark": "europe", "danish": "europe", "copenhagen": "europe",
    "finland": "europe", "finnish": "europe", "helsinki": "europe",
    "scandinavia": "europe", "scandinavian": "europe",
    "nordic": "europe", "iceland": "europe", "icelandic": "europe", "reykjavik": "europe",
    # Eastern / Central Europe
    "poland": "europe", "polish": "europe", "warsaw": "europe", "krakow": "europe",
    "czech": "europe", "prague": "europe",
    "hungary": "europe", "budapest": "europe",
    "romania": "europe", "bucharest": "europe",
    "bulgaria": "europe", "sofia": "europe",
    "ukraine": "europe", "kyiv": "europe",
    "balkans": "europe", "balkan": "europe",
    "serbia": "europe", "belgrade": "europe",
    "croatia": "europe", "zagreb": "europe", "dubrovnik": "europe", "split": "europe",
    "slovenia": "europe", "ljubljana": "europe",
    "slovakia": "europe", "bratislava": "europe",
    "albania": "europe",
    "montenegro": "europe",
    "bosnia": "europe",
    "macedonia": "europe",
    "moldova": "europe",
    "latvia": "europe", "riga": "europe",
    "lithuania": "europe", "vilnius": "europe",
    "estonia": "europe", "tallinn": "europe",
    "malta": "europe", "valletta": "europe",
    "luxembourg": "europe",
    "monaco": "europe",
    # Asia
    "asia": "asia", "asian": "asia",
    "japan": "asia", "japanese": "asia", "tokyo": "asia", "kyoto": "asia", "osaka": "asia",
    "china": "asia", "chinese": "asia", "beijing": "asia", "shanghai": "asia",
    "thailand": "asia", "thai": "asia", "bangkok": "asia", "phuket": "asia", "chiangmai": "asia",
    "india": "asia", "indian": "asia", "delhi": "asia", "mumbai": "asia", "goa": "asia", "jaipur": "asia",
    "vietnam": "asia", "vietnamese": "asia", "hanoi": "asia", "hochiminh": "asia", "hoi an": "asia",
    "indonesia": "asia", "indonesian": "asia", "bali": "asia", "jakarta": "asia", "lombok": "asia",
    "korea": "asia", "korean": "asia", "seoul": "asia", "busan": "asia",
    "malaysia": "asia", "kuala lumpur": "asia", "penang": "asia",
    "singapore": "asia",
    "philippines": "asia", "philippine": "asia", "manila": "asia", "cebu": "asia", "palawan": "asia",
    "cambodia": "asia", "cambodian": "asia", "phnom penh": "asia", "siem reap": "asia",
    "nepal": "asia", "nepali": "asia", "kathmandu": "asia",
    "taiwan": "asia", "taipei": "asia",
    "myanmar": "asia", "burma": "asia", "burmese": "asia", "yangon": "asia",
    "laos": "asia", "lao": "asia", "vientiane": "asia",
    "sri lanka": "asia", "colombo": "asia",
    "bangladesh": "asia",
    "pakistan": "asia", "karachi": "asia",
    "mongolia": "asia", "ulaanbaatar": "asia",
    "bhutan": "asia",
    "maldives": "asia",
    "east asia": "asia", "southeast asia": "asia", "south asia": "asia",
    "southeastasia": "asia", "eastasia": "asia",
    # Africa
    "africa": "africa", "african": "africa",
    "egypt": "africa", "egyptian": "africa", "cairo": "africa",
    "morocco": "africa", "moroccan": "africa", "marrakech": "africa", "casablanca": "africa", "fez": "africa",
    "kenya": "africa", "kenyan": "africa", "nairobi": "africa",
    "tanzania": "africa", "tanzanian": "africa", "dar es salaam": "africa", "zanzibar": "africa",
    "nigeria": "africa", "nigerian": "africa", "lagos": "africa",
    "ethiopia": "africa", "ethiopian": "africa", "addis ababa": "africa",
    "ghana": "africa", "ghanaian": "africa", "accra": "africa",
    "tunisia": "africa", "tunisian": "africa", "tunis": "africa",
    "namibia": "africa", "windhoek": "africa",
    "senegal": "africa", "senegalese": "africa", "dakar": "africa",
    "mauritius": "africa",
    "south africa": "africa", "cape town": "africa", "johannesburg": "africa",
    "southafrica": "africa",
    "rwanda": "africa", "kigali": "africa",
    "uganda": "africa", "kampala": "africa",
    "mozambique": "africa", "maputo": "africa",
    "zambia": "africa", "lusaka": "africa",
    "zimbabwe": "africa", "harare": "africa",
    "botswana": "africa", "gaborone": "africa",
    "madagascar": "africa",
    "seychelles": "africa",
    "cameroon": "africa",
    "ivory coast": "africa", "abidjan": "africa",
    "north africa": "africa", "sub-saharan": "africa",
    # North America
    "america": "north_america", "american": "north_america",
    "usa": "north_america", "unitedstates": "north_america", "us": "north_america",
    "new york": "north_america", "newyork": "north_america", "nyc": "north_america",
    "los angeles": "north_america", "la": "north_america",
    "chicago": "north_america", "miami": "north_america",
    "san francisco": "north_america", "sanfrancisco": "north_america",
    "boston": "north_america", "seattle": "north_america", "portland": "north_america",
    "new orleans": "north_america", "las vegas": "north_america",
    "canada": "north_america", "canadian": "north_america", "toronto": "north_america",
    "vancouver": "north_america", "montreal": "north_america",
    "mexico": "north_america", "mexican": "north_america", "mexico city": "north_america",
    "cancun": "north_america", "guadalajara": "north_america",
    "north": "north_america", "northamerica": "north_america",
    "cuba": "north_america", "havana": "north_america",
    "caribbean": "north_america",
    "costa rica": "north_america", "panama": "north_america",
    "guatemala": "north_america", "honduras": "north_america",
    "central america": "north_america", "centralamerica": "north_america",
    # South America
    "south": "south_america", "southamerica": "south_america",
    "latin": "south_america", "latinamerica": "south_america", "latin america": "south_america",
    "brazil": "south_america", "brazilian": "south_america", "rio": "south_america",
    "sao paulo": "south_america", "rio de janeiro": "south_america",
    "argentina": "south_america", "argentine": "south_america", "buenos aires": "south_america",
    "peru": "south_america", "peruvian": "south_america", "lima": "south_america", "cusco": "south_america",
    "chile": "south_america", "chilean": "south_america", "santiago": "south_america",
    "colombia": "south_america", "colombian": "south_america", "bogota": "south_america", "medellin": "south_america",
    "bolivia": "south_america", "la paz": "south_america",
    "ecuador": "south_america", "quito": "south_america",
    "venezuela": "south_america", "caracas": "south_america",
    "paraguay": "south_america", "uruguay": "south_america", "montevideo": "south_america",
    "suriname": "south_america", "guyana": "south_america",
    # Oceania
    "oceania": "oceania",
    "australia": "oceania", "australian": "oceania",
    "sydney": "oceania", "melbourne": "oceania", "brisbane": "oceania", "perth": "oceania",
    "pacific": "oceania",
    "new zealand": "oceania", "newzealand": "oceania", "auckland": "oceania", "wellington": "oceania",
    "fiji": "oceania", "fijian": "oceania",
    "papua new guinea": "oceania",
    "samoa": "oceania", "tonga": "oceania", "vanuatu": "oceania",
    # Middle East
    "middleeast": "middle_east", "middle east": "middle_east",
    "arabian": "middle_east", "arabic": "middle_east",
    "gulf": "middle_east",
    "dubai": "middle_east", "uae": "middle_east", "emirates": "middle_east",
    "qatar": "middle_east", "doha": "middle_east",
    "jordan": "middle_east", "amman": "middle_east", "petra": "middle_east",
    "oman": "middle_east", "muscat": "middle_east",
    "lebanon": "middle_east", "beirut": "middle_east",
    "israel": "middle_east", "tel aviv": "middle_east", "jerusalem": "middle_east",
    "saudi arabia": "middle_east", "riyadh": "middle_east",
    "bahrain": "middle_east", "kuwait": "middle_east",
    "iran": "middle_east", "tehran": "middle_east",
    "iraq": "middle_east", "baghdad": "middle_east",
    "turkey": "middle_east", "turkish": "middle_east", "istanbul": "middle_east", "ankara": "middle_east",
}

BUDGET_SYNONYMS = {
    # Budget
    "cheap": "Budget", "cheaper": "Budget", "cheapest": "Budget",
    "affordable": "Budget", "afford": "Budget",
    "budget": "Budget", "budgetfriendly": "Budget", "budget-friendly": "Budget",
    "inexpensive": "Budget", "low": "Budget", "economical": "Budget", "economic": "Budget",
    "frugal": "Budget", "broke": "Budget", "tight": "Budget",
    "shoestring": "Budget", "backpacker": "Budget", "backpacking": "Budget",
    "bargain": "Budget", "penny": "Budget", "thrifty": "Budget", "thrift": "Budget",
    "value": "Budget", "wallet": "Budget", "saving": "Budget", "savings": "Budget",
    "cost-effective": "Budget", "costeffective": "Budget", "no frills": "Budget",
    "basic": "Budget", "simple": "Budget",
    # Mid-range
    "moderate": "Mid-range", "mid": "Mid-range", "midrange": "Mid-range", "mid-range": "Mid-range",
    "medium": "Mid-range", "average": "Mid-range", "normal": "Mid-range",
    "reasonable": "Mid-range", "standard": "Mid-range", "decent": "Mid-range",
    "comfortable": "Mid-range", "middleground": "Mid-range", "balanced": "Mid-range",
    "fair": "Mid-range", "regular": "Mid-range",
    # Luxury
    "luxury": "Luxury", "luxurious": "Luxury",
    "expensive": "Luxury", "premium": "Luxury",
    "high": "Luxury", "lavish": "Luxury", "fancy": "Luxury",
    "splurge": "Luxury", "upscale": "Luxury", "highend": "Luxury", "high-end": "Luxury",
    "fivestar": "Luxury", "five-star": "Luxury", "5star": "Luxury",
    "posh": "Luxury", "deluxe": "Luxury", "exclusive": "Luxury", "opulent": "Luxury",
    "rich": "Luxury", "extravagant": "Luxury", "indulgent": "Luxury",
    "top-tier": "Luxury", "toptier": "Luxury", "first-class": "Luxury", "firstclass": "Luxury",
    "five stars": "Luxury",
}

DURATION_SYNONYMS = {
    # Day trip
    "day": "Day trip", "daytrip": "Day trip", "day-trip": "Day trip",
    "daylong": "Day trip", "one-day": "Day trip", "1 day": "Day trip",
    # Weekend
    "overnight": "Weekend", "weekend": "Weekend", "getaway": "Weekend",
    "two days": "Weekend", "2 days": "Weekend", "two-day": "Weekend",
    # Short trip
    "short": "Short trip", "few": "Short trip", "quick": "Short trip",
    "couple": "Short trip", "mini": "Short trip", "brief": "Short trip",
    "few days": "Short trip", "3 days": "Short trip", "4 days": "Short trip", "5 days": "Short trip",
    # One week
    "week": "One week", "weekly": "One week", "weeklong": "One week", "week-long": "One week",
    "seven days": "One week", "7 days": "One week", "1 week": "One week",
    # Long trip
    "long": "Long trip", "extended": "Long trip", "month": "Long trip",
    "monthlong": "Long trip", "fortnight": "Long trip", "two weeks": "Long trip",
    "2 weeks": "Long trip", "3 weeks": "Long trip", "several weeks": "Long trip",
    "gap year": "Long trip", "sabbatical": "Long trip",
}

CLIMATE_SYNONYMS = {
    # Warm
    "warm": "warm", "hot": "warm", "tropical": "warm",
    "sunny": "warm", "sun": "warm", "sunshine": "warm", "sunny day": "warm",
    "summer": "warm", "heat": "warm", "scorching": "warm", "boiling": "warm",
    "humid": "warm", "sweltering": "warm", "blazing": "warm",
    "beach weather": "warm", "swimwear": "warm", "tanning": "warm",
    # Mild
    "mild": "mild", "temperate": "mild", "spring": "mild",
    "autumn": "mild", "fall": "mild",
    "pleasant": "mild", "moderate temperature": "mild", "comfortable weather": "mild",
    "not too hot": "mild", "not too cold": "mild", "in between": "mild",
    # Cold
    "cold": "cold", "cool": "cold", "chilly": "cold",
    "snowy": "cold", "snow": "cold", "winter": "cold",
    "freezing": "cold", "frosty": "cold", "icy": "cold",
    "crisp": "cold", "brisk": "cold", "arctic": "cold",
}

LIFESTYLE_SYNONYMS = {
    # Culture
    "culture": "culture", "cultural": "culture",
    "history": "culture", "historical": "culture", "historic": "culture",
    "heritage": "culture", "museum": "culture", "museums": "culture",
    "art": "culture", "arts": "culture", "artwork": "culture",
    "architecture": "culture", "temple": "culture", "temples": "culture",
    "ruins": "culture", "monument": "culture", "monuments": "culture",
    "tradition": "culture", "traditional": "culture",
    "ancient": "culture", "gallery": "culture", "galleries": "culture",
    "castle": "culture", "castles": "culture", "palace": "culture", "palaces": "culture",
    "cathedral": "culture", "cathedrals": "culture", "churches": "culture", "church": "culture",
    "local culture": "culture", "theater": "culture", "theatre": "culture",
    "opera": "culture", "classical": "culture", "civilisation": "culture", "civilization": "culture",
    # Adventure
    "adventure": "adventure", "adventurous": "adventure",
    "hiking": "adventure", "hike": "adventure", "hikes": "adventure",
    "trekking": "adventure", "trek": "adventure", "treks": "adventure",
    "climbing": "adventure", "rock climbing": "adventure",
    "diving": "adventure", "scuba": "adventure", "snorkeling": "adventure",
    "surfing": "adventure", "surf": "adventure",
    "safari": "adventure",
    "adrenaline": "adventure", "outdoor": "adventure", "outdoors": "adventure",
    "extreme": "adventure", "rafting": "adventure", "kayaking": "adventure",
    "exploring": "adventure", "explore": "adventure",
    "cycling": "adventure", "biking": "adventure", "mountain biking": "adventure",
    "zip line": "adventure", "zipline": "adventure", "bungee": "adventure",
    "paragliding": "adventure", "skydiving": "adventure",
    "skiing": "adventure", "ski": "adventure", "snowboarding": "adventure",
    "camping": "adventure", "expedition": "adventure",
    "off-road": "adventure", "offroad": "adventure",
    # Nature
    "nature": "nature", "natural": "nature",
    "mountain": "nature", "mountains": "nature",
    "forest": "nature", "forests": "nature", "jungle": "nature",
    "scenery": "nature", "scenic": "nature",
    "landscape": "nature", "landscapes": "nature",
    "wildlife": "nature", "animals": "nature", "birds": "nature",
    "greenery": "nature", "green": "nature",
    "lake": "nature", "lakes": "nature",
    "river": "nature", "rivers": "nature",
    "waterfall": "nature", "waterfalls": "nature",
    "park": "nature", "parks": "nature", "national park": "nature",
    "countryside": "nature", "hills": "nature",
    "volcano": "nature", "volcanoes": "nature",
    "glacier": "nature", "canyon": "nature", "desert": "nature",
    "botanical": "nature", "flora": "nature", "fauna": "nature",
    # Beaches
    "beach": "beaches", "beaches": "beaches",
    "coast": "beaches", "coastal": "beaches",
    "sea": "beaches", "seaside": "beaches",
    "ocean": "beaches", "shore": "beaches",
    "sand": "beaches", "sandy": "beaches",
    "island": "beaches", "islands": "beaches",
    "swimming": "beaches", "sunbathing": "beaches",
    "snorkeling": "beaches", "diving": "beaches",
    "sailing": "beaches", "windsurfing": "beaches",
    "tropical beach": "beaches", "white sand": "beaches",
    "lagoon": "beaches", "reef": "beaches", "coral": "beaches",
    # Nightlife
    "nightlife": "nightlife",
    "party": "nightlife", "parties": "nightlife", "partying": "nightlife",
    "club": "nightlife", "clubs": "nightlife", "clubbing": "nightlife",
    "bar": "nightlife", "bars": "nightlife",
    "pub": "nightlife", "pubs": "nightlife",
    "dancing": "nightlife", "dance": "nightlife",
    "drinks": "nightlife", "cocktails": "nightlife",
    "festival": "nightlife", "festivals": "nightlife",
    "live music": "nightlife", "concert": "nightlife",
    "entertainment": "nightlife", "buzzing": "nightlife",
    "lively": "nightlife", "vibrant": "nightlife",
    # Cuisine
    "cuisine": "cuisine", "food": "cuisine", "foodie": "cuisine",
    "gastronomy": "cuisine", "culinary": "cuisine",
    "eating": "cuisine", "eat": "cuisine",
    "restaurant": "cuisine", "restaurants": "cuisine",
    "dishes": "cuisine", "dish": "cuisine",
    "streetfood": "cuisine", "street food": "cuisine",
    "wine": "cuisine", "beer": "cuisine",
    "dining": "cuisine", "dine": "cuisine",
    "tasting": "cuisine", "taste": "cuisine",
    "cooking": "cuisine", "local food": "cuisine",
    "brunch": "cuisine", "breakfast": "cuisine",
    "market": "cuisine", "food market": "cuisine",
    "vegan": "cuisine", "vegetarian": "cuisine",
    "seafood": "cuisine", "sushi": "cuisine", "pizza": "cuisine", "tacos": "cuisine",
    # Wellness
    "wellness": "wellness", "spa": "wellness",
    "relax": "wellness", "relaxation": "wellness", "relaxing": "wellness",
    "calm": "wellness", "yoga": "wellness",
    "retreat": "wellness", "massage": "wellness",
    "unwind": "wellness", "tranquil": "wellness",
    "healing": "wellness", "meditation": "wellness",
    "serene": "wellness", "soothing": "wellness",
    "rejuvenate": "wellness", "recharge": "wellness",
    "hot spring": "wellness", "hot springs": "wellness", "thermal": "wellness",
    "detox": "wellness", "mindfulness": "wellness",
    # Urban
    "urban": "urban", "city": "urban",
    "metropolitan": "urban", "metropolis": "urban",
    "modern": "urban", "shopping": "urban",
    "skyline": "urban", "skyscrapers": "urban",
    "downtown": "urban", "cosmopolitan": "urban",
    "business": "urban", "infrastructure": "urban",
    "hip": "urban", "trendy": "urban", "fashionable": "urban",
    "rooftop": "urban", "street art": "urban",
    "markets": "urban", "bazaar": "urban",
    # Seclusion
    "seclusion": "seclusion", "secluded": "seclusion",
    "quiet": "seclusion", "peaceful": "seclusion",
    "isolated": "seclusion", "remote": "seclusion",
    "solitude": "seclusion", "privacy": "seclusion", "private": "seclusion",
    "offgrid": "seclusion", "off-grid": "seclusion",
    "hidden": "seclusion", "untouched": "seclusion",
    "tranquility": "seclusion", "alone": "seclusion",
    "escape": "seclusion", "off the beaten path": "seclusion",
    "getaway": "seclusion", "undiscovered": "seclusion",
    "rural": "seclusion", "countryside": "seclusion",
    # Sports -> adventure
    "sports": "adventure", "sport": "adventure",
    "active": "adventure", "activities": "adventure", "activity": "adventure",
    "physical": "adventure",
}

# Season / month vocabulary  ->  canonical key used by chatbot + inference
SEASON_SYNONYMS = {
    # Seasons
    "winter": "winter", "wintry": "winter", "wintertime": "winter",
    "spring": "spring", "springtime": "spring",
    "summer": "summer", "summertime": "summer",
    "autumn": "autumn", "fall": "autumn",
    # Month names
    "january": "january", "jan": "january",
    "february": "february", "feb": "february",
    "march": "march", "mar": "march",
    "april": "april", "apr": "april",
    "may": "may",
    "june": "june", "jun": "june",
    "july": "july", "jul": "july",
    "august": "august", "aug": "august",
    "september": "september", "sep": "september", "sept": "september",
    "october": "october", "oct": "october",
    "november": "november", "nov": "november",
    "december": "december", "dec": "december",
    # Informal
    "xmas": "december", "christmas": "december",
    "new year": "january", "newyear": "january",
    "easter": "march",
    "holiday season": "december",
    "peak summer": "july",
}

def season_to_months(season_key):
    """Convert a season/month key to a list of month numbers."""
    if season_key in SEASON_MONTHS:
        return SEASON_MONTHS[season_key]
    if season_key in MONTH_NUMBER:
        return [MONTH_NUMBER[season_key]]
    return []

# Multi-word phrase aliases
CONTINENT_SYNONYMS.update({
    "latinamerica": "south_america",
    "centralamerica": "north_america",
    "northamerica": "north_america",
    "southamerica": "south_america",
    "middleeast": "middle_east",
    "newzealand": "oceania",
    "unitedstates": "north_america",
    "southeast asia": "asia",
    "east asia": "asia",
    "south asia": "asia",
    "north africa": "africa",
    "south africa": "africa",
    "central america": "north_america",
    "latin america": "south_america",
    "middle east": "middle_east",
    "new zealand": "oceania",
})