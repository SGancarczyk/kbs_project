# knowledge_base.py
# ----------------------------------------------------------------------------
# THE KNOWLEDGE BASE of our Knowledge-Based System (KBS).
# Stores: (1) the travel city table loaded from CSV, (2) domain vocabulary
# dictionaries mapping human words to dataset values, (3) a SEASON/MONTH
# vocabulary so the bot can match monthly temperature instead of yearly average,
# (4) NEW: a COUNTRY_SYNONYMS vocabulary so the user can name a specific country
#     ("I want to go to Japan") and the bot pins results to that country exactly,
#     mirroring the season/month pattern: keyword -> canonical country name,
#     with a country_to_region() helper so inference can still do the region
#     filter it already knows how to do.
# ----------------------------------------------------------------------------
import csv
import json
import os
import pandas as pd    # replaces the built-in csv module for data loading

LIFESTYLE_DIMENSIONS = ["culture", "adventure", "nature", "beaches", "nightlife", "cuisine", "wellness", "urban", "seclusion"]

# LIFESTYLE_DESCRIPTIONS: short keyword-rich text for each lifestyle dimension.
# Used by the TF-IDF cosine similarity matcher in chatbot.py to detect which
# dimensions the user is talking about from a free-text sentence, without
# needing the user to use an exact vocabulary word. Each string covers the
# typical words a traveller would use when describing that interest.
LIFESTYLE_DESCRIPTIONS = {
    # Each description includes the dimension name itself so that users who
    # literally say "nightlife" or "culture" get a direct TF-IDF match,
    # alongside related activity words for natural-language descriptions.
    "culture":   "culture cultural history museums art architecture temples heritage monuments ancient traditions cathedral ruins gallery opera civilisation",
    "adventure": "adventure adventurous hiking trekking outdoor extreme climbing surfing safari diving rafting kayaking skiing bungee paragliding expedition",
    "nature":    "nature natural mountains forests wildlife landscape scenery national parks rivers waterfalls jungle volcano glacier canyon desert",
    "beaches":   "beaches beach coast ocean swimming sunbathing sand islands seaside snorkeling sailing reef lagoon tropical shore",
    "nightlife": "nightlife bars clubs dancing parties music entertainment drinks going out cocktails pub concert festival buzzing lively",
    "cuisine":   "cuisine food restaurants local dishes street food eating wine culinary gastronomy market brunch seafood tasting cooking",
    "wellness":  "wellness spa yoga relax relaxation meditation retreat massage calm peaceful rejuvenate hot springs mindfulness detox tranquil",
    "urban":     "urban city shopping skyline modern metropolitan cosmopolitan downtown trendy rooftop street art skyscrapers hip fashionable",
    "seclusion": "seclusion secluded quiet remote isolated peaceful private off beaten path rural hidden undiscovered solitude escape untouched",
}
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

# NEW: which region each dataset country belongs to.
# Used by country_to_region() so the inference engine can apply its existing
# region filter whenever the user names a specific country.
COUNTRY_REGION = {
    # Europe
    "Albania": "europe", "Austria": "europe", "Belarus": "europe",
    "Belgium": "europe", "Bosnia and Herzegovina": "europe", "Bulgaria": "europe",
    "Croatia": "europe", "Cyprus": "europe", "Czech Republic": "europe",
    "Denmark": "europe", "Estonia": "europe", "Finland": "europe",
    "France": "europe", "Germany": "europe", "Greece": "europe",
    "Greenland": "europe", "Hungary": "europe", "Iceland": "europe",
    "Ireland": "europe", "Italy": "europe", "Kosovo": "europe",
    "Latvia": "europe", "Lithuania": "europe", "Luxembourg": "europe",
    "Malta": "europe", "Moldova": "europe", "Monaco": "europe",
    "Montenegro": "europe", "Netherlands": "europe", "North Macedonia": "europe",
    "Norway": "europe", "Poland": "europe", "Portugal": "europe",
    "Romania": "europe", "Russia": "europe", "San Marino": "europe",
    "Serbia": "europe", "Slovakia": "europe", "Slovenia": "europe",
    "Spain": "europe", "Sweden": "europe", "Switzerland": "europe",
    "Ukraine": "europe", "United Kingdom": "europe",
    # Asia
    "Azerbaijan": "asia", "Bangladesh": "asia", "Bhutan": "asia",
    "Cambodia": "asia", "China": "asia", "Hong Kong": "asia",
    "India": "asia", "Indonesia": "asia", "Japan": "asia",
    "Kazakhstan": "asia", "Kyrgyzstan": "asia", "Laos": "asia",
    "Malaysia": "asia", "Maldives": "asia", "Mongolia": "asia",
    "Myanmar": "asia", "Nepal": "asia", "Pakistan": "asia",
    "Palau": "asia", "Philippines": "asia", "Singapore": "asia",
    "South Korea": "asia", "Sri Lanka": "asia", "Taiwan": "asia",
    "Tajikistan": "asia", "Thailand": "asia", "Turkmenistan": "asia",
    "Uzbekistan": "asia", "Vietnam": "asia",
    # Africa
    "Angola": "africa", "Benin": "africa", "Botswana": "africa",
    "Burundi": "africa", "Cameroon": "africa", "Cape Verde": "africa",
    "Comoros": "africa", "Egypt": "africa", "Equatorial Guinea": "africa",
    "Eswatini": "africa", "Ethiopia": "africa", "Gabon": "africa",
    "Gambia": "africa", "Ghana": "africa", "Ivory Coast": "africa",
    "Kenya": "africa", "Lesotho": "africa", "Madagascar": "africa",
    "Malawi": "africa", "Mauritius": "africa", "Morocco": "africa",
    "Mozambique": "africa", "Namibia": "africa", "Rwanda": "africa",
    "Réunion": "africa", "São Tomé and Príncipe": "africa",
    "Senegal": "africa", "Seychelles": "africa", "Sierra Leone": "africa",
    "South Africa": "africa", "Tanzania": "africa", "Togo": "africa",
    "Tunisia": "africa", "Uganda": "africa", "Zambia": "africa",
    "Zimbabwe": "africa", "Republic of the Congo": "africa",
    # North America
    "Barbados": "north_america", "Belize": "north_america",
    "Bermuda": "north_america", "Canada": "north_america",
    "Cook Islands": "north_america", "Costa Rica": "north_america",
    "Cuba": "north_america", "Dominica": "north_america",
    "Dominican Republic": "north_america", "El Salvador": "north_america",
    "Grenada": "north_america", "Guatemala": "north_america",
    "Jamaica": "north_america", "Mexico": "north_america",
    "Nicaragua": "north_america", "Panama": "north_america",
    "Puerto Rico": "north_america", "Quebec": "north_america",
    "Saint Kitts and Nevis": "north_america", "Saint Lucia": "north_america",
    "Trinidad and Tobago": "north_america", "Turks and Caicos": "north_america",
    "U.S. Virgin Islands": "north_america", "United States": "north_america",
    # South America
    "Argentina": "south_america", "Bolivia": "south_america",
    "Brazil": "south_america", "Chile": "south_america",
    "Colombia": "south_america", "Ecuador": "south_america",
    "French Guiana": "south_america", "Guyana": "south_america",
    "Paraguay": "south_america", "Peru": "south_america",
    "Suriname": "south_america", "Uruguay": "south_america",
    "Venezuela": "south_america",
    # Oceania
    "American Samoa": "oceania", "Australia": "oceania",
    "Fiji": "oceania", "New Caledonia": "oceania",
    "New Zealand": "oceania", "Papua New Guinea": "oceania",
    "Samoa": "oceania", "Solomon Islands": "oceania",
    "Tonga": "oceania", "Tuvalu": "oceania", "Vanuatu": "oceania",
    "French Polynesia": "oceania",
    # Middle East
    "Bahrain": "middle_east", "Georgia": "middle_east",
    "Iran": "middle_east", "Iraq": "middle_east",
    "Israel": "middle_east", "Jordan": "middle_east",
    "Kuwait": "middle_east", "Lebanon": "middle_east",
    "Oman": "middle_east", "Qatar": "middle_east",
    "Saudi Arabia": "middle_east", "Turkey": "middle_east",
    "United Arab Emirates": "middle_east",
}

# NEW: COUNTRY_SYNONYMS maps user keywords -> canonical country name (as it
# appears in the dataset's "country" column). Pattern mirrors SEASON_SYNONYMS:
# keyword -> canonical value. country_to_region() then resolves to region.
# No spellcheck on country names (same reason as season: short exact words
# are safer). Adjectives, demonyms, capitals, and common abbreviations are
# included so "japanese", "tokyo", "jp" all resolve to "Japan".
COUNTRY_SYNONYMS = {
    # Europe
    "albania": "Albania", "albanian": "Albania",
    "austria": "Austria", "austrian": "Austria", "vienna": "Austria",
    "belarus": "Belarus", "belarusian": "Belarus", "minsk": "Belarus",
    "belgium": "Belgium", "belgian": "Belgium", "brussels": "Belgium", "bruges": "Belgium",
    "bosnia": "Bosnia and Herzegovina", "bosnian": "Bosnia and Herzegovina", "sarajevo": "Bosnia and Herzegovina",
    "bulgaria": "Bulgaria", "bulgarian": "Bulgaria", "sofia": "Bulgaria",
    "croatia": "Croatia", "croatian": "Croatia", "zagreb": "Croatia", "dubrovnik": "Croatia", "split": "Croatia",
    "cyprus": "Cyprus", "cypriot": "Cyprus", "nicosia": "Cyprus",
    "czech republic": "Czech Republic", "czech": "Czech Republic", "czechia": "Czech Republic", "prague": "Czech Republic",
    "denmark": "Denmark", "danish": "Denmark", "copenhagen": "Denmark",
    "estonia": "Estonia", "estonian": "Estonia", "tallinn": "Estonia",
    "finland": "Finland", "finnish": "Finland", "helsinki": "Finland",
    "france": "France", "french": "France", "paris": "France", "lyon": "France", "nice, france": "France",
    "germany": "Germany", "german": "Germany", "berlin": "Germany", "munich": "Germany", "hamburg": "Germany",
    "greece": "Greece", "greek": "Greece", "athens": "Greece", "thessaloniki": "Greece",
    "greenland": "Greenland", "greenlandic": "Greenland", "nuuk": "Greenland",
    "hungary": "Hungary", "hungarian": "Hungary", "budapest": "Hungary",
    "iceland": "Iceland", "icelandic": "Iceland", "reykjavik": "Iceland",
    "ireland": "Ireland", "irish": "Ireland", "dublin": "Ireland",
    "italy": "Italy", "italian": "Italy", "rome": "Italy", "milan": "Italy", "venice": "Italy", "florence": "Italy", "naples": "Italy",
    "kosovo": "Kosovo", "pristina": "Kosovo",
    "latvia": "Latvia", "latvian": "Latvia", "riga": "Latvia",
    "lithuania": "Lithuania", "lithuanian": "Lithuania", "vilnius": "Lithuania",
    "luxembourg": "Luxembourg",
    "malta": "Malta", "maltese": "Malta", "valletta": "Malta",
    "moldova": "Moldova", "moldovan": "Moldova", "chisinau": "Moldova",
    "monaco": "Monaco", "monegasque": "Monaco",
    "montenegro": "Montenegro", "podgorica": "Montenegro",
    "netherlands": "Netherlands", "dutch": "Netherlands", "amsterdam": "Netherlands",
    "north macedonia": "North Macedonia", "macedonian": "North Macedonia", "skopje": "North Macedonia",
    "norway": "Norway", "norwegian": "Norway", "oslo": "Norway", "bergen": "Norway",
    "poland": "Poland", "polish": "Poland", "warsaw": "Poland", "krakow": "Poland",
    "portugal": "Portugal", "portuguese": "Portugal", "lisbon": "Portugal", "porto": "Portugal",
    "romania": "Romania", "romanian": "Romania", "bucharest": "Romania",
    "russia": "Russia", "russian": "Russia", "moscow": "Russia", "st petersburg": "Russia",
    "san marino": "San Marino",
    "serbia": "Serbia", "serbian": "Serbia", "belgrade": "Serbia",
    "slovakia": "Slovakia", "slovak": "Slovakia", "bratislava": "Slovakia",
    "slovenia": "Slovenia", "slovenian": "Slovenia", "ljubljana": "Slovenia",
    "spain": "Spain", "spanish": "Spain", "madrid": "Spain", "barcelona": "Spain", "seville": "Spain",
    "sweden": "Sweden", "swedish": "Sweden", "stockholm": "Sweden", "gothenburg": "Sweden",
    "switzerland": "Switzerland", "swiss": "Switzerland", "zurich": "Switzerland", "geneva": "Switzerland",
    "ukraine": "Ukraine", "ukrainian": "Ukraine", "kyiv": "Ukraine",
    "united kingdom": "United Kingdom", "british": "United Kingdom",
    "england": "United Kingdom", "london": "United Kingdom", "scotland": "United Kingdom",
    "edinburgh": "United Kingdom", "wales": "United Kingdom",
    # Asia
    "azerbaijan": "Azerbaijan", "azerbaijani": "Azerbaijan", "baku": "Azerbaijan",
    "bangladesh": "Bangladesh", "bangladeshi": "Bangladesh", "dhaka": "Bangladesh",
    "bhutan": "Bhutan", "bhutanese": "Bhutan", "thimphu": "Bhutan",
    "cambodia": "Cambodia", "cambodian": "Cambodia", "phnom penh": "Cambodia", "siem reap": "Cambodia",
    "china": "China", "chinese": "China", "beijing": "China", "shanghai": "China", "chengdu": "China",
    "hong kong": "Hong Kong",
    "india": "India", "indian": "India", "delhi": "India", "mumbai": "India", "goa": "India", "jaipur": "India",
    "indonesia": "Indonesia", "indonesian": "Indonesia", "bali": "Indonesia", "jakarta": "Indonesia",
    "japan": "Japan", "japanese": "Japan", "tokyo": "Japan", "kyoto": "Japan", "osaka": "Japan",
    "kazakhstan": "Kazakhstan", "kazakh": "Kazakhstan", "almaty": "Kazakhstan",
    "kyrgyzstan": "Kyrgyzstan", "kyrgyz": "Kyrgyzstan", "bishkek": "Kyrgyzstan",
    "laos": "Laos", "lao": "Laos", "vientiane": "Laos",
    "malaysia": "Malaysia", "malaysian": "Malaysia", "kuala lumpur": "Malaysia", "penang": "Malaysia",
    "maldives": "Maldives", "maldivian": "Maldives",
    "mongolia": "Mongolia", "mongolian": "Mongolia", "ulaanbaatar": "Mongolia",
    "myanmar": "Myanmar", "burmese": "Myanmar", "yangon": "Myanmar", "burma": "Myanmar",
    "nepal": "Nepal", "nepali": "Nepal", "kathmandu": "Nepal",
    "pakistan": "Pakistan", "pakistani": "Pakistan", "karachi": "Pakistan", "lahore": "Pakistan",
    "palau": "Palau",
    "philippines": "Philippines", "filipino": "Philippines", "manila": "Philippines", "cebu": "Philippines",
    "singapore": "Singapore", "singaporean": "Singapore",
    "south korea": "South Korea", "korean": "South Korea", "seoul": "South Korea", "busan": "South Korea",
    "sri lanka": "Sri Lanka", "sri lankan": "Sri Lanka", "colombo": "Sri Lanka",
    "taiwan": "Taiwan", "taiwanese": "Taiwan", "taipei": "Taiwan",
    "tajikistan": "Tajikistan", "tajik": "Tajikistan", "dushanbe": "Tajikistan",
    "thailand": "Thailand", "thai": "Thailand", "bangkok": "Thailand", "phuket": "Thailand", "chiang mai": "Thailand",
    "turkmenistan": "Turkmenistan", "ashgabat": "Turkmenistan",
    "uzbekistan": "Uzbekistan", "uzbek": "Uzbekistan", "tashkent": "Uzbekistan", "samarkand": "Uzbekistan",
    "vietnam": "Vietnam", "vietnamese": "Vietnam", "hanoi": "Vietnam", "ho chi minh": "Vietnam", "hoi an": "Vietnam",
    # Africa
    "angola": "Angola", "angolan": "Angola", "luanda": "Angola",
    "benin": "Benin", "beninese": "Benin", "cotonou": "Benin",
    "botswana": "Botswana", "gaborone": "Botswana",
    "burundi": "Burundi", "bujumbura": "Burundi",
    "cameroon": "Cameroon", "cameroonian": "Cameroon", "yaounde": "Cameroon",
    "cape verde": "Cape Verde", "cabo verde": "Cape Verde",
    "comoros": "Comoros",
    "egypt": "Egypt", "egyptian": "Egypt", "cairo": "Egypt",
    "equatorial guinea": "Equatorial Guinea",
    "eswatini": "Eswatini", "swaziland": "Eswatini",
    "ethiopia": "Ethiopia", "ethiopian": "Ethiopia", "addis ababa": "Ethiopia",
    "gabon": "Gabon", "libreville": "Gabon",
    "gambia": "Gambia", "banjul": "Gambia",
    "ghana": "Ghana", "ghanaian": "Ghana", "accra": "Ghana",
    "ivory coast": "Ivory Coast", "cote d'ivoire": "Ivory Coast", "abidjan": "Ivory Coast",
    "kenya": "Kenya", "kenyan": "Kenya", "nairobi": "Kenya",
    "lesotho": "Lesotho", "maseru": "Lesotho",
    "madagascar": "Madagascar", "malagasy": "Madagascar", "antananarivo": "Madagascar",
    "malawi": "Malawi", "lilongwe": "Malawi",
    "mauritius": "Mauritius",
    "morocco": "Morocco", "moroccan": "Morocco", "marrakech": "Morocco", "casablanca": "Morocco", "fez": "Morocco",
    "mozambique": "Mozambique", "maputo": "Mozambique",
    "namibia": "Namibia", "windhoek": "Namibia",
    "rwanda": "Rwanda", "kigali": "Rwanda",
    "reunion": "Réunion",
    "sao tome": "São Tomé and Príncipe",
    "senegal": "Senegal", "senegalese": "Senegal", "dakar": "Senegal",
    "seychelles": "Seychelles",
    "sierra leone": "Sierra Leone", "freetown": "Sierra Leone",
    "south africa": "South Africa", "cape town": "South Africa", "johannesburg": "South Africa",
    "tanzania": "Tanzania", "tanzanian": "Tanzania", "dar es salaam": "Tanzania", "zanzibar": "Tanzania",
    "togo": "Togo", "lome": "Togo",
    "tunisia": "Tunisia", "tunisian": "Tunisia", "tunis": "Tunisia",
    "uganda": "Uganda", "kampala": "Uganda",
    "zambia": "Zambia", "lusaka": "Zambia",
    "zimbabwe": "Zimbabwe", "harare": "Zimbabwe",
    "republic of the congo": "Republic of the Congo", "brazzaville": "Republic of the Congo",
    # North America
    "barbados": "Barbados", "bridgetown": "Barbados",
    "belize": "Belize", "belmopan": "Belize",
    "bermuda": "Bermuda",
    "canada": "Canada", "canadian": "Canada", "toronto": "Canada", "vancouver": "Canada", "montreal": "Canada",
    "costa rica": "Costa Rica", "san jose": "Costa Rica",
    "cuba": "Cuba", "cuban": "Cuba", "havana": "Cuba",
    "dominica": "Dominica", "roseau": "Dominica",
    "dominican republic": "Dominican Republic", "santo domingo": "Dominican Republic",
    "el salvador": "El Salvador", "san salvador": "El Salvador",
    "grenada": "Grenada",
    "guatemala": "Guatemala", "guatemalan": "Guatemala", "guatemala city": "Guatemala",
    "jamaica": "Jamaica", "jamaican": "Jamaica", "kingston": "Jamaica",
    "mexico": "Mexico", "mexican": "Mexico", "mexico city": "Mexico", "cancun": "Mexico",
    "nicaragua": "Nicaragua", "managua": "Nicaragua",
    "panama": "Panama", "panamanian": "Panama", "panama city": "Panama",
    "puerto rico": "Puerto Rico", "san juan": "Puerto Rico",
    "quebec": "Quebec",
    "saint kitts": "Saint Kitts and Nevis", "st kitts": "Saint Kitts and Nevis",
    "saint lucia": "Saint Lucia", "st lucia": "Saint Lucia",
    "trinidad": "Trinidad and Tobago", "tobago": "Trinidad and Tobago", "port of spain": "Trinidad and Tobago",
    "turks and caicos": "Turks and Caicos",
    "us virgin islands": "U.S. Virgin Islands",
    "united states": "United States", "usa": "United States",
    "new york": "United States", "nyc": "United States", "los angeles": "United States",
    "chicago": "United States", "miami": "United States", "las vegas": "United States",
    # South America
    "argentina": "Argentina", "argentine": "Argentina", "buenos aires": "Argentina",
    "bolivia": "Bolivia", "la paz": "Bolivia",
    "brazil": "Brazil", "brazilian": "Brazil", "rio": "Brazil", "sao paulo": "Brazil",
    "chile": "Chile", "chilean": "Chile", "santiago": "Chile",
    "colombia": "Colombia", "colombian": "Colombia", "bogota": "Colombia", "medellin": "Colombia",
    "ecuador": "Ecuador", "quito": "Ecuador",
    "french guiana": "French Guiana",
    "guyana": "Guyana", "georgetown": "Guyana",
    "paraguay": "Paraguay", "asuncion": "Paraguay",
    "peru": "Peru", "peruvian": "Peru", "lima": "Peru", "cusco": "Peru",
    "suriname": "Suriname", "paramaribo": "Suriname",
    "uruguay": "Uruguay", "montevideo": "Uruguay",
    "venezuela": "Venezuela", "venezuelan": "Venezuela", "caracas": "Venezuela",
    # Oceania
    "american samoa": "American Samoa",
    "australia": "Australia", "australian": "Australia", "sydney": "Australia", "melbourne": "Australia",
    "fiji": "Fiji", "fijian": "Fiji", "suva": "Fiji",
    "new caledonia": "New Caledonia", "noumea": "New Caledonia",
    "new zealand": "New Zealand", "kiwi": "New Zealand", "auckland": "New Zealand", "wellington": "New Zealand",
    "papua new guinea": "Papua New Guinea", "port moresby": "Papua New Guinea",
    "samoa": "Samoa", "apia": "Samoa",
    "solomon islands": "Solomon Islands", "honiara": "Solomon Islands",
    "tonga": "Tonga", "nukualofa": "Tonga",
    "tuvalu": "Tuvalu", "funafuti": "Tuvalu",
    "vanuatu": "Vanuatu", "port vila": "Vanuatu",
    "french polynesia": "French Polynesia", "tahiti": "French Polynesia", "bora bora": "French Polynesia",
    # Middle East
    "bahrain": "Bahrain", "manama": "Bahrain",
    "georgia": "Georgia", "tbilisi": "Georgia",
    "iran": "Iran", "iranian": "Iran", "tehran": "Iran",
    "iraq": "Iraq", "iraqi": "Iraq", "baghdad": "Iraq",
    "israel": "Israel", "israeli": "Israel", "tel aviv": "Israel", "jerusalem": "Israel",
    "jordan": "Jordan", "jordanian": "Jordan", "amman": "Jordan", "petra": "Jordan",
    "kuwait": "Kuwait", "kuwait city": "Kuwait",
    "lebanon": "Lebanon", "lebanese": "Lebanon", "beirut": "Lebanon",
    "oman": "Oman", "omani": "Oman", "muscat": "Oman",
    "qatar": "Qatar", "qatari": "Qatar", "doha": "Qatar",
    "saudi arabia": "Saudi Arabia", "saudi": "Saudi Arabia", "riyadh": "Saudi Arabia",
    "turkey": "Turkey", "turkish": "Turkey", "istanbul": "Turkey", "ankara": "Turkey",
    "united arab emirates": "United Arab Emirates", "uae": "United Arab Emirates",
    "dubai": "United Arab Emirates", "abu dhabi": "United Arab Emirates",
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
    # REWRITE: use pandas.read_csv() instead of csv.DictReader + for-loop.
    if csv_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, "Worldwide Travel Cities Dataset (Ratings and Climate).csv")
    df = pd.read_csv(csv_path)
    df["avg_temp_monthly"] = df["avg_temp_monthly"].apply(json.loads)
    df["ideal_durations"]  = df["ideal_durations"].apply(json.loads)
    for dim in LIFESTYLE_DIMENSIONS:
        df[dim] = df[dim].astype(int)
    df["avg_temp_yearly"] = df["avg_temp_monthly"].apply(lambda m: round(yearly_avg_temp(m), 1))
    df["climate"]         = df["avg_temp_yearly"].apply(climate_label)
    return df.to_dict(orient="records")

def country_to_region(country_name):
    """Return the region for a canonical country name, or None if unknown."""
    return COUNTRY_REGION.get(country_name)

def season_to_months(season_key):
    """Convert a season/month key to a list of month numbers."""
    if season_key in SEASON_MONTHS:
        return SEASON_MONTHS[season_key]
    if season_key in MONTH_NUMBER:
        return [MONTH_NUMBER[season_key]]
    return []

# ----------------------------------------------------------------------------
# DOMAIN VOCABULARY
# ----------------------------------------------------------------------------

CONTINENT_SYNONYMS = {
    # Pure region/continent keywords only.
    # Country names, city names, and demonyms have been moved to COUNTRY_SYNONYMS
    # so that a token like 'japan' only ever matches one vocabulary.
    # Europe
    "balkan": "europe",
    "balkans": "europe",
    "britain": "europe",
    "english": "europe",
    "europe": "europe",
    "european": "europe",
    "macedonia": "europe",
    "nordic": "europe",
    "scandinavia": "europe",
    "scandinavian": "europe",
    "scottish": "europe",
    "uk": "europe",
    "welsh": "europe",
    # Asia
    "asia": "asia",
    "asian": "asia",
    "chiangmai": "asia",
    "east asia": "asia",
    "eastasia": "asia",
    "hochiminh": "asia",
    "korea": "asia",
    "lombok": "asia",
    "palawan": "asia",
    "philippine": "asia",
    "south asia": "asia",
    "southeast asia": "asia",
    "southeastasia": "asia",
    # Africa
    "africa": "africa",
    "african": "africa",
    "lagos": "africa",
    "nigeria": "africa",
    "nigerian": "africa",
    "north africa": "africa",
    "southafrica": "africa",
    "sub-saharan": "africa",
    # North America
    "america": "north_america",
    "american": "north_america",
    "boston": "north_america",
    "caribbean": "north_america",
    "central america": "north_america",
    "centralamerica": "north_america",
    "guadalajara": "north_america",
    "honduras": "north_america",
    "la": "north_america",
    "new orleans": "north_america",
    "newyork": "north_america",
    "north": "north_america",
    "northamerica": "north_america",
    "portland": "north_america",
    "san francisco": "north_america",
    "sanfrancisco": "north_america",
    "seattle": "north_america",
    "unitedstates": "north_america",
    "us": "north_america",
    # South America
    "latin": "south_america",
    "latin america": "south_america",
    "latinamerica": "south_america",
    "rio de janeiro": "south_america",
    "south": "south_america",
    "southamerica": "south_america",
    # Oceania
    "brisbane": "oceania",
    "newzealand": "oceania",
    "oceania": "oceania",
    "pacific": "oceania",
    "perth": "oceania",
    # Middle East
    "arabian": "middle_east",
    "arabic": "middle_east",
    "emirates": "middle_east",
    "gulf": "middle_east",
    "middle east": "middle_east",
    "middleeast": "middle_east",
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