import pandas as pd
import streamlit as st
import knowledge_base as kb
from chatbot import TravelChatbot
from inference import explain_human, explain
st.set_page_config(page_title="Travel Destination Assistant",
                   page_icon="🌍", layout="wide")
_MONTH_NAME = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
               6: "June", 7: "July", 8: "August", 9: "September", 10: "October",
               11: "November", 12: "December"}
_EXAMPLE_PROMPTS = [
    "Cheap warm weekend in Europe",
    "Nature and adventure in South America",
    "Luxury cultural trip in Japan",
    "Beach trip but avoid nightlife",
    "I don't care about budget, culture matters most",
]
def _join(items):
    items = [str(i) for i in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return items[0] + " and " + items[1]
    return ", ".join(items[:-1]) + ", and " + items[-1]
def _pretty_region(region):
    return str(region).replace("_", " ").title()
def _fmt_months(months):
    return _join([_MONTH_NAME.get(m, str(m)) for m in months])
def _card_summary(card):
    dest = card["dest"]
    region = _pretty_region(card["region"])
    sentence = (f"**{card['match']}.** {dest['city']} is a "
                f"{card['budget'].lower()} destination in {region}, averaging "
                f"about {card['temp']}.")
    if card["strong_on"]:
        sentence += f" It looks strongest for {_join(card['strong_on'])}."
    return sentence
def _tradeoffs(card):
    dest = card["dest"]
    details = card["score"].get("details", {})
    out = []
    if details.get("climate", (0, 0))[1] <= -0.1:
        out.append("the weather isn't an ideal match for what you asked for")
    if details.get("budget", (0, 0))[1] <= -0.1:
        out.append(f"it's {dest['budget_level'].lower()}, not quite the budget you had in mind")
    if details.get("duration", (0, 0))[1] <= -0.1:
        out.append("it suits a slightly different trip length")
    if details.get("lifestyle", (0, 0))[1] <= -0.1:
        out.append("it's a little weak on your main interests")
    return out
def _avoidance_warnings(dest, query):
    warns = []
    for dim in query.get("lifestyle_exclude") or []:
        score = dest.get(dim, 0)
        if isinstance(score, (int, float)) and score >= 4:
            warns.append(f"You asked to avoid **{dim}**, but {dest['city']} scores "
                         f"fairly high on it ({score}/5) — treat that as a trade-off "
                         f"rather than a perfect match.")
    for level in query.get("budget_exclude") or []:
        if dest.get("budget_level") == level:
            warns.append(f"You wanted to steer clear of **{level.lower()}** pricing, "
                         f"yet {dest['city']} is {level.lower()}.")
    for term in query.get("climate_exclude") or []:
        if dest.get("climate") == term:
            warns.append(f"You wanted to avoid **{term}** weather, and {dest['city']} "
                         f"reads as {term} overall.")
    return warns
def _city_profile_sentences(dest):
    region = _pretty_region(dest["region"])
    temp = round(dest.get("avg_temp_yearly", 0))
    strong = [d for d in kb.LIFESTYLE_DIMENSIONS if dest.get(d, 0) >= 4]
    weak = [d for d in kb.LIFESTYLE_DIMENSIONS if dest.get(d, 0) <= 2]
    parts = [f"{dest['city']} is a {dest['budget_level'].lower()} destination in "
             f"{region} with an average yearly temperature of around {temp}°C."]
    if strong:
        parts.append(f"It is especially strong for {_join(strong)}, which makes it a "
                     f"good choice for travellers drawn to those things.")
    if dest.get("ideal_durations"):
        parts.append(f"It tends to suit {_join([d.lower() for d in dest['ideal_durations']])}.")
    if weak:
        parts.append(f"It is less focused on {_join(weak)}, so it may suit a calmer or "
                     f"different kind of trip if those matter to you.")
    try:
        comfy = kb.comfortable_months(dest["avg_temp_monthly"])
        if comfy:
            parts.append(f"The most pleasant warm-weather stretch is around "
                         f"{kb.format_month_ranges(comfy)}.")
    except Exception:
        pass
    if kb.hemisphere(dest.get("latitude")) == "southern":
        parts.append("It sits in the southern hemisphere, so its summer falls around "
                     "December to February.")
    if dest.get("short_description"):
        parts.append(dest["short_description"])
    return " ".join(parts)
def _lifestyle_chart(dest):
    dims = kb.LIFESTYLE_DIMENSIONS
    df = pd.DataFrame({"score (1-5)": [dest.get(d, 0) for d in dims]},
                      index=[d.capitalize() for d in dims])
    st.bar_chart(df, height=240)
def _why_panel(card, query):
    plain_tab, adv_tab = st.tabs(["In plain English", "Advanced reasoning"])
    with plain_tab:
        for line in explain_human(card["dest"], card["score"], query):
            st.markdown("- " + line)
        warns = _avoidance_warnings(card["dest"], query)
        for w in warns:
            st.warning(w)
    with adv_tab:
        score = card["score"]
        st.caption("Internal expert-system reasoning (kept for transparency):")
        st.markdown("`" + explain(card["dest"], score, query) + "`")
        bits = []
        if score.get("lifestyle_fit") is not None:
            bits.append(f"weighted lifestyle strength (cosine × strength): "
                        f"{round(score['lifestyle_fit'], 3)}")
        bits.append(f"final combined confidence: {round(score.get('confidence', 0), 3)}")
        if score.get("hard_failures"):
            bits.append("hard-failure gate triggered on: " + ", ".join(score["hard_failures"]))
        for b in bits:
            st.markdown("- " + b)
@st.cache_resource
def load_destinations():
    return kb.load_destinations()
def init_session():
    if "bot" not in st.session_state:
        st.session_state.bot = TravelChatbot(load_destinations())
        st.session_state.history = []
        st.session_state.history.append(("bot", st.session_state.bot.greeting()))
def submit(text):
    bot = st.session_state.bot
    st.session_state.history.append(("user", text))
    st.session_state.history.append(("bot", bot.respond(text)))
def render_sidebar(bot):
    q = bot.query
    with st.sidebar:
        st.header("🧭 Your Trip Profile")
        st.caption("What I've understood so far.")
        def row(label, value):
            if value:
                st.markdown(f"**{label}:** {value}")
        wanted_places = (q.get("countries_include") or []) + \
            [_pretty_region(r) for r in (q.get("regions_include") or [])]
        avoided_places = (q.get("countries_exclude") or []) + \
            [_pretty_region(r) for r in (q.get("regions_exclude") or [])]
        row("Destinations", _join(wanted_places))
        row("Avoiding", _join(avoided_places))
        row("Budget", _join(q.get("budget") or []))
        row("Not budget", _join(q.get("budget_exclude") or []))
        row("Climate", _join(q.get("climate") or []))
        row("Not climate", _join(q.get("climate_exclude") or []))
        row("Trip length", _join(q.get("duration") or []))
        row("Travel months", _fmt_months(q.get("travel_months") or []))
        life = q.get("lifestyle") or {}
        if life:
            def tag(v):
                if v == -1:
                    return "choosing…"
                return {5: "top", 4: "high", 3: "medium", 2: "low", 1: "low"}.get(v, str(v))
            row("Interests", _join([f"{d} ({tag(w)})" for d, w in life.items()]))
        row("Avoiding features", _join(q.get("lifestyle_exclude") or []))
        row("Dropped cities", _join(sorted(bot.rejected_cities)))
        imp = q.get("importance") or {}
        if imp:
            labels = {"climate": "climate", "budget": "budget",
                      "duration": "trip length", "lifestyle": "interests"}
            row("Priorities", _join([f"{labels.get(c, c)} {'↑' if f > 1 else '↓'}"
                                     for c, f in imp.items()]))
        nothing = not any([wanted_places, avoided_places, q.get("budget"),
                           q.get("climate"), q.get("duration"), q.get("travel_months"),
                           life, q.get("lifestyle_exclude"), bot.rejected_cities, imp])
        if nothing:
            st.info("Tell me about your trip and this fills in.")
        st.divider()
        if st.button("🔄 Restart / clear preferences", use_container_width=True):
            submit("restart")
            st.rerun()
def render_recommendations(payload, idx, is_last):
    with st.chat_message("assistant"):
        if payload.get("lead"):
            st.markdown(payload["lead"])
        for advisory in payload.get("advisories", []):
            st.info(advisory)
        if payload.get("summary"):
            st.caption(payload["summary"])
        st.markdown("**" + payload["header"] + "**")
        query = payload.get("query", {})
        results = payload["results"]
        for card in results:
            with st.container(border=True):
                st.markdown(f"### #{card['rank']} · {card['city']}, {card['country']}")
                st.markdown(_card_summary(card))
                st.write(card["description"])
                tr = _tradeoffs(card)
                if tr:
                    st.markdown("_The main trade-off is that " + _join(tr) + "._")
                for w in _avoidance_warnings(card["dest"], query):
                    st.warning(w)
                with st.expander("Why this pick?"):
                    _why_panel(card, query)
                with st.expander(f"More about {card['city']} (profile + lifestyle chart)"):
                    st.write(_city_profile_sentences(card["dest"]))
                    _lifestyle_chart(card["dest"])
        if len(results) > 1:
            st.markdown("**At a glance**")
            rows = [{"#": c["rank"], "City": c["city"], "Country": c["country"],
                     "Match": c["match"], "Budget": c["budget"], "Temp": c["temp"],
                     "Strongest": ", ".join(c["strong_on"][:3]) or "—"}
                    for c in results]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if is_last:
            city1 = results[0]["city"]
            st.caption("Quick follow-ups (or just keep typing):")
            actions = [
                ("✨ More like #1", f"more like {city1}"),
                ("💸 Cheaper", f"more like {city1} but cheaper"),
                ("☀️ Warmer", f"more like {city1} but warmer"),
                ("🏛 More culture", f"more like {city1} but more culture"),
                ("🌙 Less nightlife", f"more like {city1} but without nightlife"),
                ("🗑 Remove #1", f"not {city1}"),
                ("❓ Why?", "why"),
                ("🔄 Restart", "restart"),
            ]
            cols = st.columns(4)
            for i, (label, msg) in enumerate(actions):
                if cols[i % 4].button(label, key=f"act_{idx}_{i}", use_container_width=True):
                    submit(msg)
                    st.rerun()
        st.caption(payload["footer"])
def render_message(role, message, idx, is_last):
    if isinstance(message, dict) and message.get("type") == "recommendations":
        render_recommendations(message, idx, is_last)
        return
    pretty = str(message).replace("\n", "  \n")
    if role == "user":
        _, right = st.columns([3, 7])
        with right:
            with st.chat_message("user"):
                st.markdown(pretty)
    else:
        left, _ = st.columns([7, 3])
        with left:
            with st.chat_message("assistant"):
                st.markdown(pretty)
init_session()
bot = st.session_state.bot
render_sidebar(bot)
st.title("🌍 Travel Destination Assistant")
st.caption("I recommend travel destinations through conversation — just describe "
           "your ideal trip and I'll reason out the best matches and explain why. "
           "Commands like **go**, **why**, **restart** and *not Paris* still work.")
if len(st.session_state.history) <= 1 and not bot.finished:
    st.markdown("**Try one of these to get started:**")
    cols = st.columns(len(_EXAMPLE_PROMPTS))
    for i, prompt in enumerate(_EXAMPLE_PROMPTS):
        if cols[i].button(prompt, key=f"ex_{i}", use_container_width=True):
            submit(prompt)
            st.rerun()
total = len(st.session_state.history)
for idx, (role, message) in enumerate(st.session_state.history):
    render_message(role, message, idx, is_last=(idx == total - 1))
if bot.finished:
    st.chat_input("The conversation has ended.", disabled=True)
    st.success("Safe travels! 🌴 Refresh the page to start a new conversation.")
else:
    user_text = st.chat_input("Tell me about your ideal trip...")
    if user_text:
        submit(user_text)
        st.rerun()
