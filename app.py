# app.py
# ----------------------------------------------------------------------------
# STREAMLIT GUI for the Travel Destination chatbot.
# This is a thin presentation layer: it does NOT contain any travel logic of its
# own. It simply wraps the SAME TravelChatbot class that the command-line
# main.py uses, so the GUI and the CLI behave identically. All the real work
# (NLP, fuzzy logic, inference, conversation state) still lives in the other
# modules - we only add a chat-style web front end on top.
#
# Run it with:   streamlit run app.py
# (Install Streamlit first:  pip install streamlit)
# ----------------------------------------------------------------------------
import streamlit as st
import knowledge_base as kb
from chatbot import TravelChatbot

# Basic page setup (title shown in the browser tab + a globe icon).
st.set_page_config(page_title="Travel Destination Assistant", page_icon="🌍")


@st.cache_resource
def load_destinations():
    # Load the 560-city knowledge base ONCE per server process and reuse it.
    # @st.cache_resource means Streamlit keeps this object in memory across
    # reruns/users instead of re-reading the CSV on every interaction, which
    # would be wasteful (the data never changes while the app runs).
    return kb.load_destinations()


def init_session():
    # Set up the per-user session the first time they open the page. We store
    # BOTH the chatbot and the chat transcript in st.session_state so they
    # survive across reruns. (Streamlit re-executes this whole script top to
    # bottom on every interaction, so anything we want to keep MUST live in
    # session_state - otherwise the bot would be rebuilt and forget everything,
    # including the rejected-cities memory, on every message.)
    if "bot" not in st.session_state:
        st.session_state.bot = TravelChatbot(load_destinations())
        # history is a list of (role, message) pairs, oldest first.
        st.session_state.history = []
        # Show the bot's greeting automatically on first load, exactly like the
        # CLI does before the loop starts.
        greeting = st.session_state.bot.greeting()
        st.session_state.history.append(("bot", greeting))


def render_message(role, message):
    # Draw one chat bubble. Requirement: user on the RIGHT, bot on the LEFT.
    # Streamlit stacks vertically by default, so we fake left/right alignment by
    # placing each bubble in a wide/narrow column pair. The bot's multi-line
    # replies use single "\n" newlines; in Markdown a single newline collapses
    # to a space, so we convert each "\n" into "  \n" (two trailing spaces),
    # which Markdown renders as a real line break - keeping the picks readable.
    pretty = message.replace("\n", "  \n")
    if role == "user":
        # Narrow empty column on the left pushes the bubble to the right.
        _, right = st.columns([3, 7])
        with right:
            with st.chat_message("user"):
                st.markdown(pretty)
    else:
        # Bot bubble sits in the left column.
        left, _ = st.columns([7, 3])
        with left:
            with st.chat_message("assistant"):
                st.markdown(pretty)


# --- Build the page on every rerun ------------------------------------------
init_session()
bot = st.session_state.bot

st.title("🌍 Travel Destination Assistant")
st.caption(
    "Describe your ideal trip in plain English. Commands still work: "
    "**go** (see picks), **why** (reasoning), **restart**, **help**, **exit** - "
    "and after you get picks you can say things like *not Paris* to drop a city."
)

# Replay the whole conversation so far, in order.
for role, message in st.session_state.history:
    render_message(role, message)

if bot.finished:
    # The user typed 'exit' (or similar): lock the input and say goodbye. The
    # bot's own farewell is already the last item in history above; here we add
    # a clear, disabled state plus a hint on how to start fresh.
    st.chat_input("The conversation has ended.", disabled=True)
    st.success("Safe travels! 🌴 Refresh the page to start a new conversation.")
else:
    # st.chat_input renders a text box pinned to the bottom of the page with its
    # own built-in send action (Enter / the send arrow), so we satisfy the
    # "input + send, no st.form" requirement without managing a button manually.
    user_text = st.chat_input("Tell me about your ideal trip...")
    if user_text:
        # Record what the user said, get the bot's reply, record that too, then
        # rerun so both new bubbles appear immediately. We reuse the SAME bot
        # instance from session_state - it is never re-created per message.
        st.session_state.history.append(("user", user_text))
        reply = bot.respond(user_text)
        st.session_state.history.append(("bot", reply))
        st.rerun()
