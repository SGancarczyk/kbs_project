# main.py
# ----------------------------------------------------------------------------
# ENTRY POINT. Run the whole system with:  python3 main.py
# It loads the Knowledge Base, creates the chatbot, prints the greeting, then
# loops: read a line the user types -> ask the bot to respond -> print the
# reply, until the user exits. This file is intentionally tiny: all the real
# work lives in the other modules (knowledge base, nlp, fuzzy, rules,
# inference, chatbot), which keeps responsibilities cleanly separated.
# ----------------------------------------------------------------------------
import knowledge_base as kb
from chatbot import TravelChatbot, render_response_text
def run():
    # Load all destinations once at start-up.
    destinations = kb.load_destinations()
    bot = TravelChatbot(destinations)
    # Print the opening message (greeting + first question).
    print(bot.greeting())
    # The conversation loop.
    while not bot.finished:
        try:
            # Read one line from the user. The "> " is just a prompt symbol.
            user_text = input("> ")
        except (EOFError, KeyboardInterrupt):
            # If input ends (Ctrl-D) or is interrupted (Ctrl-C), exit cleanly.
            print("\nGoodbye!")
            break
        # Ask the bot for its reply and print it. A successful recommendation
        # comes back as a structured dict, so we flatten it to text for the
        # terminal; every other reply is already a plain string.
        print(render_response_text(bot.respond(user_text)))
# Standard Python idiom: only run when this file is executed directly, not when
# it is imported by another module (e.g. by the tests).
if __name__ == "__main__":
    run()
