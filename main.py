import knowledge_base as kb
from chatbot import TravelChatbot, render_response_text
def run():
    destinations = kb.load_destinations()
    bot = TravelChatbot(destinations)
    print(bot.greeting())
    while not bot.finished:
        try:
            user_text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        print(render_response_text(bot.respond(user_text)))
if __name__ == "__main__":
    run()
