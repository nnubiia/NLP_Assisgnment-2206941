from my_chatbot import CineBot #GENRE_MAP
#print("GENRE_MAP:", GENRE_MAP)

if __name__ == "__main__":
    chatbot = CineBot()
    chatbot.greeting()
    response = chatbot.respond()
    while chatbot.conversation_is_active():
        response = chatbot.respond(response)
    chatbot.farewell()