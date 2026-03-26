from chat_service import ask_chatbot

result = ask_chatbot("What are common language milestones at 2 years?")

print("ANSWER:")
print(result["answer"])
print()
print("SOURCES:")
print(result["sources"])
print()
print("STATUS:")
print(result["status"])
print("SAFETY LABEL:")
print(result["safety_label"])