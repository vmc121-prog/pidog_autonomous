from pidog.llm import Ollama

INSTRUCTIONS = "You are a helpful assistant."
WELCOME = "Hello, I am a helpful assistant. How can I help you?"

# If Ollama runs on the same Raspberry Pi, use "localhost".
# If it runs on another computer in your LAN, replace with that computer's IP address.
llm = Ollama(
    ip="192.168.1.178",
    model="qwen3-coder:30b"   # you can replace with any model
)

# Basic configuration
llm.set_max_messages(20)
llm.set_instructions(INSTRUCTIONS)
llm.set_welcome(WELCOME)

print(WELCOME)

while True:
    text = input(">>> ")
    if text.strip().lower() in {"exit", "quit"}:
        break

    # Response with streaming output
    response = llm.prompt(text, stream=True)
    for token in response:
        if token:
            print(token, end="", flush=True)
    print("")
