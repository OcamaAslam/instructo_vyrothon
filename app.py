import gradio as gr
from inference import run

def chat_interface(message, history):
   
    formatted_history = []
    for user_msg, bot_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        formatted_history.append({"role": "assistant", "content": bot_msg})
    
    response = run(message, formatted_history)
    return response

demo = gr.ChatInterface(
    fn=chat_interface,
    title="Pocket-Agent Local Demo",
    description="Test your model's tool-calling and refusal capabilities below.",
    examples=[
        "What's the weather in Tokyo?",
        "Convert 100 USD to PKR",
        "Book a flight to New York (Testing Refusal)",
        "SELECT * FROM users;"
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)