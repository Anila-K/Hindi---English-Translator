import gradio as gr
from transformers import MarianTokenizer, MarianMTModel

# Load the fine-tuned model from your local folder
model_path = "marian_hi_en_finetuned"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Translation function
def translate(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Gradio Interface
interface = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=4, label="Enter Hindi Text"),
    outputs=gr.Textbox(label="English Translation"),
    title="Hindi to English Translator",
    description="A MarianMT-based fine-tuned model that translates Hindi to English.",
    examples=["मुझे अंग्रेज़ी में अनुवाद चाहिए।", "आप कैसे हैं?"]
)

# Launch interface
interface.launch()
