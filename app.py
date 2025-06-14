from flask import Flask, render_template, request, send_file
import os
import requests
import time
from langchain_community.llms import CTransformers
from fpdf import FPDF

app = Flask(__name__)

# Use lightweight model to avoid memory timeout
MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"
MODEL_URL = "https://huggingface.co/saad07777/llama-2-7b-chat-binary/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin"

# Download model if not already present
def download_model():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded.")

download_model()

# Load model (cached for efficiency)
llm = CTransformers(
    model=MODEL_PATH,
    model_type="llama",
    config={"temperature": 0.01}
)

def getLLamaResponse(text_input, no_words, blog_style):
    try:
        no_words = int(no_words)
        max_tokens = min(no_words * 2, 512)
        prompt = f"""
        Write a blog for {blog_style} job profile on the topic "{text_input}" 
        within {no_words} words.
        """
        response = llm.invoke(prompt, max_new_tokens=max_tokens)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def save_as_pdf(blog_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in blog_text.split("\n"):
        pdf.multi_cell(0, 10, txt=line)
    pdf_path = "blog_output.pdf"
    pdf.output(pdf_path)
    return pdf_path

@app.route("/", methods=["GET", "POST"])
def index():
    blog = None
    pdf_ready = False
    if request.method == "POST":
        topic = request.form["topic"]
        word_count = request.form["word_count"]
        style = request.form["style"]
        blog = getLLamaResponse(topic, word_count, style)
        save_as_pdf(blog)
        pdf_ready = True
    return render_template("index.html", blog=blog, pdf_ready=pdf_ready)

@app.route("/download")
def download():
    return send_file("blog_output.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
