from flask import Flask, render_template, request, send_file
import os
import requests
import time
from langchain_community.llms import CTransformers
from fpdf import FPDF

app = Flask(__name__)

# 🔗 Hugging Face URL to your model binary
MODEL_URL = "https://huggingface.co/saad07777/llama-2-7b-chat-binary/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin"
MODEL_PATH = os.path.join("models", "llama-2-7b-chat.ggmlv3.q8_0.bin")

# ✅ Download model from Hugging Face if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("📥 Downloading model from Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded successfully!")

download_model()

# ✅ Load model once
llm = CTransformers(
    model=MODEL_PATH,
    model_type="llama",
    config={"temperature": 0.01}
)

# ✅ Blog generation logic
def generate_blog(topic, word_count, style):
    try:
        word_count = int(word_count)
        max_tokens = min(word_count * 2, 512)

        prompt = f"""
        Write a blog for {style} job profile on the topic "{topic}" 
        within {word_count} words.
        """

        response = llm.invoke(prompt, max_new_tokens=max_tokens)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ✅ Save blog as PDF
def save_as_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True, align='L')
    pdf_path = "blog_output.pdf"
    pdf.output(pdf_path)
    return pdf_path

# ✅ Flask routes
@app.route('/', methods=["GET", "POST"])
def home():
    blog = ""
    if request.method == "POST":
        topic = request.form["topic"]
        word_count = request.form["word_count"]
        style = request.form["style"]
        blog = generate_blog(topic, word_count, style)
    return render_template("index.html", blog=blog)

@app.route('/download', methods=["POST"])
def download():
    content = request.form["blog_content"]
    pdf_path = save_as_pdf(content)
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

