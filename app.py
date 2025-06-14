from flask import Flask, render_template, request, send_file
import os
import gdown
from langchain_community.llms import CTransformers


from fpdf import FPDF

app = Flask(__name__)
MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"
GDRIVE_FILE_ID = "1j8Jti8LX1sRg-7jDWFWj16_R09lhdXaH"

# Download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

download_model()

# Load model only once
llm = CTransformers(
    model=MODEL_PATH,
    model_type="llama",
    config={"temperature": 0.01}
)

# Blog generation logic
def generate_blog(topic, word_count, style):
    try:
        word_count = int(word_count)
        max_tokens = min(word_count * 2, 512)
        prompt = f"""
        Write a blog for {style} job profile on the topic "{topic}" 
        within {word_count} words.
        """
        return llm(prompt, max_new_tokens=max_tokens)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Save blog as PDF
def save_as_pdf(text, filename="blog_output.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, txt=line)
    pdf.output(filename)
    return filename

@app.route("/", methods=["GET", "POST"])
def index():
    blog_text = None
    if request.method == "POST":
        topic = request.form["topic"]
        word_count = request.form["word_count"]
        style = request.form["style"]
        blog_text = generate_blog(topic, word_count, style)
        save_as_pdf(blog_text)
    return render_template("index.html", blog=blog_text)

@app.route("/download")
def download():
    return send_file("blog_output.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
