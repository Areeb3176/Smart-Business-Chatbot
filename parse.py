import os
from llama_parse import LlamaParse

# Replace with your actual API key
LLAMA_API_KEY = "llx-L3HUY6TCSsgZTTW4iYH1lv4k1r4wdLDP6CCW2g0XhutucLGV"

# Set your folders
PDF_FOLDER = "E:/ml projects/Smart-Business-Guide-1.0-main - Copy/pdf_folder"      # Folder with PDF files
OUTPUT_FOLDER = "E:/ml projects/Smart-Business-Guide-1.0-main - Copy/data"       # Where to save the .md files

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize parser
parser = LlamaParse(
    api_key=LLAMA_API_KEY,
    result_type="markdown",  # Return Markdown
    verbose=True,
)

# Loop over each PDF in folder
for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        print(f"Parsing: {pdf_path}")

        try:
            # Load document chunks
            documents = parser.load_data(pdf_path)

            # Combine all markdown chunks into one string
            full_markdown = "\n\n".join(doc.text for doc in documents)

            # Save as a single markdown file
            output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_markdown)

            print(f"✅ Saved: {output_path}")
        except Exception as e:
            print(f"❌ Failed to parse {filename}: {e}")
