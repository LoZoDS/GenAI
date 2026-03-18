from selenium import webdriver
import time
import re
import json
import re
from pathlib import Path
import pymupdf  # Change 1: pdf library name is changed from fitz to pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup


# == PART 1: Gather documents. Set parameters == #

BASE_DIR = Path(__file__).resolve().parent  # Change 2: relative path instead of absolute
FILES_DIR = str(BASE_DIR)
OUTPUT_PATH = FILES_DIR + "/cdev_knowledge_base.json"

# b) Get CDC's pdf path
CDC_PATH = "cdc-milestone-checklists-ltsae-english-508.pdf"

# c) Scrape UNICEF webiste to get source code for each development milestone
SLEEP = 3
age_list = [
    'your-babys-developmental-milestones-2-months', # 2 months old
    'your-babys-developmental-milestones-4-months', # 4 months old
    'your-babys-developmental-milestones-6-months', # 6 months old
    'your-babys-developmental-milestones-9-months', # 9 months old
    'your-toddlers-developmental-milestones-1-year', # 1 year old
    'your-toddlers-developmental-milestones-18-months', # 18 months old
    'your-toddlers-developmental-milestones-2-years' # 2 year old
]

for age in age_list:
    # Get the page source for the URL
    URL = f"https://www.unicef.org/parenting/child-development/{age}"
    driver = webdriver.Chrome()
    driver.get(URL)
    time.sleep(SLEEP)
    page_source = driver.page_source

    # Parse HTML to get readable content
    soup = BeautifulSoup(page_source, "html.parser")

    # Find all the <div> elements with class "field_component_text_content"
    divs = soup.find_all("div", class_ = "field_component_text_content")

    content = []

    # For all divisions found remove any that are hyperlinks since these are not content based on inspect page
    for div in divs:
        for url_tag in div.find_all("a"): # <a> tag represents a link in HTML
            url_tag.decompose()
        text = div.get_text(separator = "\n") # to get only the text content
        lines = [line.strip() for line in text.splitlines() if line.strip()] # breaks text into lines and removes whitespaces
        lines = [line for line in lines if not re.match(r'https?://\S+', line)] # Final check to remove lines that are URLs
        content.append("\n".join(lines))

    cleaned_file = "\n\n".join(content) # Remove any double breaks

    with open(f"{FILES_DIR}/{age}.txt",'w') as file:
        file.write(cleaned_file)
    driver.quit()

# d) Mapping for UNICEF and CDC (pages)
UNICEF_PAGES = [
    ("2 months",  "your-babys-developmental-milestones-2-months"),
    ("4 months",  "your-babys-developmental-milestones-4-months"),
    ("6 months",  "your-babys-developmental-milestones-6-months"),
    ("9 months",  "your-babys-developmental-milestones-9-months"),
    ("1 year",    "your-toddlers-developmental-milestones-1-year"),
    ("18 months", "your-toddlers-developmental-milestones-18-months"),
    ("2 years",   "your-toddlers-developmental-milestones-2-years"),
]

CDC_PAGES = [
    ("2 months", 0, 1),
    ("4 months", 2, 3),
    ("6 months", 4, 5),
    ("9 months", 6, 7),
    ("12 months", 8, 9),
    ("15 months", 10, 11),
    ("18 months", 12, 13),
    ("2 years", 14, 15),
    ("30 months", 16, 17),
    ("3 years", 18, 19),
    ("4 years", 20, 21),
    ("5 years", 22, 23)
]

# Chunking parameters for knowlege base
CHUNK_SIZE    = 500  # This is the maximum number of characters per chunk
CHUNK_OVERLAP = 100  # Overlapping characters with neighbouring chunks


# == Part 2: Clean data == #
# b) Load text files
def load_txt_file(filepath: str) -> str:
    """
    Read UNICEF page source saved as txt.
    filepath:
        relative to the .txt file
    Returns:
        raw file content as a single string
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# c) Define removable text
# Use regexp to identify and remove unwanted UNICEF patterns from files (identified using AI)
# compile into a single regexp object for efficiency
# update if new patterns are discovered when handling data later in the RAG system.
UNICEF_REMOVE_PATTERNS = [
    r"^Skip to main content$",
    r"^Search(?: form)?$",
    r"^Clear search input$",
    r"^Search for result$",
    r"^(Hide|Show) filters$",
    r"^Selecting a value will cause the page",
    r"^Show results from:",
    r"^Max$",
    r"^Explainer$",
    r"^(English|Français|Español|العربية)$",
    r"^Test your knowledge$",
    r"^True or false\?",
    r"^True!$",
    r"^< Back to Parenting Milestones$",
    r"^\| $",
    r"^(2|4|6|9|1|18) (Months?|Year)$",
    r"^Explore age groups$",
    r"^Baby tips and resources$",
    r"^(Video|Article)$",
    r"^(Watch|Read) now$",
    r"^ShareThis Copy and Paste$",
    r"^>>",
    r"^To learn more, see our$",
    r"^mini parenting master class",
    r"^\.$",
]
_REMOVE_RE = [re.compile(p, re.IGNORECASE) for p in UNICEF_REMOVE_PATTERNS]

# c\d) function to identify unwanted patterns
def is_removable_text(line: str) -> bool:
    """Returns true if line has removable pattern from UNICEF_REMOVE_PATTERNS"""
    return any(pattern.match(line) for pattern in _REMOVE_RE)


# e) function to clean UNICEF files
def clean_unicef_text(raw: str) -> str:
    """
    Clean UNICEF page source text and return cleaned text by [1] splitting text into lines,
    [2] stripping whitespaces, [3] dropping empty or removable lines, [4] rejoining and collapsing gaps.
    """

    lines = raw.splitlines()            #[1]

    keep = []
    for line in lines:
        line = line.strip()             #[2]
        if not line:                    #[3]
            continue
        if is_removable_text(line):     #[3]
            continue
        keep.append(line)

    text = "\n".join(keep)              #[4]
    text = re.sub(r"\n{3,}", "\n\n", text) # replace 3+ lines for only 2
    text = re.sub(r"[\u200b\u00a0]+", " ", text) # replace invisible unicode spaces with regular text spaces

    return text.strip()

# e) Function to clean CDC PDF file
def clean_cdc_pdf(raw: str) -> str:
    """ Using PyMuPDF to clean CDC PDF and normalize data by [1] removing footer lines,
    [2] removing checklist/bullet symbols, [3] collapsing gaps and breaks"""
    raw = re.sub(r"(www\.cdc\.gov/ActEarly|1-800-CDC-INFO|Learn the Signs\. Act Early\.)",
        "", raw, flags = re.IGNORECASE)     #[1]

    raw = re.sub(r"[◦□✓■◼•]\s*", "", raw)   #[2]

    raw = re.sub(r"[ \t]{2,}", " ", raw)    #[3]
    raw = re.sub(r"\n{3,}", "\n\n", raw)    #[3]

    return raw.strip()

# == PART 3: CHUNK DATA == #
# a) function to create chunks
def create_chunk(text: str, metadata: dict) -> list[dict]:
    """ Split text into chunks and attach metadata to each chunk.
    Uses LangChain's RecursiveCharacterTextSpitter function with parameters set on part 1

    text:
        cleaned plain text of each source
    metadata:
        dictionary attached to each chunk
    returns:
        list of dictionaries containing the data for each chunk
    """
    splitter = RecursiveCharacterTextSplitter( # LangChain function
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators= ["\n\n", "\n", ". ", " ", ""]
    )

    raw_chunks = splitter.split_text(text) # Apply splitter to the actual text

    return [   # Enumerates each chunk and for each index, chunk returns the chunk and matadata information to identify chunk
        {
            "text": chunk,
            "metadata": {**metadata, "chunk_index": i}
        }
        for i, chunk in enumerate(raw_chunks)
    ]

# == Part 4: Ingest and clean the data using functions above == #
def ingest_unicef() -> list[dict]:
    """ For each UNICEF URL [1] load the raw text data, [2] clean it, [3] chunk it.
    return:
        list of chunk dictionaries for milestone stages (i.e. 2 months)"""
    all_chunks = []

    for age_label, page in UNICEF_PAGES:
        filepath = f"{FILES_DIR}/{page}.txt"
        print(f" Reading UNICEF [{age_label}] ← {filepath}")

        # load text and clean
        raw = load_txt_file(filepath)
        text = clean_unicef_text(raw)

        metadata = {
            "source": "UNICEF",
            "age": age_label,
            "Category": "Developmental Milestones"
        }

        chunks = create_chunk(text, metadata)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks")

    return all_chunks

def ingest_cdc(pathpdf: str) -> list[dict]:
    """ For each age group in the combined CDC PDF [1] extract data, [2] clean it,
    [3] chunk it.
    return:
        list of chunk dictionaries for milestone stages (i.e. 5 years)"""
    all_chunks = []
    doc = pymupdf.open(pathpdf)

    for age_label, first_page, last_page in CDC_PAGES:
        print(f"Processing CDC PDF [{age_label}] (pages {first_page} - {last_page}")

        raw = ""
        for page_number in range(first_page, last_page +1):
            if page_number < len(doc):
                raw += doc[page_number].get_text()

        text = clean_cdc_pdf(raw)

        metadata = {
            "source": "CDC",
            "age": age_label,
            "Category": "Developmental Milestones"
        }

        chunks = create_chunk(text, metadata)
        all_chunks.extend(chunks)
        print(f" → {len(chunks)} chunks")

    doc.close()
    return all_chunks

# == PART 5: Run Program == #
def main():

    #a) Start program
    print("\n RAG Indexing Pipeline - Early Childhood Development Milestones")
    knowledge_base: list[dict] = []

    #b) Load UNICEF
    print("\n Step 1/3 ~ Loading UNICEF text files")
    unicef_knowledge_base = ingest_unicef()
    knowledge_base.extend(unicef_knowledge_base)

    #c) Load CDC
    print("\n Step 2/3 ~ Loading CDC PDF File")
    cdc_knowledge_base = ingest_cdc(CDC_PATH)
    knowledge_base.extend(cdc_knowledge_base)

    #d) Save data into knowledge base json file
    print("\n Step 3/3 ~ Saving {len(knowledge_base)} total chunks → {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

    #e) finalize program
    print("\n Indexing Complete")
    print(f"\n UNICEF chunks: {len(unicef_knowledge_base)}, CDC chunks: {len(cdc_knowledge_base)}")

# == Part 6: Run Program ==#
if __name__ == "__main__":
    main()