import html
import json
import random
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_FILE = DATA_DIR / "dutch_subset.jsonl"
LANGUAGES = {"Standaardnederlands", "Standaardnederlands (allochtoon)"}
MIN_WORDS = 100
MAX_WORDS = 500
SUBSET_SIZE = 1000
SEED = 64

# these are the parameters I created the subset with, if you change them ,please tell me.
# This script selects Dutch stories (without an @ to exclude weird email entries), of at min 100
# and max 500 words for chunking. For now 1000 entries, can be less without affecting the qa pairs written,
# as long as the stories of those pairs are in the first N, I would to create questions only from the first 100.
# I also only include stories with a title. 

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(text):
    if re.search(r"</?(?:p|br|div|span|em|strong|b|i|a|ul|ol|li|h[1-6]|script|style)\b", text, re.I):
        parser = TextExtractor()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
    text = unicodedata.normalize("NFC", html.unescape(text)).replace("\u00ad", "")
    return " ".join(text.split())


# Keep public Dutch stories with a title and 100-500 words.
stories = []
for path in sorted(DATA_DIR.glob("20231201_verhalenbank_*.json")):
    with path.open(encoding="utf-8") as file:
        documents = json.load(file)["response"]["docs"]

    for document in documents:
        languages = [label.strip() for label in document.get("language", []) if label.strip()]
        if document["public"] is not True:
            continue
        if not languages or not set(languages).issubset(LANGUAGES):
            continue

        title = clean_text("\n".join(document.get("title", [])))
        if not title:
            continue

        text = clean_text("\n".join(document.get("main_text", [])))
        # Remove sentences containing email addresses or another @ character.
        text = " ".join(
            sentence for sentence in re.split(r"(?<=[.!?])\s+", text)
            if "@" not in sentence
        )
        word_count = len(re.findall(r"\b\w+(?:['\u2019-]\w+)*\b", text))
        if not MIN_WORDS <= word_count <= MAX_WORDS:
            continue


        if document["word_count"] is None:
            word_count = len(re.findall(r"\b\w+(?:['\u2019-]\w+)*\b", text))
        
        doc_date = document.get("date", None)

        doc_type = document.get("type", None)
        
        doc_tag = document.get("tag", [])
        
        doc_description = document.get("description", None)

        stories.append({
            "story_id": document["id"],
            "title": title,
            "text": text,
            "language": languages,
            "subgenre": [genre.strip() for genre in document.get("subgenre", []) if genre.strip()],
            "word_count": document["word_count"],
            "date": doc_date,
            "type": doc_type,
            "tags": doc_tag,
            "description": doc_description,
        })

# Remove duplicate texts, keeping the first story by ID.
unique = {}
for story in sorted(stories, key=lambda story: story["story_id"]):
    unique.setdefault(story["text"].casefold(), story)
print(f"Removed {len(stories) - len(unique)} duplicate texts.")
stories = list(unique.values())

# Shuffle all eligible stories and save the first N.
rng = random.Random(SEED)
rng.shuffle(stories)
subset = stories[:SUBSET_SIZE]

with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as file:
    for story in subset:
        file.write(json.dumps(story, ensure_ascii=False) + "\n")

print(f"Saved {len(subset)} stories to {OUTPUT_FILE}")
