"""
Fetch the Book of Proverbs (World English Bible, public domain) from bible-api.com,
strip verse numbers / titles, and format each chapter as a single prose document.

Outputs (in this directory):
    proverbs.jsonl       one {"text": <chapter prose>} per line, 31 chapters
    proverbs_prose.txt    human-readable, blank line between chapters
"""
import json, re, ssl, time, urllib.request, os

HERE = os.path.dirname(os.path.abspath(__file__))
# The sandbox proxy presents a self-signed cert; disable verification for this
# read-only fetch of public-domain text.
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

N_CHAPTERS = 31
RAW_DIR = os.path.join(HERE, "raw_chapters")


def fetch_chapter(ch: int) -> str:
    # Resume support: cache each chapter's raw JSON so a rate-limit restart
    # doesn't re-request what we already have.
    cache = os.path.join(RAW_DIR, f"proverbs_{ch:02d}.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
    else:
        url = f"https://bible-api.com/proverbs%20{ch}?translation=web"
        d = None
        for attempt in range(6):
            try:
                r = urllib.request.urlopen(url, timeout=30, context=CTX)
                d = json.load(r)
                break
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"  ch{ch} attempt {attempt} failed: {type(e).__name__} {e} "
                      f"-- waiting {wait}s")
                time.sleep(wait)
        if d is None:
            raise RuntimeError(f"could not fetch chapter {ch}")
        os.makedirs(RAW_DIR, exist_ok=True)
        json.dump(d, open(cache, "w"))
    verses = [v["text"] for v in d["verses"]]
    # Join verses into flowing prose: collapse the poetic line breaks and any
    # repeated whitespace into single spaces. Verse numbers are not in the text
    # field, so nothing else to strip.
    prose = re.sub(r"\s+", " ", " ".join(verses)).strip()
    return prose


def main():
    chapters = []
    for ch in range(1, N_CHAPTERS + 1):
        cached = os.path.exists(os.path.join(RAW_DIR, f"proverbs_{ch:02d}.json"))
        prose = fetch_chapter(ch)
        chapters.append(prose)
        print(f"chapter {ch}: {len(prose)} chars, ~{len(prose.split())} words"
              + ("" if cached else "  (fetched)"))
        if not cached:
            time.sleep(3)  # be polite; bible-api rate-limits bursts

    with open(os.path.join(HERE, "proverbs.jsonl"), "w") as f:
        for prose in chapters:
            f.write(json.dumps({"text": prose}) + "\n")
    with open(os.path.join(HERE, "proverbs_prose.txt"), "w") as f:
        f.write("\n\n".join(chapters) + "\n")
    total_words = sum(len(c.split()) for c in chapters)
    print(f"\nwrote {len(chapters)} chapters, ~{total_words} words total")


if __name__ == "__main__":
    main()
