import datetime as dt 
from pathlib import Path
from typing import List

import srsly
import tqdm
import arxiv
from arxiv import Result
from retry import retry 
import spacy
from spacy.language import Language
from .types import ArxivArticle
from rich.console import Console 

from .constants import MAX_ARTICLE_AGE_DAYS, SEARCH_QUERY

console = Console()


def age_in_days(res: Result) -> float:
    """Get total seconds from now from Arxiv result"""
    now = dt.datetime.now(dt.timezone.utc)
    return (now - res.published).total_seconds() / 3600 / 24


def parse(res: Result, nlp: Language) -> ArxivArticle:
    """Parse proper Pydantic object from Arxiv"""
    summary = res.summary.replace("\n", " ")
    doc = nlp(summary)
    sents = [s.text for s in doc.sents]
    
    return ArxivArticle(
        created=str(res.published)[:19], 
        title=str(res.title),
        abstract=summary,
        sentences=sents,
        url=str(res.entry_id)
    )

@retry(tries=5, delay=1, backoff=2)
def main():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    console.log(f"Starting arxiv search.")
    items = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    results = list(items.results())

    console.log(f"Found {len(results)} results.")

    articles = [dict(parse(r, nlp=nlp)) 
                for r in tqdm.tqdm(results) 
                if age_in_days(r) < MAX_ARTICLE_AGE_DAYS and r.primary_category.startswith("cs")]

    dist = [age_in_days(r) for r in results]
    if dist:
        console.log(f"Minimum article age: {min(dist)}")
        console.log(f"Maximum article age: {max(dist)}")
    articles_dict = {ex['title']: ex for ex in articles}
    data_downloads = list(sorted(Path("data/downloads/").glob("*.jsonl")))
    if len(data_downloads) > 0:
        most_recent = srsly.read_jsonl(data_downloads[-1])
        old_articles_dict = {ex['title']: ex for ex in most_recent}
    else:
        old_articles_dict = {}

    new_articles = [ex for title, ex in articles_dict.items() if title not in old_articles_dict.keys()]
    old_articles = [ex for title, ex in articles_dict.items() if title in old_articles_dict.keys()]
    if old_articles:
        console.log(f"Found {len(old_articles)} old articles in current batch. Skipping.")
    if new_articles:
        console.log(f"Found {len(new_articles)} new articles in current batch to write.")
        filename = str(dt.datetime.now()).replace(" ", "-")[:13] + "h.jsonl"
        srsly.write_jsonl(Path("data") / "downloads" / filename, new_articles)
        console.log(f"Wrote {len(new_articles)} articles into {filename}.")
