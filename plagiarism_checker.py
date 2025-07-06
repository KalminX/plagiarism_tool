import os
import re
import aiohttp
import asyncio
import nltk
from bs4 import BeautifulSoup
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import GOOGLE_API_KEY, SEARCH_ENGINE_ID, THRESHOLD, NUM_KEYWORDS, NUM_SEARCH_RESULTS


nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:NUM_KEYWORDS]

def compute_similarity(original, documents):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([original] + documents)
    return cosine_similarity(vectors[0:1], vectors[1:])[0]

async def search_google(session, query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": NUM_SEARCH_RESULTS,
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        return [item["link"] for item in data.get("items", [])]

async def extract_text_from_url(session, url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=' ', strip=True)
    except:
        return ""

async def analyze_text(original_text):
    keywords = extract_keywords(original_text)
    all_texts = []
    all_urls = []

    async with aiohttp.ClientSession() as session:
        for keyword in keywords:
            urls = await search_google(session, keyword)
            fetched = await asyncio.gather(*(extract_text_from_url(session, url) for url in urls))
            for url, text in zip(urls, fetched):
                if text:
                    all_texts.append(text)
                    all_urls.append(url)

    similarities = compute_similarity(original_text, all_texts)
    highest_score = max(similarities) if similarities.size else 0
    highest_index = similarities.argmax() if similarities.size else -1
    verdict = "⚠️ Plagiarism Suspected" if any(score >= THRESHOLD for score in similarities) else "✅ Clean"

    return {
        "keywords": keywords,
        "similarities": [round(float(s) * 100, 2) for s in similarities],
        "results": [
            {
                "url": str(all_urls[i]),
                "score": round(float(score) * 100, 2),  # now in percentage
                "plagiarized": bool(score >= THRESHOLD)
            }
            for i, score in enumerate(similarities)
        ],
        "highest_score": round(float(highest_score) * 100, 2),  # now in percentage
        "highest_url": str(all_urls[highest_index]) if highest_index >= 0 else None,
        "verdict": verdict
    }
