import os
import time
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Set, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from urllib.parse import urlparse, urlunparse, urljoin
import logging
import aiohttp
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Setup ===
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

DEMO_CASE_URLS = [
    "https://supreme.justia.com/cases/federal/us/513/123/",
    "https://supreme.justia.com/cases/federal/us/262/390/",
]

MAX_DEMO_CASES = 2

# === Ollama Embeddings ===
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text", endpoint: str = "http://localhost:11434/api/embeddings"):
        self.model = model
        self.endpoint = endpoint

    def embed_query(self, text: str):
        return self._embed(text)

    def embed_documents(self, texts: List[str]):
        return [self._embed(t) for t in texts]

    def _embed(self, text: str):
        response = requests.post(self.endpoint, json={"model": self.model, "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]
############################################


async def fetch_all_related_documents(related_doc_urls, case_metadata):
    documents = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url_type, urls in related_doc_urls.items():
            for url in urls:
                tasks.append(fetch_related_document(session, url, case_metadata))
        
        results = await asyncio.gather(*tasks)
        for doc in results:
            if doc:
                documents.append(doc)
    
    return documents

##########################################
# === Metadata & Section Extractor ===

#############################


def extract_related_documents(soup, base_url, case_metadata):
    """
    Extract links to related documents for a Supreme Court case.
    
    Args:
        soup: BeautifulSoup object of the case page
        base_url: Base URL for the case
        case_metadata: Metadata dictionary for the case
        
    Returns:
        Dictionary with related document information
    """
    related_docs = {}
    
    # Find links to oral arguments
    oral_arg_links = soup.find_all("a", href=lambda h: h and "oral-argument" in h)
    if oral_arg_links:
        oral_arg_urls = []
        for link in oral_arg_links:
            url = urljoin(base_url, link["href"])
            oral_arg_urls.append(url)
        
        if oral_arg_urls:
            related_docs["oral_argument_urls"] = oral_arg_urls
    
    # Find links to briefs
    brief_links = soup.find_all("a", href=lambda h: h and "brief" in h)
    if brief_links:
        brief_urls = []
        for link in brief_links:
            url = urljoin(base_url, link["href"])
            brief_urls.append(url)
        
        if brief_urls:
            related_docs["brief_urls"] = brief_urls
    
    # Find links to lower court decisions
    lower_court_links = soup.find_all("a", href=lambda h: h and ("circuit" in h or "district" in h))
    if lower_court_links:
        lower_court_urls = []
        for link in lower_court_links:
            url = urljoin(base_url, link["href"])
            lower_court_urls.append(url)
        
        if lower_court_urls:
            related_docs["lower_court_urls"] = lower_court_urls
    
    return related_docs

async def fetch_related_document(session, url, case_metadata):
    """
    Fetch and process a related document.
    
    Args:
        session: aiohttp ClientSession
        url: URL of the related document
        case_metadata: Metadata of the parent case
        
    Returns:
        Document object for the related document
    """
    try:
        async with session.get(url, headers=HEADERS) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                
                # Determine document type from URL
                doc_type = "unknown"
                if "oral-argument" in url:
                    doc_type = "oral_argument"
                elif "brief" in url:
                    doc_type = "brief"
                elif "circuit" in url or "district" in url:
                    doc_type = "lower_court_decision"
                
                # Extract content based on document type
                content_div = None
                if doc_type == "oral_argument":
                    content_div = soup.find("div", class_="transcript")
                elif doc_type in ["brief", "lower_court_decision"]:
                    content_div = soup.find("div", class_="opinion") or soup.find("div", class_="column-center")
                
                if not content_div:
                    content_div = soup.find("div", class_="entry-content") or soup
                
                # Create metadata
                metadata = {
                    **case_metadata,  # Include parent case metadata
                    "source": url,
                    "document_type": doc_type,
                    "relation": f"related_to_{case_metadata.get('citation', 'unknown_case')}"
                }
                
                # Create document
                return Document(
                    page_content=content_div.get_text(separator="\n", strip=True),
                    metadata=metadata
                )
    except Exception as e:
        print(f"Error fetching related document {url}: {e}")
    
    return None


##################################

#################################

def create_legal_document_splitter():
    """
    Create a document splitter optimized for legal documents.
    
    Returns:
        A configured text splitter
    """
    # Legal-specific separators in order of priority
    legal_separators = [
        "\n\n## ",  # Markdown section headers
        "\n\nSYLLABUS\n\n",
        "\n\nOPINION OF THE COURT\n\n",
        "\n\nCONCURRING OPINION\n\n",
        "\n\nDISSENTING OPINION\n\n",
        "\n\n",      # Paragraphs
        ". ",        # Sentences
        " ",         # Words
        ""           # Characters
    ]
    
    return RecursiveCharacterTextSplitter(
        separators=legal_separators,
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )


def generate_chunk_metadata(doc, chunk_idx, total_chunks):
    """
    Generate metadata for each chunk that maintains context about its position
    within the original document.
    
    Args:
        doc: The original document
        chunk_idx: The index of this chunk
        total_chunks: Total number of chunks from the document
        
    Returns:
        Enhanced metadata dictionary for the chunk
    """
    # Copy original metadata
    metadata = doc.metadata.copy()
    
    # Add chunking metadata
    metadata["chunk_id"] = chunk_idx
    metadata["total_chunks"] = total_chunks
    metadata["is_first_chunk"] = (chunk_idx == 0)
    metadata["is_last_chunk"] = (chunk_idx == total_chunks - 1)
    
    # Add start/end markers
    if chunk_idx == 0:
        metadata["contains"] = metadata.get("contains", []) + ["document_start"]
    if chunk_idx == total_chunks - 1:
        metadata["contains"] = metadata.get("contains", []) + ["document_end"]
    
    return metadata

######################################

def smart_legal_chunking(documents):
    """
    Apply smart chunking to legal documents with enhanced metadata.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of chunked Document objects with enhanced metadata
    """
    splitter = create_legal_document_splitter()
    chunked_docs = []
    
    for doc in documents:
        # Split this document
        chunks = splitter.split_text(doc.page_content)
        total_chunks = len(chunks)
        
        # Create new documents with enhanced metadata
        for i, chunk_text in enumerate(chunks):
            enhanced_metadata = generate_chunk_metadata(doc, i, total_chunks)
            chunked_docs.append(Document(
                page_content=chunk_text,
                metadata=enhanced_metadata
            ))
    
    return chunked_docs


############################
def extract_citation_network(soup, full_text):
    """
    Extract citation network information from a Supreme Court case.
    
    Args:
        soup: BeautifulSoup object of the case page
        full_text: Full text of the case
        
    Returns:
        Dictionary with citation network information
    """
    citation_network = {}
    
    # Find cases cited in this opinion
    # Pattern for US Reports citations
    cited_cases = set()
    citation_pattern = r"(\d{1,3})\s+U\.\s*S\.\s+(\d{1,4})"
    
    for match in re.finditer(citation_pattern, full_text):
        citation = match.group(0)
        if citation not in cited_cases:
            cited_cases.add(citation)
    
    if cited_cases:
        citation_network["cited_cases"] = list(cited_cases)
    
    # For cases that cite this one, we'd need to either:
    # 1. Look for "later citations" sections on the page
    # 2. Maintain a separate database of citation relationships
    later_citations_div = soup.find("div", class_=lambda c: c and "later-citations" in c.lower())
    if later_citations_div:
        citing_cases = []
        citation_links = later_citations_div.find_all("a")
        for link in citation_links:
            case_name = link.get_text(strip=True)
            if case_name and case_name not in citing_cases:
                citing_cases.append(case_name)
        
        if citing_cases:
            citation_network["cited_by"] = citing_cases
    
    # Extract key legal principles and holdings
    holdings = []
    principle_patterns = [
        r"We hold that\s+(.+?\.)",
        r"The Court h[eo]ld[s]? that\s+(.+?\.)",
        r"It is h[eo]ld that\s+(.+?\.)"
    ]
    
    for pattern in principle_patterns:
        for match in re.finditer(pattern, full_text):
            holding = match.group(1).strip()
            if holding not in holdings:
                holdings.append(holding)
    
    if holdings:
        citation_network["holdings"] = holdings
    
    return citation_network
###########################
def extract_enhanced_metadata(soup, url):
    """
    Extract comprehensive metadata from Supreme Court case HTML.
    
    Args:
        soup: BeautifulSoup object of the case page
        url: The URL of the case page
        
    Returns:
        Dictionary of metadata
    """
    metadata = {"source": url}
    case_data_div = soup.find("div", class_="case-data")

    # Basic case information
    # Try to extract title from h1 or fall back to <title> tag
    title_element = soup.find("h1", class_="title") or soup.find("title")
    if title_element:
        full_title = title_element.get_text(strip=True)
        metadata["title_full"] = full_title

    # Try to extract something like "X v. Y"
    case_match = re.search(r"([A-Z][A-Za-z0-9\s\.\-]+ v\. [A-Z][A-Za-z0-9\s\.\-]+)", full_title)
    if case_match:
        metadata["title_short"] = case_match.group(1)
    else:
        metadata["title_short"] = full_title.split("|")[0].strip()

    if case_data_div:
        # Extract citation
        citation_element = case_data_div.find("p", class_="citation")
        if citation_element:
            citation_text = citation_element.get_text(strip=True)
            citation_match = re.search(r"(\d+)\s+U\.S\.\s+(\d+)", citation_text)
            if citation_match:
                metadata["citation"] = citation_match.group(0)
                metadata["volume"] = citation_match.group(1)
                metadata["page_start"] = citation_match.group(2)
        
        # Extract docket number
        docket_element = case_data_div.find("p", class_="docket")
        if docket_element:
            docket_text = docket_element.get_text(strip=True)
            docket_match = re.search(r"No\.\s+([\w\-\.]+)", docket_text)
            if docket_match:
                metadata["docket"] = docket_match.group(1)
        
        # Extract dates
        date_elements = case_data_div.find_all("p", class_=lambda c: c and "date" in c)
        for date_element in date_elements:
            date_text = date_element.get_text(strip=True)
            
            # Argued date
            argued_match = re.search(r"Argued:\s+([A-Za-z]+ \d+, \d{4})", date_text)
            if argued_match:
                metadata["date_argued"] = argued_match.group(1)
            
            # Decided date
            decided_match = re.search(r"Decided:\s+([A-Za-z]+ \d+, \d{4})", date_text)
            if decided_match:
                metadata["date_decided"] = decided_match.group(1)
    
    # Extract court composition and vote information
    full_text = soup.get_text()
    
    # Try to extract vote count
    vote_patterns = [
        r"(?:decided|affirmed|reversed|remanded).*?by a vote of (\d+)[–-](\d+)",
        r"(\d+)[–-](\d+) (?:decision|vote|ruling)",
        r"(\d+)[–-](\d+) majority"
    ]
    
    for pattern in vote_patterns:
        vote_match = re.search(pattern, full_text, re.IGNORECASE)
        if vote_match:
            metadata["vote_majority"] = vote_match.group(1)
            metadata["vote_minority"] = vote_match.group(2)
            break
    
    # Try to extract justices information
    majority_justices = []
    dissenting_justices = []
    concurring_justices = []
    
    justice_matches = re.finditer(r"(?:Justice|Chief Justice)\s+([A-Z][a-z]+)", full_text)
    for match in justice_matches:
        justice_name = match.group(1)
        
        # Look at context to determine role
        context_start = max(0, match.start() - 50)
        context_end = min(len(full_text), match.end() + 50)
        context = full_text[context_start:context_end]
        
        if re.search(r"dissent", context, re.IGNORECASE):
            if justice_name not in dissenting_justices:
                dissenting_justices.append(justice_name)
        elif re.search(r"concur", context, re.IGNORECASE):
            if justice_name not in concurring_justices:
                concurring_justices.append(justice_name)
        else:
            # Assume majority opinion if not specified
            if justice_name not in majority_justices:
                majority_justices.append(justice_name)
    
    if majority_justices:
        metadata["majority_justices"] = ", ".join(majority_justices)
    if dissenting_justices:
        metadata["dissenting_justices"] = ", ".join(dissenting_justices)
    if concurring_justices:
        metadata["concurring_justices"] = ", ".join(concurring_justices)
    
    # Extract legal subject categories
    tags_div = soup.find("div", class_="tags")
    if tags_div:
        tags = [tag.get_text(strip=True) for tag in tags_div.find_all("a")]
        if tags:
            metadata["legal_topics"] = ", ".join(tags)
    
    # Try to extract statutes and constitution sections referenced
    statute_patterns = [
        r"\d+\s+U\.S\.C\.\s+[§\s]+\d+[a-z]*",  # US Code
        r"\d+\s+C\.F\.R\.\s+[§\s]+\d+\.\d+",   # CFR
        r"Section\s+\d+\s+of\s+the\s+[A-Za-z\s]+Act",  # Acts
        r"Amendment\s+[IVX]+",  # Constitutional amendments
        r"Article\s+[IVX]+,\s+Section\s+\d+",  # Constitution articles
    ]
    
    all_statutes = []
    for pattern in statute_patterns:
        statute_matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in statute_matches:
            statute = match.group(0)
            if statute not in all_statutes:
                all_statutes.append(statute)
    
    if all_statutes:
        metadata["referenced_statutes"] = ", ".join(all_statutes)
    
    # Extract procedural history
    procedural_patterns = [
        r"(?:The|This)\s+case\s+comes\s+to\s+us\s+from\s+the\s+([A-Za-z\s]+Court)",
        r"The\s+([A-Za-z\s]+Court)\s+(?:affirmed|reversed|remanded|held)",
        r"on\s+(?:writ\s+of\s+|petition\s+for\s+)?certiorari\s+to\s+the\s+([A-Za-z\s]+Court)"
    ]
    
    for pattern in procedural_patterns:
        court_match = re.search(pattern, full_text, re.IGNORECASE)
        if court_match:
            metadata["previous_court"] = court_match.group(1)
            break
    
    return metadata
########################
def extract_case_documents(html: str, url: str) -> List[Document]:
    """
    Extract case documents from Supreme Court case HTML with improved robustness.

    Each document type (syllabus, majority, dissent, etc.) is processed independently
    to avoid interference.
    """
    url = normalize_url(url)  # normalize URL to prevent duplicates with fragments

    documents = []
    soup = BeautifulSoup(html, "html.parser")
    metadata = extract_enhanced_metadata(soup, url)

    # Title fallback
    if "title" not in metadata and "title_full" not in metadata:
        title_element = soup.find("h1", class_="title") or soup.find("title")
        metadata["title"] = title_element.get_text(strip=True) if title_element else "Unknown Case"
    elif "title" not in metadata:
        metadata["title"] = metadata.get("title_full")

    # Full text for fallback
    full_text = soup.get_text(separator="\n", strip=True)
    metadata.update(extract_citation_network(soup, full_text))

    # Related documents
    related_doc_urls = extract_related_documents(soup, url, metadata)
    if related_doc_urls:
        metadata["related_documents"] = related_doc_urls

    # Define opinion types
    base_opinion_types = [
        {"name": "opinion_majority", "patterns": [r"OPINION OF THE COURT", r"OPINION OF (?:MR\\. )?JUSTICE", r"(?:MR\\. )?CHIEF JUSTICE .+ delivered the opinion", r"(?:MR\\. )?JUSTICE .+ delivered the opinion", r"PER CURIAM"]},
        {"name": "opinion_concurrence", "patterns": [r"(?:MR\\. )?JUSTICE .+ concurring", r"CONCURRING OPINION", r"(?:MR\\. )?JUSTICE .+, concurring"]},
        {"name": "opinion_dissent", "patterns": [r"(?:MR\\. )?JUSTICE .+ dissenting", r"DISSENTING OPINION", r"(?:MR\\. )?JUSTICE .+, dissenting"]}
    ]

    # Check for historical patterns
    if "date_decided" in metadata and re.search(r"(\\d{4})", metadata["date_decided"]):
        year = int(re.search(r"(\\d{4})", metadata["date_decided"]).group(1))
        if year < 1900:
            historical_patterns = {
                "opinion_majority": [r"The (unanimous )?opinion of the Court", r"The Chief Justice delivered the opinion", r"BY THE COURT", r"The Court was of opinion"],
                "opinion_concurrence": [r"(?:Justice|Judge) .+ concurred", r"[A-Z][a-z]+, Justice, concurring"],
                "opinion_dissent": [r"(?:Justice|Judge) .+ dissented", r"[A-Z][a-z]+, Justice, dissenting"]
            }
            for ot in base_opinion_types:
                ot["patterns"].extend(historical_patterns.get(ot["name"], []))

    # --- Process each section type separately ---

    # Syllabus
    def extract_syllabus():
        content_div = soup.find("div", id="opinion") or soup.find("article") or soup
        syllabus_markers = ["SYLLABUS", "Syllabus", "syllabus"]
        for marker in syllabus_markers:
            header = content_div.find(lambda tag: tag.name in ["h1", "h2", "h3", "h4", "h5"] and tag.get_text(strip=True) == marker)
            if header:
                text = extract_section_text(header)
                if text and len(text) > 20:
                    return text
        for marker in syllabus_markers:
            if marker in full_text:
                parts = full_text.split(marker, 1)
                if len(parts) > 1:
                    text = parts[1]
                    for end_marker in ["OPINION", "MR. JUSTICE", "JUSTICE", "CHIEF JUSTICE", "PER CURIAM"]:
                        if end_marker in text:
                            text = text.split(end_marker, 1)[0]
                            break
                    return text.strip()
        return None

    syllabus = extract_syllabus()
    if syllabus:
        documents.append(Document(page_content=syllabus, metadata={**metadata, "section": "syllabus"}))

    # Opinions (only one match per opinion type)
    for opinion_type in base_opinion_types:
        content_div = soup.find("div", id="opinion") or soup.find("article") or soup
        found = False
        for tag in content_div.find_all(["h1", "h2", "h3", "h4", "h5", "p", "div"]):
            tag_text = tag.get_text(strip=True)
            if any(re.search(p, tag_text, re.IGNORECASE) for p in opinion_type["patterns"]):
                text = extract_section_text(tag)
                if text and len(text) > 50:
                    author = None
                    for p in [r"(?:MR\\. )?JUSTICE\\s+([A-Z][A-Za-z]+)", r"(?:MR\\. )?CHIEF JUSTICE\\s+([A-Z][A-Za-z]+)", r"([A-Z][A-Za-z]+),\\s+(?:Justice|Chief Justice)"]:
                        m = re.search(p, tag_text, re.IGNORECASE)
                        if m:
                            author = m.group(1)
                            break
                    doc_meta = {**metadata, "section": opinion_type["name"]}
                    if author:
                        doc_meta["opinion_author"] = author
                    documents.append(Document(page_content=tag_text + "\n" + text, metadata=doc_meta))
                    found = True
                    break
        if not found:
            logger.info(f"No {opinion_type['name']} found for {url}")

    if not documents:
        logger.warning(f"Could not extract specific sections for {url}, using full text")
        content_div = soup.find("div") or soup
        documents.append(Document(
            page_content=content_div.get_text(separator="\n", strip=True),
            metadata={**metadata, "section": "full_text"}
        ))

    return documents

def extract_section_text(header_element):
    section_text = ""
    element = header_element.next_sibling
    stop_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
    stop_texts = ["OPINION", "SYLLABUS", "DISSENT", "CONCUR"]

    while element:
        if element.name in stop_tags and any(t in element.get_text(strip=True).upper() for t in stop_texts):
            break
        if isinstance(element, str):
            section_text += element
        elif element.name and element.get_text(strip=True):
            section_text += element.get_text() + "\n"
        element = element.next_sibling

    if len(section_text.strip()) < 100:
        parent = header_element.parent
        if parent:
            full = parent.get_text(separator="\n", strip=True)
            header_txt = header_element.get_text(strip=True)
            idx = full.find(header_txt)
            if idx >= 0:
                section_text = full[idx + len(header_txt):].strip()

    return section_text.strip()


# === Crawl and collect cases ===

def normalize_url(url: str) -> str:
    """Normalize a URL by removing fragments and query strings."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

CASE_URL_PATTERN = re.compile(
    r"^https://supreme\.justia\.com/cases/federal/us/[^/]+/[^/]+/?$",
    re.IGNORECASE,
)

INDEX_URL_PREFIX = "https://supreme.justia.com/cases/federal/us/"


def is_case_url(url: str) -> bool:
    return bool(CASE_URL_PATTERN.match(normalize_url(url)))


def is_justia_supreme_url(url: str) -> bool:
    return normalize_url(url).startswith(INDEX_URL_PREFIX)


def extract_candidate_links(html: str, current_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        new_url = normalize_url(urljoin(current_url, href))

        if not is_justia_supreme_url(new_url):
            continue

        # Keep case pages and useful index pages.
        if (
            is_case_url(new_url)
            or "/year/" in new_url
            or "/volume/" in new_url
            or new_url.rstrip("/") == INDEX_URL_PREFIX.rstrip("/")
        ):
            links.append(new_url)

    return list(dict.fromkeys(links))


def crawl_site(start_url, max_pages=100, visited=None):
    if visited is None:
        visited = set()

    from collections import deque

    queue = deque([normalize_url(start_url)])
    documents = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        norm_url = normalize_url(url)

        if norm_url in visited:
            continue

        if not is_justia_supreme_url(norm_url):
            continue

        visited.add(norm_url)
        logger.info(f"🌐 Fetching [{len(visited)}/{max_pages}]: {norm_url}")

        try:
            html = fetch_html(norm_url)

            if is_case_url(norm_url):
                docs = extract_case_documents(html, norm_url)
                logger.info(f"✅ Extracted {len(docs)} document sections from case: {norm_url}")
                documents.extend(docs)
            else:
                logger.info(f"📚 Index/discovery page, not extracting case text: {norm_url}")

            for new_url in extract_candidate_links(html, norm_url):
                if new_url not in visited:
                    queue.append(new_url)

        except Exception as e:
            logger.warning(f"⚠️ Failed to process {norm_url}: {e}")
            continue

    logger.info(f"📄 Total extracted document sections: {len(documents)}")
    return documents
############################
def fetch_html(url: str) -> str:
    try:
        time.sleep(REQUEST_DELAY_SECONDS)

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 403:
            raise RuntimeError(f"403 Forbidden. Justia blocked this request: {url}")

        if response.status_code == 404:
            raise RuntimeError(f"404 Not Found: {url}")

        response.raise_for_status()

        if not response.text or len(response.text.strip()) < 500:
            raise RuntimeError(f"Response was too small or empty: {url}")

        return response.text

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed for {url}: {e}") from e
# === Main process ===
def main():
    base_url = "https://supreme.justia.com/cases/federal/us/year/2024.html"
    print(f"🚀 Starting crawl from: {base_url}")
documents = []

for case_url in DEMO_CASE_URLS[:MAX_DEMO_CASES]:
    print(f"🚀 Processing demo case: {case_url}")

    try:
        html = fetch_html(case_url)
        case_docs = extract_case_documents(html, case_url)
        print(f"✅ Extracted {len(case_docs)} sections from {case_url}")
        documents.extend(case_docs)

    except Exception as e:
        logger.warning(f"⚠️ Failed to process demo case {case_url}: {e}")

    print("🔗 Fetching related documents...")
    related_docs = []
    print("⏭️ Skipping related documents for small demo validation.")

  

    print(f"📎 Retrieved {len(related_docs)} related documents.")
all_documents = documents + related_docs

if not all_documents:
    raise RuntimeError(
        "No documents were collected. Check the crawl start URL, request headers, "
        "or Justia response status before creating a vector store."
    )

print("🔪 Performing smart legal-aware chunking...")
docs = smart_legal_chunking(all_documents)

if not docs:
    raise RuntimeError(
        "Documents were collected, but chunking produced zero chunks. "
        "Check extract_case_documents() and smart_legal_chunking()."
    )

print(f"📦 Prepared {len(docs)} chunks for embedding.")

print("🔐 Creating vector store with Ollama embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("rag_vectorstore_justia7")

print("✅ Vector store saved as 'rag_vectorstore_justia7'")


if __name__ == "__main__":
    main()