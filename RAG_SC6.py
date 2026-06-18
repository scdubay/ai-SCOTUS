import os
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Set, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import logging
import aiohttp
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Setup ===
os.environ["USER_AGENT"] = "Mozilla/5.0"
HEADERS = {"User-Agent": os.environ["USER_AGENT"]}

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

###############################


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
    
    Args:
        html: The HTML content of the Supreme Court case page
        url: The URL of the case page
    
    Returns:
        A list of Document objects representing different sections of the case
    """
    soup = BeautifulSoup(html, "html.parser")
    documents = []
    
    # Extract metadata using enhanced function
    metadata = extract_enhanced_metadata(soup, url)
    
    # Ensure we have a title
    if "title" not in metadata and "title_full" not in metadata:
        # Try multiple approaches to find the title
        title_element = soup.find("h1", class_="title") or soup.find("title")
        if title_element:
            metadata["title"] = title_element.get_text(strip=True)
        else:
            metadata["title"] = "Unknown Case"
    elif "title" not in metadata:
        metadata["title"] = metadata.get("title_full")
    
    # Extract full text for processing
    full_text = soup.get_text(separator="\n", strip=True)
    
    # Update metadata with citation network info
    metadata.update(extract_citation_network(soup, full_text))
    
    # Extract related document links
    related_doc_urls = extract_related_documents(soup, url, metadata)
    if related_doc_urls:
        metadata["related_documents"] = related_doc_urls
    
    # Better content div detection with multiple fallbacks
    main_content_candidates = [
        soup.find("div", id="opinion"),
        soup.find("div", class_="opinion"),
        soup.find("div", class_="column-center"),
        soup.find("div", class_="entry-content"),
        soup.find("div", id="content"),
        soup.find("div", class_="content"),
        soup.find("article")
    ]
    
    content_div = next((div for div in main_content_candidates if div is not None), 
                      soup.find("div") if soup.find("div") else soup)
    
    # Try to detect if we're dealing with a historical case (pre-1900)
    is_historical_case = False
    if "date_decided" in metadata:
        match = re.search(r"(\d{4})", metadata["date_decided"])
        if match and int(match.group(1)) < 1900:
            is_historical_case = True
    
    # Adjust patterns based on case era
    base_opinion_types = [
        {"name": "opinion_majority", "patterns": [
            r"OPINION OF THE COURT", 
            r"OPINION OF (?:MR\. )?JUSTICE",
            r"(?:MR\. )?CHIEF JUSTICE .+ delivered the opinion",
            r"(?:MR\. )?JUSTICE .+ delivered the opinion",
            r"PER CURIAM"
        ]},
        {"name": "opinion_concurrence", "patterns": [
            r"(?:MR\. )?JUSTICE .+ concurring", 
            r"CONCURRING OPINION",
            r"(?:MR\. )?JUSTICE .+, concurring"
        ]},
        {"name": "opinion_dissent", "patterns": [
            r"(?:MR\. )?JUSTICE .+ dissenting", 
            r"DISSENTING OPINION",
            r"(?:MR\. )?JUSTICE .+, dissenting"
        ]}
    ]
    
    # Add historical patterns if needed
    if is_historical_case:
        historical_patterns = {
            "opinion_majority": [
                r"The (unanimous )?opinion of the Court",
                r"The Chief Justice delivered the opinion",
                r"BY THE COURT",
                r"The Court was of opinion"
            ],
            "opinion_concurrence": [
                r"(?:Justice|Judge) .+ concurred",
                r"[A-Z][a-z]+, Justice, concurring"
            ],
            "opinion_dissent": [
                r"(?:Justice|Judge) .+ dissented",
                r"[A-Z][a-z]+, Justice, dissenting"
            ]
        }
        
        # Add historical patterns to the base patterns
        for opinion_type in base_opinion_types:
            if opinion_type["name"] in historical_patterns:
                opinion_type["patterns"].extend(historical_patterns[opinion_type["name"]])
    
    # Extract syllabus with multiple approaches
    syllabus_section = None
    
    # 1. Look for syllabus by header
    syllabus_markers = ["SYLLABUS", "Syllabus", "syllabus"]
    for marker in syllabus_markers:
        syllabus_element = content_div.find(lambda tag: tag.name in ["h1", "h2", "h3", "h4", "h5"] and 
                                          tag.get_text(strip=True) == marker)
        if syllabus_element:
            syllabus_text = extract_section_text(syllabus_element)
            if syllabus_text:
                syllabus_section = syllabus_text
                break
    
    # 2. Try text-based detection if header approach failed
    if not syllabus_section:
        for marker in syllabus_markers:
            if marker in full_text:
                parts = full_text.split(marker, 1)
                if len(parts) > 1:
                    syllabus_text = parts[1]
                    
                    # Find end of syllabus by looking for opinion markers
                    end_markers = ["OPINION", "MR. JUSTICE", "JUSTICE", "MR. CHIEF JUSTICE", 
                                  "PER CURIAM", "The Chief Justice"]
                    for end_marker in end_markers:
                        if end_marker in syllabus_text:
                            syllabus_text = syllabus_text.split(end_marker, 1)[0]
                            break
                    
                    syllabus_section = syllabus_text.strip()
                    break
    
    # Add syllabus if found
    if syllabus_section and len(syllabus_section) > 20:  # Ensure minimum meaningful content
        documents.append(Document(
            page_content=syllabus_section,
            metadata={**metadata, "section": "syllabus"}
        ))
        logger.info(f"Found syllabus section for {metadata.get('citation', 'Unknown')}")
    
    # Extract opinions using both HTML structure and text patterns
    opinion_sections = []
    
    # 1. Try HTML structure-based extraction first
    opinion_headers = content_div.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div"], 
                                      class_=lambda c: c and any(t in str(c).lower() for t in 
                                                               ["opinion", "concur", "dissent"]))
    
    # Add any element that matches our patterns
    opinion_headers.extend(content_div.find_all(lambda tag: tag.name in ["h1", "h2", "h3", "h4", "h5", "p", "div"] and
                                             any(re.search(pattern, tag.get_text(strip=True), re.IGNORECASE) 
                                                 for ot in base_opinion_types for pattern in ot["patterns"])))
    
    # Remove duplicates while preserving order
    seen = set()
    opinion_headers = [x for x in opinion_headers if not (x in seen or seen.add(x))]
    
    for header in opinion_headers:
        header_text = header.get_text(strip=True)
        opinion_type_match = None
        
        # Determine opinion type
        for opinion_type in base_opinion_types:
            if any(re.search(pattern, header_text, re.IGNORECASE) for pattern in opinion_type["patterns"]):
                opinion_type_match = opinion_type["name"]
                break
        
        if opinion_type_match:
            # Extract opinion text using better section extraction
            opinion_text = extract_section_text(header)
            
            if opinion_text and len(opinion_text) > 50:  # Ensure minimum meaningful content
                # Extract justice name if possible
                justice_name = None
                justice_patterns = [
                    r"(?:MR\. )?JUSTICE\s+([A-Z][A-Za-z]+)",
                    r"(?:MR\. )?CHIEF JUSTICE\s+([A-Z][A-Za-z]+)",
                    r"([A-Z][A-Za-z]+),\s+(?:Justice|Chief Justice)"
                ]
                
                for pattern in justice_patterns:
                    match = re.search(pattern, header_text, re.IGNORECASE)
                    if match:
                        justice_name = match.group(1)
                        break
                
                opinion_metadata = {
                    **metadata,
                    "section": opinion_type_match
                }
                
                if justice_name:
                    opinion_metadata["opinion_author"] = justice_name
                
                opinion_sections.append({
                    "text": header_text + "\n" + opinion_text,
                    "metadata": opinion_metadata
                })
    
    # 2. If no opinions found, try text pattern-based extraction
    if not opinion_sections:
        # Build combined pattern for all opinion types
        combined_patterns = []
        for opinion_type in base_opinion_types:
            for pattern in opinion_type["patterns"]:
                combined_patterns.append(f"({pattern})")
        
        combined_regex = "|".join(combined_patterns)
        opinion_matches = list(re.finditer(combined_regex, full_text, re.IGNORECASE))
        
        if opinion_matches:
            for i, match in enumerate(opinion_matches):
                start_idx = match.start()
                end_idx = opinion_matches[i+1].start() if i < len(opinion_matches)-1 else len(full_text)
                
                opinion_text = full_text[start_idx:end_idx].strip()
                match_text = match.group(0)
                
                if len(opinion_text) > 50:  # Ensure minimum meaningful content
                    # Determine opinion type
                    opinion_type_match = None
                    for opinion_type in base_opinion_types:
                        if any(re.search(pattern, match_text, re.IGNORECASE) for pattern in opinion_type["patterns"]):
                            opinion_type_match = opinion_type["name"]
                            break
                    
                    # Extract justice name if possible
                    justice_name = None
                    justice_patterns = [
                        r"(?:MR\. )?JUSTICE\s+([A-Z][A-Za-z]+)",
                        r"(?:MR\. )?CHIEF JUSTICE\s+([A-Z][A-Za-z]+)",
                        r"([A-Z][A-Za-z]+),\s+(?:Justice|Chief Justice)"
                    ]
                    
                    for pattern in justice_patterns:
                        name_match = re.search(pattern, match_text, re.IGNORECASE)
                        if name_match:
                            justice_name = name_match.group(1)
                            break
                    
                    opinion_metadata = {
                        **metadata,
                        "section": opinion_type_match or "unknown_opinion"
                    }
                    
                    if justice_name:
                        opinion_metadata["opinion_author"] = justice_name
                    
                    opinion_sections.append({
                        "text": opinion_text,
                        "metadata": opinion_metadata
                    })
    
    # Add all found opinions to documents
    for opinion in opinion_sections:
        documents.append(Document(
            page_content=opinion["text"],
            metadata=opinion["metadata"]
        ))
        logger.info(f"Found {opinion['metadata'].get('section', 'unknown')} " +
                   f"by {opinion['metadata'].get('opinion_author', 'unknown')} " +
                   f"for {metadata.get('citation', 'Unknown')}")
    
    # If no specific sections were found, use smarter fallback approaches
    if not documents:
        # Try structural division if possible
        main_divs = content_div.find_all("div", recursive=False)
        if len(main_divs) > 1:
            # The page might have logical divisions we can use
            for i, div in enumerate(main_divs):
                div_text = div.get_text(strip=True)
                if len(div_text) > 100:  # Ensure minimum meaningful content
                    section_name = f"section_{i+1}"
                    
                    # Try to determine section type from content
                    if i == 0 and ("syllabus" in div_text.lower() or "summary" in div_text.lower()):
                        section_name = "syllabus"
                    elif any(term in div_text.lower() for term in ["opinion", "court", "justice"]):
                        section_name = "opinion_section"
                    
                    documents.append(Document(
                        page_content=div_text,
                        metadata={**metadata, "section": section_name}
                    ))
            
            if documents:
                logger.info(f"Used structural division for {metadata.get('citation', 'Unknown')}")
        
        # If still no documents, use paragraphs as a last resort
        if not documents:
            paragraphs = content_div.find_all("p")
            if paragraphs and len(paragraphs) > 0:
                # Combine paragraphs into a reasonable document structure
                current_section = []
                current_length = 0
                max_section_length = 5000
                
                for p in paragraphs:
                    p_text = p.get_text(strip=True)
                    if p_text:
                        current_section.append(p_text)
                        current_length += len(p_text)
                        
                        # Create document when section reaches target length
                        if current_length >= max_section_length:
                            section_text = "\n\n".join(current_section)
                            documents.append(Document(
                                page_content=section_text,
                                metadata={**metadata, "section": f"section_{len(documents)+1}"}
                            ))
                            current_section = []
                            current_length = 0
                
                # Add any remaining content
                if current_section:
                    section_text = "\n\n".join(current_section)
                    documents.append(Document(
                        page_content=section_text,
                        metadata={**metadata, "section": f"section_{len(documents)+1}"}
                    ))
                
                if documents:
                    logger.info(f"Used paragraph-based division for {metadata.get('citation', 'Unknown')}")
            
            # Absolute last resort: full text
            if not documents:
                logger.warning(f"Could not extract specific sections for {url}, using full text")
                documents.append(Document(
                    page_content=content_div.get_text(separator="\n", strip=True),
                    metadata={**metadata, "section": "full_text"}
                ))
    
    return documents

def extract_section_text(header_element):
    """
    Extract text from a section starting with the given header element.
    Uses multiple strategies to extract the complete section content.
    """
    # Strategy 1: Use next siblings until another header or section boundary
    section_text = ""
    element = header_element.next_sibling
    stop_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
    stop_texts = ["OPINION", "SYLLABUS", "DISSENT", "CONCUR"]
    
    while element:
        # Stop if we hit a header element
        if element.name in stop_tags:
            # Check if this header marks the beginning of a new section
            if any(stop_text in element.get_text(strip=True).upper() for stop_text in stop_texts):
                break
        
        # Extract text from this element
        if isinstance(element, str):
            section_text += element
        elif element.name and element.get_text(strip=True):
            section_text += element.get_text() + "\n"
        
        element = element.next_sibling
    
    # If the previous approach yielded insufficient content, try parent-based extraction
    if len(section_text.strip()) < 100:
        # Strategy 2: Find the parent container and extract all text after the header
        parent = header_element.parent
        if parent:
            full_parent_text = parent.get_text(separator="\n", strip=True)
            header_text = header_element.get_text(strip=True)
            
            # Find position of header in parent text
            header_pos = full_parent_text.find(header_text)
            if header_pos >= 0:
                section_text = full_parent_text[header_pos + len(header_text):].strip()
    
    return section_text.strip()
# === Crawl and collect cases ===
def crawl_site(base_url: str, max_pages: int = 50, delay: float = 1.0) -> List[Document]:
    visited: Set[str] = set()
    to_visit: List[str] = [base_url]
    documents: List[Document] = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue
        try:
            print(f"🔍 Visiting: {url}")
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
            visited.add(url)

            if "https://supreme.justia.com/cases/federal/" in url:
                docs = extract_case_documents(res.text, url)
                documents.extend(docs)

            soup = BeautifulSoup(res.text, "html.parser")
            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link["href"])
                if (
                    full_url.startswith(base_url) or
                    full_url.startswith("https://supreme.justia.com/cases/federal/")
                ) and full_url not in visited:
                    to_visit.append(full_url)

            time.sleep(delay)
        except Exception as e:
            print(f"⚠️ Error with {url}: {e}")

    print(f"✅ Finished crawling {len(documents)} case sections.")
    return documents

# === Main process ===
def main():
    base_url = "https://supreme.justia.com/cases-by-topic/"
    print(f"🚀 Starting crawl from: {base_url}")
    documents = crawl_site(base_url, max_pages=50)

    print("🔗 Fetching related documents...")
    related_docs = []

    async def process_related_docs():
        nonlocal related_docs
        for doc in documents:
            if "related_documents" in doc.metadata:
                fetched = await fetch_all_related_documents(
                    doc.metadata["related_documents"],
                    doc.metadata
                )
                related_docs.extend(fetched)

    asyncio.run(process_related_docs())

    print(f"📎 Retrieved {len(related_docs)} related documents.")
    all_documents = documents + related_docs

    print("🔪 Performing smart legal-aware chunking...")
    docs = smart_legal_chunking(all_documents)

    print("🔐 Creating vector store with Ollama embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("rag_vectorstore_justia6")

    print("✅ Vector store saved as 'rag_vectorstore_justia6'")


if __name__ == "__main__":
    main()