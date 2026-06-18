import os
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Set
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

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

# === Metadata & Section Extractor ===
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_case_documents(html: str, url: str) -> List[Document]:
    """
    Extract case documents from Supreme Court case HTML.
    
    Args:
        html: The HTML content of the Supreme Court case page
        url: The URL of the case page
    
    Returns:
        A list of Document objects representing different sections of the case
    """
    soup = BeautifulSoup(html, "html.parser")
    documents = []
    
    # Extract the case title
    title_element = soup.find("h1", class_="title")
    title = title_element.get_text(strip=True) if title_element else "Unknown Case"
    
    # Extract basic case metadata from the case-data section
    metadata = {"source": url, "title": title}
    
    case_data_div = soup.find("div", class_="case-data")
    if case_data_div:
        # Extract citation
        citation_element = case_data_div.find("p", class_="citation")
        if citation_element:
            citation_text = citation_element.get_text(strip=True)
            citation_match = re.search(r"\d+\s+U\.S\.\s+\d+", citation_text)
            if citation_match:
                metadata["citation"] = citation_match.group(0)
        
        # Extract docket number
        docket_element = case_data_div.find("p", class_="docket")
        if docket_element:
            docket_text = docket_element.get_text(strip=True)
            docket_match = re.search(r"No\.\s+([\w\-\.]+)", docket_text)
            if docket_match:
                metadata["docket"] = docket_match.group(1)
        
        # Extract decision date
        date_element = case_data_div.find("p", class_="decision-date")
        if date_element:
            date_text = date_element.get_text(strip=True)
            date_match = re.search(r"Decided:\s+([A-Za-z]+ \d+, \d{4})", date_text)
            if date_match:
                metadata["date_decided"] = date_match.group(1)
    
    # If we couldn't get metadata from the case-data div, try alternative methods
    if "citation" not in metadata:
        full_text = soup.get_text(separator="\n", strip=True)
        citation_match = re.search(r"\b\d{1,3}\s+U\.S\.\s+\d{1,4}\b", full_text)
        if citation_match:
            metadata["citation"] = citation_match.group(0)
    
    if "docket" not in metadata:
        full_text = soup.get_text(separator="\n", strip=True)
        docket_match = re.search(r"(No\.|Docket):\s*([\w\-\.]+)", full_text)
        if docket_match:
            metadata["docket"] = docket_match.group(2)
    
    if "date_decided" not in metadata:
        full_text = soup.get_text(separator="\n", strip=True)
        date_match = re.search(r"Decided:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", full_text)
        if date_match:
            metadata["date_decided"] = date_match.group(1)
    
    # Extract case content sections
    content_div = soup.find("div", id="opinion")
    
    if not content_div:
        # Try alternative content containers
        content_div = soup.find("div", class_="opinion")
        if not content_div:
            content_div = soup.find("div", class_="column-center")
            if not content_div:
                content_div = soup
    
    # Try to extract syllabus
    syllabus_section = None
    
    # First, try to find a specific syllabus section tag
    syllabus_header = content_div.find(["h2", "h3", "h4"], 
                                     string=lambda text: text and "SYLLABUS" in text.upper())
    
    if syllabus_header:
        # Extract the syllabus text
        syllabus_text = ""
        element = syllabus_header.next_sibling
        
        while element and not (element.name in ["h2", "h3", "h4"] and 
                              element.get_text(strip=True).upper() != "SYLLABUS"):
            if isinstance(element, str):
                syllabus_text += element
            elif element.name and element.get_text(strip=True):
                syllabus_text += element.get_text() + "\n"
            element = element.next_sibling
        
        if syllabus_text.strip():
            syllabus_section = syllabus_text.strip()
    
    # If we couldn't find a specific syllabus section, try another approach
    if not syllabus_section:
        full_text = content_div.get_text(separator="\n", strip=True)
        if "SYLLABUS" in full_text:
            parts = full_text.split("SYLLABUS", 1)
            if len(parts) > 1:
                ending_markers = ["OPINION", "MR. JUSTICE", "JUSTICE", "MR. CHIEF JUSTICE"]
                syllabus_text = parts[1]
                
                for marker in ending_markers:
                    if marker in syllabus_text:
                        syllabus_text = syllabus_text.split(marker, 1)[0]
                
                syllabus_section = syllabus_text.strip()
    
    # Add syllabus document if found
    if syllabus_section:
        documents.append(Document(
            page_content=syllabus_section,
            metadata={**metadata, "section": "syllabus"}
        ))
        logger.info(f"Found syllabus section for {metadata.get('citation', 'Unknown')}")
    
    # Extract opinions
    opinion_types = [
        {"name": "opinion_majority", "patterns": [
            r"OPINION OF THE COURT", r"OPINION OF (?:MR\. )?JUSTICE", 
            r"(?:MR\. )?CHIEF JUSTICE .+ delivered the opinion of the Court",
            r"(?:MR\. )?JUSTICE .+ delivered the opinion of the Court"
        ]},
        {"name": "opinion_concurrence", "patterns": [
            r"(?:MR\. )?JUSTICE .+ concurring", r"CONCURRING OPINION",
            r"(?:MR\. )?JUSTICE .+, concurring"
        ]},
        {"name": "opinion_dissent", "patterns": [
            r"(?:MR\. )?JUSTICE .+ dissenting", r"DISSENTING OPINION",
            r"(?:MR\. )?JUSTICE .+, dissenting"
        ]}
    ]
    
    opinion_texts = []
    
    # First try to find opinions by HTML structure
    opinion_headers = content_div.find_all(["h2", "h3", "h4"])
    
    for header in opinion_headers:
        header_text = header.get_text(strip=True)
        
        # Check if this header marks the start of an opinion section
        is_opinion_header = False
        for opinion_type in opinion_types:
            for pattern in opinion_type["patterns"]:
                if re.search(pattern, header_text, re.IGNORECASE):
                    is_opinion_header = True
                    break
            if is_opinion_header:
                break
        
        if is_opinion_header:
            # Extract the opinion text
            opinion_text = ""
            element = header.next_sibling
            
            while element and not (element.name in ["h2", "h3", "h4"] and 
                                 any(re.search(p, element.get_text(strip=True), re.IGNORECASE) 
                                     for ot in opinion_types for p in ot["patterns"])):
                if isinstance(element, str):
                    opinion_text += element
                elif element.name and element.get_text(strip=True):
                    opinion_text += element.get_text() + "\n"
                element = element.next_sibling
            
            if opinion_text.strip():
                # Determine opinion type and author
                opinion_metadata = {**metadata}
                for opinion_type in opinion_types:
                    for pattern in opinion_type["patterns"]:
                        if re.search(pattern, header_text, re.IGNORECASE):
                            opinion_metadata["section"] = opinion_type["name"]
                            
                            # Try to extract the justice name
                            justice_match = re.search(r"(?:MR\. )?JUSTICE\s+([A-Z][A-Za-z]+)", 
                                                     header_text, re.IGNORECASE)
                            if justice_match:
                                opinion_metadata["opinion_author"] = justice_match.group(1)
                            
                            break
                    if "section" in opinion_metadata:
                        break
                
                opinion_texts.append({
                    "text": header_text + "\n" + opinion_text.strip(),
                    "metadata": opinion_metadata
                })
    
    # If we couldn't find opinions by HTML structure, try another approach with text parsing
    if not opinion_texts:
        full_text = content_div.get_text(separator="\n", strip=True)
        
        # Create a combined pattern to find all opinion sections
        combined_pattern = "|".join(
            f"({pattern})" for opinion_type in opinion_types for pattern in opinion_type["patterns"]
        )
        
        # Find all start indices of opinions
        opinion_matches = list(re.finditer(f"({combined_pattern})", full_text, re.IGNORECASE))
        
        if opinion_matches:
            # Extract each opinion section
            for i, match in enumerate(opinion_matches):
                start_idx = match.start()
                end_idx = opinion_matches[i+1].start() if i < len(opinion_matches)-1 else len(full_text)
                
                opinion_text = full_text[start_idx:end_idx].strip()
                match_text = match.group(0)
                
                # Determine opinion type and author
                opinion_metadata = {**metadata}
                for opinion_type in opinion_types:
                    for pattern in opinion_type["patterns"]:
                        if re.search(pattern, match_text, re.IGNORECASE):
                            opinion_metadata["section"] = opinion_type["name"]
                            
                            # Try to extract the justice name
                            justice_match = re.search(r"(?:MR\. )?JUSTICE\s+([A-Z][A-Za-z]+)", 
                                                     match_text, re.IGNORECASE)
                            if justice_match:
                                opinion_metadata["opinion_author"] = justice_match.group(1)
                            
                            break
                    if "section" in opinion_metadata:
                        break
                
                opinion_texts.append({
                    "text": opinion_text,
                    "metadata": opinion_metadata
                })
    
    # Add all found opinions to documents
    for opinion in opinion_texts:
        documents.append(Document(
            page_content=opinion["text"],
            metadata=opinion["metadata"]
        ))
        logger.info(f"Found {opinion['metadata'].get('section', 'unknown')} "
                   f"by {opinion['metadata'].get('opinion_author', 'unknown')} "
                   f"for {metadata.get('citation', 'Unknown')}")
    
    # If no documents were found, add the entire page as a single document
    if not documents:
        logger.warning(f"Could not extract specific sections for {url}, using full text")
        documents.append(Document(
            page_content=content_div.get_text(separator="\n", strip=True),
            metadata={**metadata, "section": "full_text"}
        ))
    
    return documents

# === Crawl and collect cases ===
def crawl_site(base_url: str, max_pages: int = 500, delay: float = 1.0) -> List[Document]:
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
    documents = crawl_site(base_url, max_pages=500)

    print("🔪 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    print("🔐 Creating vector store with Ollama embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("rag_vectorstore_justia6")

    print("✅ Vector store saved as 'rag_vectorstore_justia6'")

if __name__ == "__main__":
    main()
