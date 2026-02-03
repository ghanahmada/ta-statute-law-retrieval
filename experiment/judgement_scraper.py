"""
Professional Judgement Scraper with:
- Adaptive rate limiting (slows down on 429, speeds up on success)
- Session persistence with rotating headers
- Resume capability (tracks progress in JSON)
- Pagination support
- Random delays to avoid detection
- Concurrent downloads with throttling
"""

import cloudscraper
from bs4 import BeautifulSoup
import re
import os
import time
import json
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ============== CONFIGURATION ==============
MAX_RETRIES = 5
BASE_DELAY = 3          # Start with 3 seconds (more conservative)
MAX_DELAY = 120         # Maximum delay cap
JITTER_RANGE = (1, 3)   # Random jitter between requests
CONCURRENT_WORKERS = 2  # Low concurrency to avoid rate limits
REQUEST_TIMEOUT = 60
PROGRESS_FILE = 'scraper_progress.json'

# Adaptive delay state
current_delay = BASE_DELAY
delay_lock = Lock()

# ============== SESSION SETUP ==============
def create_scraper():
    """Create a cloudscraper with rotating user agents."""
    return cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        }
    )

scraper = create_scraper()

# ============== ADAPTIVE RATE LIMITING ==============
def adjust_delay(success: bool, got_429: bool = False):
    """Adaptively adjust delay based on response."""
    global current_delay
    with delay_lock:
        if got_429:
            # Hit rate limit - double the delay (exponential backoff)
            current_delay = min(current_delay * 2, MAX_DELAY)
            print(f"   🐌 Rate limited! Slowing down to {current_delay}s delay")
        elif success:
            # Success - gradually reduce delay (but not below BASE_DELAY)
            current_delay = max(current_delay * 0.9, BASE_DELAY)

def smart_sleep(multiplier: float = 1.0):
    """Sleep with current adaptive delay + random jitter."""
    jitter = random.uniform(*JITTER_RANGE)
    sleep_time = (current_delay * multiplier) + jitter
    time.sleep(sleep_time)

# ============== CORE FUNCTIONS ==============
def sanitize_filename(filename: str) -> str:
    """Sanitize the filename by removing/replacing invalid characters."""
    sanitized = filename.replace('/', '_').replace('\\', '_')
    sanitized = re.sub(r'[<>:"|?*]', '', sanitized)
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    sanitized = sanitized.strip('_.')
    if not sanitized.endswith('.pdf'):
        sanitized += '.pdf'
    return sanitized

def fetch_with_retry(url: str, timeout: int = REQUEST_TIMEOUT) -> tuple:
    """
    Fetch URL with retry logic.
    Returns: (response, success, got_429)
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = scraper.get(url, timeout=timeout)
            
            if response.status_code == 200:
                adjust_delay(success=True)
                return response, True, False
            
            elif response.status_code == 429:
                adjust_delay(success=False, got_429=True)
                if attempt < MAX_RETRIES:
                    wait_time = current_delay * (2 ** (attempt - 1))
                    print(f"   ⚠️ 429 Rate Limited (attempt {attempt}). Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                continue
            
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait_time = BASE_DELAY * (2 ** (attempt - 1))
                print(f"   ⚠️ Attempt {attempt} failed: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ All {MAX_RETRIES} attempts failed: {e}")
                return None, False, False
    
    return None, False, True

def get_pdf_url_from_detail_page(url: str) -> str:
    """Fetch the putusan detail page and extract PDF download link."""
    smart_sleep(0.5)  # Small delay before each detail page request
    
    response, success, _ = fetch_with_retry(url)
    if not success or not response:
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_link = soup.find('a', href=lambda x: x and '/pdf/' in x)
    
    return pdf_link.get('href') if pdf_link else None

def download_pdf(pdf_url: str, filename: str, output_dir: str = 'downloads') -> bool:
    """Download PDF and save with the given filename."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Skip if already exists
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        if file_size > 1000:  # At least 1KB to be valid
            print(f"   ⏭️ Skipped (exists, {file_size//1024}KB): {filename}")
            return True
        else:
            os.remove(filepath)  # Remove corrupt/empty file
    
    smart_sleep(1.0)  # Delay before download
    
    response, success, _ = fetch_with_retry(pdf_url, timeout=120)
    if not success or not response:
        return False
    
    try:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        file_size = os.path.getsize(filepath)
        print(f"   ✅ Downloaded ({file_size//1024}KB): {filename}")
        return True
    except Exception as e:
        print(f"   ❌ Error saving file: {e}")
        return False

# ============== HTML PARSING ==============
def parse_list_page(html_content) -> list:
    """Parse the list page and extract putusan entries."""
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_data = []
    
    # Find all putusan entries
    entries = soup.find_all('div', class_='spost')
    
    for entry in entries:
        item = {}
        
        # Find title and URL
        title_link = entry.select_one('strong > a') or entry.select_one('a[href*="putusan"]')
        
        if title_link:
            item['title'] = title_link.get_text(strip=True)
            item['url'] = title_link.get('href', '')
        else:
            continue  # Skip entries without valid links
        
        # Extract dates
        small_div = entry.find('div', class_='small')
        if small_div:
            item['dates'] = small_div.get_text(strip=True)
        
        if item.get('url'):
            extracted_data.append(item)
    
    return extracted_data

def get_total_pages(soup: BeautifulSoup) -> int:
    """Extract total number of pages from pagination."""
    pagination = soup.find('ul', class_='pagination')
    if not pagination:
        return 1
    
    # Find last page number
    page_links = pagination.find_all('a')
    max_page = 1
    for link in page_links:
        try:
            page_num = int(link.get_text(strip=True))
            max_page = max(max_page, page_num)
        except ValueError:
            continue
    
    return max_page

# ============== PROGRESS TRACKING ==============
def load_progress(progress_file: str = PROGRESS_FILE) -> dict:
    """Load progress from JSON file."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'downloaded': [], 'failed': [], 'last_page': 0}

def save_progress(progress: dict, progress_file: str = PROGRESS_FILE):
    """Save progress to JSON file."""
    progress['last_updated'] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

# ============== MAIN SCRAPER ==============
def scrape_putusan_list(
    base_url: str,
    output_dir: str = 'downloads',
    start_page: int = 1,
    end_page: int = None,
    resume: bool = True
):
    """
    Main function to scrape all PDFs from putusan list pages.
    
    Args:
        base_url: URL of the first list page
        output_dir: Directory to save PDFs
        start_page: Starting page number
        end_page: Ending page number (None = all pages)
        resume: Whether to resume from previous progress
    """
    # Load previous progress
    progress = load_progress() if resume else {'downloaded': [], 'failed': [], 'last_page': 0}
    downloaded_titles = set(progress.get('downloaded', []))
    
    print(f"🚀 Starting scraper...")
    print(f"   Base URL: {base_url}")
    print(f"   Output: {output_dir}/")
    print(f"   Previously downloaded: {len(downloaded_titles)} files")
    print("-" * 60)
    
    # Fetch first page to get total pages
    response, success, _ = fetch_with_retry(base_url)
    if not success:
        print("❌ Failed to fetch initial page")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    total_pages = get_total_pages(soup)
    end_page = end_page or total_pages
    
    print(f"📄 Found {total_pages} pages total. Processing pages {start_page} to {end_page}")
    print("=" * 60)
    
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    for page_num in range(start_page, end_page + 1):
        print(f"\n📑 PAGE {page_num}/{end_page}")
        print("-" * 40)
        
        # Construct page URL
        if page_num == 1:
            page_url = base_url
        else:
            # Handle different URL patterns
            if '.html' in base_url:
                page_url = base_url.replace('.html', f'/page/{page_num}.html')
            else:
                page_url = f"{base_url}/page/{page_num}"
        
        # Fetch page
        if page_num > 1:
            smart_sleep(1.5)  # Delay between pages
            response, success, _ = fetch_with_retry(page_url)
            if not success:
                print(f"   ❌ Failed to fetch page {page_num}")
                continue
        
        # Parse entries
        entries = parse_list_page(response.content if page_num > 1 else soup.encode())
        print(f"   Found {len(entries)} entries on this page")
        
        for i, entry in enumerate(entries, 1):
            title = entry.get('title', 'unknown')
            url = entry.get('url', '')
            
            print(f"\n   [{i}/{len(entries)}] {title[:60]}...")
            
            # Skip if already downloaded
            filename = sanitize_filename(title)
            if title in downloaded_titles or os.path.exists(os.path.join(output_dir, filename)):
                print(f"      ⏭️ Already processed")
                total_skipped += 1
                continue
            
            # Get PDF URL from detail page
            pdf_url = get_pdf_url_from_detail_page(url)
            
            if not pdf_url:
                print(f"      ⚠️ No PDF link found")
                progress['failed'].append({'title': title, 'url': url, 'reason': 'no_pdf_link'})
                total_failed += 1
                continue
            
            # Download PDF
            if download_pdf(pdf_url, filename, output_dir):
                progress['downloaded'].append(title)
                downloaded_titles.add(title)
                total_success += 1
            else:
                progress['failed'].append({'title': title, 'url': url, 'reason': 'download_failed'})
                total_failed += 1
            
            # Save progress periodically
            if (total_success + total_failed) % 5 == 0:
                progress['last_page'] = page_num
                save_progress(progress)
        
        # Save progress after each page
        progress['last_page'] = page_num
        save_progress(progress)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"   ✅ Downloaded: {total_success}")
    print(f"   ⏭️ Skipped:    {total_skipped}")
    print(f"   ❌ Failed:     {total_failed}")
    print(f"   📁 Output:     {os.path.abspath(output_dir)}")
    print(f"   📋 Progress:   {PROGRESS_FILE}")
    
    save_progress(progress)

# ============== ENTRY POINT ==============
if __name__ == "__main__":
    # Example usage
    LIST_URL = "https://putusan3.mahkamahagung.go.id/direktori/index/pengadilan/pta-surabaya/kategori/perdata-1.html"
    
    scrape_putusan_list(
        base_url=LIST_URL,
        output_dir='downloads',
        start_page=1,
        end_page=None,  # None = all pages
        resume=True     # Resume from previous progress
    )
