import cloudscraper
from bs4 import BeautifulSoup
import re
import os
import time

scraper = cloudscraper.create_scraper()

MAX_RETRIES = 10
RETRY_DELAY = 2  # Initial delay in seconds

def retry_request(func):
    """Decorator to retry a function up to MAX_RETRIES times."""
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
                # If result is None, treat as failure and retry
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * (1)  # Exponential backoff
                    print(f"   ⏳ Attempt {attempt} returned None, retrying in {delay}s...")
                    time.sleep(delay)
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * (1)  # Exponential backoff
                    print(f"   ⚠️ Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"   ❌ All {MAX_RETRIES} attempts failed. Last error: {e}")
        return None
    return wrapper

def sanitize_filename(filename):
    """Sanitize the filename by removing/replacing invalid characters."""
    sanitized = filename.replace('/', '_').replace('\\', '_')
    sanitized = re.sub(r'[<>:"|?*]', '', sanitized)
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    sanitized = sanitized.strip('_')
    if not sanitized.endswith('.pdf'):
        sanitized += '.pdf'
    return sanitized

@retry_request
def fetch_page(url):
    """Fetch a page with retry logic."""
    response = scraper.get(url, timeout=120)
    if response.status_code == 200:
        return response
    raise Exception(f"HTTP {response.status_code}")

def get_pdf_from_putusan_page(url):
    """Fetch the putusan detail page and extract PDF download link."""
    response = fetch_page(url)
    if not response:
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_link = soup.find('a', href=lambda x: x and '/pdf/' in x)
    
    if pdf_link:
        return pdf_link.get('href')
    return None

def download_pdf(pdf_url, filename, output_dir='downloads'):
    """Download PDF and save with the given filename."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        print(f"   ⏭️ Skipped (already exists): {filename}")
        return True
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pdf_response = scraper.get(pdf_url, timeout=120)
            if pdf_response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(pdf_response.content)
                print(f"   ✅ Downloaded: {filename}")
                return True
            else:
                raise Exception(f"HTTP {pdf_response.status_code}")
        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * (1)
                print(f"   ⚠️ Download attempt {attempt} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"   ❌ Failed to download after {MAX_RETRIES} attempts: {filename}")
                return False
    return False

def parse_putusan_html(html_content):
    """Parse HTML and extract putusan entries."""
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_data = []

    main_container = soup.find('div', id='tabs-1')
    if not main_container:
        print("Container tabs-1 tidak ditemukan.")
        return []

    entries = main_container.find_all('div', class_='spost')

    for entry in entries:
        item = {}
        
        title_tag = entry.select_one('strong > a')
        
        if title_tag:
            item['title'] = title_tag.get_text(strip=True)
            item['url'] = title_tag['href']
        else:
            for strong in entry.find_all('strong'):
                if strong.find('a'):
                    item['title'] = strong.find('a').get_text(strip=True)
                    item['url'] = strong.find('a')['href']
                    break
            else:
                item['title'] = "Judul Tidak Ditemukan"
                item['url'] = "#"

        small_divs = entry.find_all('div', class_='small')
        for div in small_divs:
            text = div.get_text(strip=True)
            if "Register" in text:
                item['dates'] = " ".join(text.split())
                break
        
        all_divs = entry.find_all('div')
        for div in all_divs:
            text = div.get_text(strip=True)
            if text.startswith("Tanggal") and "Register" not in text:
                item['summary'] = text
                break
        
        extracted_data.append(item)

    return extracted_data

def download_putusan_pdfs(list_page_url, output_dir='downloads'):
    """Main function to download all PDFs from a putusan list page."""
    print(f"📄 Fetching list page: {list_page_url}")
    
    response = fetch_page(list_page_url)
    if not response:
        print("❌ Failed to fetch list page after all retries")
        return
    
    results = parse_putusan_html(response.content)
    print(f"📋 Found {len(results)} putusan entries\n")
    
    success_count = 0
    fail_count = 0
    
    for i, data in enumerate(results, 1):
        title = data.get('title', 'unknown')
        url = data.get('url', '#')
        
        print(f"[{i}/{len(results)}] Processing: {title}")
        
        if url == '#':
            print("   ⚠️ Skipping - No valid URL")
            fail_count += 1
            continue
        
        pdf_url = get_pdf_from_putusan_page(url)
        
        if pdf_url:
            filename = sanitize_filename(title)
            if download_pdf(pdf_url, filename, output_dir):
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"   ⚠️ No PDF link found")
            fail_count += 1
        
        print("-" * 60)
    
    print(f"\n📊 Summary: {success_count} downloaded, {fail_count} failed")