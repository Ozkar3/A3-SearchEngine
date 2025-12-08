import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def check_link_exists(source_url, target_url):
    try:
        response = requests.get(source_url, timeout=10)
        response.raise_for_status()  
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {source_url}: {e}")
        return False

    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=True)

    normalized_target_url = urljoin(target_url, urlparse(target_url).path)

    for link in links:
        href = link.get('href')
        resolved_href = urljoin(source_url, href)
        
        normalized_resolved_href = urljoin(resolved_href, urlparse(resolved_href).path)

        if normalized_resolved_href == normalized_target_url:
            return True

    return False

# test:
if __name__ == "__main__":
    source_page = "https://ics.uci.edu/"
    target_page = "https://ics.uci.edu/2025/11/17/vijay-vazirani-to-be-honored-with-2025-focs-test-of-time-award/"

    if check_link_exists(source_page, target_page):
        print(f"'{source_page}' links to '{target_page}'.")
    else:
        print(f"'{source_page}' does not link to '{target_page}'.")