import requests
from bs4 import BeautifulSoup


class WebScraperTool:

    def scrape(self, url: str) -> str:
        raise Exception('Mock exception')
        headers = {
            'User-Agent': 'Chrome/124.0.0.0',
            'Accept-Language': 'en-Us, en'
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.ok:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        else:
            raise Exception(f'Could not retrieve the webpage: {response.status_code}')
