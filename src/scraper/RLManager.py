import requests
from fake_useragent import UserAgent
import random
import time
from typing import List, Optional
from urllib.parse import urlparse
from io import StringIO
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RLManager:
    def __init__(self, target_url: str, max_retries: int = 3, base_delay: float = 1.0,
                 proxies: Optional[List[str]] = None, webshare_token: Optional[str] = None):
        """Initialize the rate limit bypasser for web scraping.

        Args:
            target_url (str): The target URL to scrape.
            max_retries (int): Maximum number of retries per request.
            base_delay (float): Base delay between requests in seconds.
        """
        self.target_url = target_url
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.ua = UserAgent()

        if proxies:
            self.proxies = proxies
        elif webshare_token:
            self.proxies = self._fetch_webshare_proxies(webshare_token)
        else:
            self.proxies = self._fetch_free_proxies()

        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.ua.random})

    def _fetch_free_proxies(self) -> List[str]:
        """Fetch a list of free proxies from a public proxy list.

        Returns:
            List[str]: List of proxy URLs (e.g., 'http://ip:port').
        """
        proxy_list_url = 'https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all'
        try:
            response = requests.get(proxy_list_url, timeout=10)
            if response.status_code == 200:
                proxies = [f'http://{proxy}' for proxy in response.text.splitlines() if proxy]
                logger.info(f"Fetched {len(proxies)} free proxies")
                return proxies
            else:
                logger.error(f"Failed to fetch proxies: Status {response.status_code}")
                return []
        except requests.RequestException as e:
            logger.error(f"Error fetching proxies: {e}")
            return []

    def _fetch_webshare_proxies(self, api_token: str) -> List[str]:
        """Fetch proxies from Webshare using API token."""
        headers = {
            'Authorization': f'Token {api_token}'
        }
        url = 'https://proxy.webshare.io/api/v2/proxy/list/?mode=direct&page=1&page_size=25'
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            proxies = []
            for proxy_info in data['results']:
                ip = proxy_info['proxy_address']
                port = proxy_info['port']
                user = proxy_info['username']
                pwd = proxy_info['password']
                proxy_url = f"http://{user}:{pwd}@{ip}:{port}"
                proxies.append(proxy_url)

            logger.info(f"Fetched {len(proxies)} Webshare proxies")
            return proxies
        except Exception as e:
            logger.error(f"Failed to fetch Webshare proxies: {e}")
            return []

    def _get_random_proxy(self) -> Optional[dict]:
        """Select a random proxy from the proxy list.

        Returns:
            Optional[dict]: Proxy dictionary for requests or None if no proxies available.
        """
        if not self.proxies:
            logger.warning("No proxies available")
            return None
        proxy = random.choice(self.proxies)
        return {'http': proxy, 'https': proxy}

    def _rotate_user_agent(self) -> None:
        """Rotate the user agent for the session."""
        new_user_agent = self.ua.random
        self.session.headers.update({'User-Agent': new_user_agent})
        logger.debug(f"Rotated user agent to: {new_user_agent}")

    def _rotate_proxy(self) -> None:
        """Rotate the proxy for the session."""
        proxy = self._get_random_proxy()
        if proxy:
            self.session.proxies.update(proxy)
            logger.debug(f"Rotated proxy to: {proxy}")

    def scrape(self, params: Optional[dict] = None) -> Optional[requests.Response]:
        """Scrape the target URL with rate limit bypassing.

        Args:
            params (dict, optional): Query parameters for the GET request.

        Returns:
            Optional[requests.Response]: Response object if successful, None otherwise.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                self._rotate_user_agent()
                self._rotate_proxy()

                # Add random delay to avoid detection
                delay = self.base_delay * (1 + random.uniform(0, 0.5))
                logger.debug(f"Waiting {delay:.2f} seconds before request")
                time.sleep(delay)

                response = self.session.get(self.target_url, params=params, timeout=10)

                if response.status_code == 200:
                    logger.info(f"Successfully scraped {self.target_url}")
                    return response
                elif response.status_code == 429:
                    logger.warning(f"Rate limit hit (429). Retrying after delay...")
                    retries += 1
                    time.sleep(delay * (2 ** retries))  # Exponential backoff
                else:
                    logger.error(f"Request failed with status {response.status_code}")
                    retries += 1

            except requests.RequestException as e:
                logger.error(f"Request error: {e}")
                retries += 1
                time.sleep(delay * (2 ** retries))

        logger.error(f"Max retries ({self.max_retries}) reached for {self.target_url}")
        return None

    def read_html(self, params: Optional[dict] = None) -> Optional[List[pd.DataFrame]]:
        """Scrape the target URL and parse HTML tables using pandas.

        Args:
            params (dict, optional): Query parameters for the GET request.

        Returns:
            Optional[List[pd.DataFrame]]: List of DataFrames containing parsed tables, or None if failed.
        """
        response = self.scrape(params)
        if response:
            try:
                tables = pd.read_html(StringIO(response.text))
                logger.info(f"Parsed {len(tables)} HTML tables from {self.target_url}")
                return tables
            except ValueError as e:
                logger.error(f"No tables found in response: {e}")
                return None
            except Exception as e:
                logger.error(f"Error parsing HTML tables: {e}")
                return None
        return None

    def close(self) -> None:
        """Close the session."""
        self.session.close()
        logger.info("Session closed")