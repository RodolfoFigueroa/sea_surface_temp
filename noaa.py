"""Explore the NOAA OISST archive independently of the CMIP6 notebook."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import xarray as xr
from bs4 import BeautifulSoup

ROOT_URL = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/"
    "v2.1/access/avhrr/"
)
EXAMPLE_URL = f"{ROOT_URL}198109/oisst-avhrr-v02r01.19810901.nc"


def get_all_urls(root_url: str) -> list[str]:
    """Return child archive links, excluding parents and off-site links."""
    response = requests.get(root_url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    urls: list[str] = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if not isinstance(href, str) or "Parent" in link.get_text():
            continue
        url = urljoin(root_url, href)
        if url.startswith(root_url) and url != root_url:
            urls.append(url)
    return list(dict.fromkeys(urls))


def build_url_table(urls: list[str]) -> pd.DataFrame:
    """Retain OISST NetCDF links and the date token in their filenames."""
    records = []
    for url in urls:
        filename = urlparse(url).path.rsplit("/", 1)[-1]
        if filename.startswith("oisst-avhrr-v02r01.") and filename.endswith(".nc"):
            records.append({"url": url, "date": filename.split(".")[1]})
    return pd.DataFrame(records, columns=["url", "date"])


def discover_archive(root_url: str = ROOT_URL) -> pd.DataFrame:
    """Enumerate the archive; this can make many HTTP requests."""
    all_urls: list[str] = []
    for month_url in get_all_urls(root_url):
        directory = urlparse(month_url).path.rstrip("/").rsplit("/", 1)[-1]
        if re.fullmatch(r"\d{6}", directory):
            all_urls.extend(get_all_urls(f"{month_url.rstrip('/')}/"))
    return build_url_table(all_urls)


def open_example(url: str = EXAMPLE_URL) -> xr.Dataset:
    """Download and load an OISST sample without retaining an open file."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with xr.open_dataset(response.content) as dataset:
        return dataset.load()


def main() -> tuple[pd.DataFrame, xr.Dataset]:
    """Return the archive table and sample; run with Python -i to explore them."""
    return discover_archive(), open_example()


if __name__ == "__main__":
    df_urls, ds = main()
