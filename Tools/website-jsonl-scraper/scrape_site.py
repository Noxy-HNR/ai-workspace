#!/usr/bin/env python3
"""Crawl a website politely and save readable page text as JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura
from bs4 import BeautifulSoup


LOG = logging.getLogger("website_jsonl_scraper")
DEFAULT_UA = "WebsiteJSONLTrainingDataBot/1.0 (+local research crawler)"
SKIP_URL_PARTS = (
    "/login", "/log-in", "/logout", "/sign-in", "/signin", "/register",
    "/signup", "/account", "/cart", "/basket", "/checkout", "/search",
    "/wp-admin", "/oauth", "/auth", "/password", "/subscribe/manage",
)
SKIP_QUERY_KEYS = {"q", "query", "search", "s", "keyword", "keywords", "term"}
SKIP_EXTENSIONS = {
    ".7z", ".aac", ".avi", ".bin", ".bmp", ".bz2", ".css", ".csv",
    ".doc", ".docx", ".eot", ".epub", ".exe", ".flac", ".gif", ".gz",
    ".ico", ".iso", ".jpeg", ".jpg", ".js", ".m4a", ".m4v", ".mov",
    ".mp3", ".mp4", ".mpeg", ".mpg", ".ods", ".odt", ".ogg", ".ogv",
    ".pdf", ".png", ".ppt", ".pptx", ".rar", ".rss", ".svg", ".tar",
    ".tif", ".tiff", ".ttf", ".wav", ".webm", ".webp", ".woff", ".woff2",
    ".xls", ".xlsx", ".xml", ".zip",
}
NON_HTML_TYPES = ("application/pdf", "image/", "audio/", "video/", "font/")


def normalize_url(url: str) -> str | None:
    """Return a stable HTTP(S) URL without a fragment, or None for other schemes."""
    url, _fragment = urldefrag(url.strip())
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError:
        return None
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def host_key(url: str) -> tuple[str, int | None]:
    parsed = urlparse(url)
    return parsed.hostname.lower(), parsed.port


def should_skip_url(url: str) -> bool:
    parsed = urlparse(url)
    lowered_path = parsed.path.lower()
    if any(part in lowered_path for part in SKIP_URL_PARTS):
        return True
    if Path(lowered_path).suffix in SKIP_EXTENSIONS:
        return True
    query_keys = {piece.partition("=")[0].lower() for piece in parsed.query.split("&") if piece}
    return bool(query_keys & SKIP_QUERY_KEYS)


def normalized_text_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_page(html: str, url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(("script", "style", "noscript", "svg", "canvas", "iframe", "nav", "footer", "header", "form")):
        element.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
        output_format="txt",
    )
    if not extracted or len(extracted.strip()) < 80:
        main = soup.find("main") or soup.find("article") or soup.body or soup
        extracted = main.get_text("\n", strip=True)
    text = "\n".join(line.strip() for line in extracted.splitlines() if line.strip())
    return title, text


def fetch_robots(session: requests.Session, start_url: str, timeout: float) -> RobotFileParser:
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser(robots_url)
    try:
        response = session.get(robots_url, timeout=timeout, allow_redirects=False)
        if response.status_code in (404, 410):
            parser.parse([])
        elif response.status_code == 200:
            parser.parse(response.text.splitlines())
        else:
            # If robots.txt cannot be checked, fail closed for this host.
            LOG.warning("robots.txt returned HTTP %s; skipping crawl for safety", response.status_code)
            parser.parse(["User-agent: *", "Disallow: /"])
    except requests.RequestException as exc:
        LOG.warning("Could not retrieve robots.txt (%s); skipping crawl for safety", exc)
        parser.parse(["User-agent: *", "Disallow: /"])
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl readable pages from one website into training-friendly JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("start_url", help="Starting HTTP or HTTPS URL")
    parser.add_argument("-o", "--output", type=Path, default=Path("website_data.jsonl"), help="Output JSONL file")
    parser.add_argument("--max-pages", type=int, default=100, help="Maximum HTML pages to request")
    parser.add_argument("--delay", type=float, default=1.0, help="Minimum seconds between requests to the site")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout in seconds")
    parser.add_argument("--min-chars", type=int, default=200, help="Skip extracted text shorter than this")
    parser.add_argument("--user-agent", default=DEFAULT_UA, help="Crawler user-agent (also used for robots.txt rules)")
    parser.add_argument("--verbose", action="store_true", help="Log each page decision")
    return parser


def crawl(args: argparse.Namespace) -> tuple[int, int]:
    start_url = normalize_url(args.start_url)
    if not start_url:
        raise ValueError("start_url must be an absolute HTTP or HTTPS URL")
    if args.max_pages < 1 or args.delay < 0 or args.timeout <= 0 or args.min_chars < 0:
        raise ValueError("max-pages must be >= 1; delay/min-chars must be >= 0; timeout must be > 0")

    start_host = urlparse(start_url).hostname.lower()
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"})
    robots = fetch_robots(session, start_url, args.timeout)
    queue = deque([start_url])
    queued = {start_url}
    visited: set[str] = set()
    content_hashes: set[str] = set()
    page_count = record_count = 0
    # robots.txt is also a request to this host; include it in the rate limit.
    last_request = time.monotonic()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        while queue and page_count < args.max_pages:
            url = queue.popleft()
            if url in visited or should_skip_url(url):
                continue
            visited.add(url)
            if not robots.can_fetch(args.user_agent, url):
                LOG.info("robots.txt disallows %s", url)
                continue

            wait_for = args.delay - (time.monotonic() - last_request)
            if wait_for > 0:
                time.sleep(wait_for)
            try:
                response = session.get(url, timeout=args.timeout, allow_redirects=False)
            except requests.RequestException as exc:
                LOG.warning("Request failed for %s: %s", url, exc)
                last_request = time.monotonic()
                continue
            last_request = time.monotonic()

            if 300 <= response.status_code < 400:
                location = response.headers.get("Location")
                target = normalize_url(urljoin(url, location)) if location else None
                if target and host_key(target) == host_key(start_url):
                    if target not in queued and not should_skip_url(target):
                        queue.append(target)
                        queued.add(target)
                LOG.info("Redirect HTTP %s: %s", response.status_code, url)
                continue
            if response.status_code != 200:
                LOG.info("HTTP %s: %s", response.status_code, url)
                continue
            if host_key(response.url) != host_key(start_url):
                LOG.info("Skipping cross-host response: %s", response.url)
                continue
            content_type = response.headers.get("Content-Type", "").lower()
            if not ("text/html" in content_type or "application/xhtml+xml" in content_type or not content_type):
                if any(marker in content_type for marker in NON_HTML_TYPES):
                    LOG.info("Skipping non-HTML content: %s", url)
                continue

            page_count += 1
            title, text = extract_page(response.text, url)
            if len(text) >= args.min_chars:
                digest = normalized_text_hash(text)
                if digest not in content_hashes:
                    record = {"url": url, "title": title, "text": text, "content_hash": digest}
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    content_hashes.add(digest)
                    record_count += 1
                    LOG.info("Saved %s (%s chars)", url, len(text))
                else:
                    LOG.info("Duplicate text skipped: %s", url)
            else:
                LOG.info("Too little readable text (%s chars): %s", len(text), url)

            soup = BeautifulSoup(response.text, "html.parser")
            for anchor in soup.find_all("a", href=True):
                candidate = normalize_url(urljoin(url, anchor["href"]))
                if not candidate or candidate in queued or should_skip_url(candidate):
                    continue
                candidate_host = urlparse(candidate).hostname.lower()
                same_host = candidate_host == start_host
                if (same_host and host_key(candidate) == host_key(start_url)):
                    queue.append(candidate)
                    queued.add(candidate)

    session.close()
    return page_count, record_count


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        pages, records = crawl(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Crawled {pages} HTML page(s); wrote {records} unique record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
