# Website to JSONL scraper

A small Windows-friendly crawler that saves readable HTML page content as one JSON object per line. It is intended as a data collection step; review and curate the resulting text before using it for fine-tuning.

## Install

The project has its own virtual environment at `.venv`, so these packages do not change other Python projects:

```powershell
cd C:\AI\Tools\website-jsonl-scraper
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Replace the start URL with a site you are allowed to crawl. This example saves up to 500 pages to the indicated JSONL file, waits at least 1.5 seconds between requests, and skips short pages:

```powershell
& 'C:\AI\Tools\website-jsonl-scraper\.venv\Scripts\python.exe' 'C:\AI\Tools\website-jsonl-scraper\scrape_site.py' 'https://example.org/' --output 'C:\AI\Finetuning\website-data.jsonl' --max-pages 500 --delay 1.5 --min-chars 200 --verbose
```

Options:

- `start_url` (required): HTTP or HTTPS URL to begin from.
- `--output`, `-o`: destination JSONL path (default: `website_data.jsonl` in the current directory).
- `--max-pages`: maximum HTML pages requested (default: 100).
- `--delay`: minimum wait between requests to the site, including its `robots.txt` request (default: 1 second).
- `--timeout`: request timeout in seconds (default: 20).
- `--min-chars`: minimum extracted text length to save (default: 200).
- `--user-agent`: crawler identity, also used when checking robots rules.
- `--verbose`: show page-level decisions.

The crawler stays on the starting hostname and port, observes `robots.txt`, and fails closed if `robots.txt` cannot be checked. It skips common login, registration, account, cart, checkout, search, media, and document URLs. It does not submit forms or follow cross-host redirects.

Each output line contains `url`, `title`, `text`, and `content_hash`. The hash is SHA-256 over whitespace-normalized, case-folded text and is used to avoid saving duplicate text under different URLs.
