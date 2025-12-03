from __future__ import annotations

import html
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List, Optional

from search_engine import SearchEngine, SearchResult


INDEX_DIR = Path("index_output")
HOST = "127.0.0.1"
PORT = 8000


def render_page(query: str, results: List[SearchResult], elapsed_ms: Optional[float]) -> str:
    """Render a simple HTML search page with results and timing info."""
    safe_query = html.escape(query or "")

    rows = []
    for i, r in enumerate(results, 1):
        url = html.escape(r.url)
        score = f"{r.score:.4f}"
        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><a href=\"{url}\" target=\"_blank\">{url}</a></td>"
            f"<td>{score}</td>"
            f"</tr>"
        )

    table_html = (
        "<table class='results-table'>"
        "<thead><tr><th>#</th><th>URL</th><th>Score</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
        if results
        else "<p class='muted'>No results found.</p>" if query else ""
    )

    timing_html = (
        f"<p class='timing'>Search time: <strong>{elapsed_ms:.2f} ms</strong></p>"
        if elapsed_ms is not None and query
        else ""
    )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ICS Search Engine</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      margin: 0;
      background: #f5f5f7;
      color: #222;
    }}
    .page {{
      max-width: 900px;
      margin: 2rem auto;
      padding: 2rem 2.5rem;
      background: #ffffff;
      box-shadow: 0 8px 24px rgba(0,0,0,0.08);
      border-radius: 12px;
    }}
    h1 {{
      margin-top: 0;
      margin-bottom: 0.5rem;
      font-size: 1.9rem;
    }}
    .subtitle {{
      margin: 0 0 1.5rem 0;
      color: #666;
      font-size: 0.95rem;
    }}
    form {{
      display: flex;
      gap: 0.6rem;
      margin-bottom: 0.75rem;
    }}
    input[type="text"] {{
      flex: 1;
      padding: 0.55rem 0.75rem;
      border-radius: 6px;
      border: 1px solid #ccc;
      font-size: 0.95rem;
    }}
    input[type="text"]:focus {{
      outline: none;
      border-color: #2563eb;
      box-shadow: 0 0 0 2px rgba(37,99,235,0.25);
    }}
    input[type="submit"] {{
      padding: 0.55rem 1.1rem;
      border-radius: 6px;
      border: none;
      background: #2563eb;
      color: #fff;
      font-size: 0.95rem;
      cursor: pointer;
    }}
    input[type="submit"]:hover {{
      background: #1d4ed8;
    }}
    .timing {{
      margin: 0.25rem 0 0.75rem 0;
      color: #444;
      font-size: 0.9rem;
    }}
    .muted {{
      color: #777;
      font-size: 0.9rem;
      margin-top: 0.75rem;
    }}
    .results-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.5rem;
      font-size: 0.92rem;
    }}
    .results-table thead {{
      background: #f3f4f6;
    }}
    .results-table th,
    .results-table td {{
      padding: 0.5rem 0.6rem;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
    }}
    .results-table tr:hover td {{
      background: #f9fafb;
    }}
    .results-table a {{
      color: #2563eb;
      text-decoration: none;
      word-break: break-all;
    }}
    .results-table a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>ICS Search Engine</h1>
    <form method="get" action="/">
      <input type="text" id="q" name="q" placeholder="Try 'cristina lopes' or 'machine learning'..." value="{safe_query}" />
      <input type="submit" value="Search" />
    </form>
    {timing_html}
    {table_html}
  </div>
</body>
</html>
"""


class SearchRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that exposes the search engine via GET /?q=..."""

    search_engine: SearchEngine | None = None

    def do_GET(self) -> None:
        if self.search_engine is None:
            if not INDEX_DIR.exists():
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"ERROR: index_output directory not found. "
                    b"Please build the index before starting the web server."
                )
                return
            self.__class__.search_engine = SearchEngine(INDEX_DIR)

        # Parse query string
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        query = params.get("q", [""])[0]

        elapsed_ms: Optional[float] = None
        if query:
            start = time.perf_counter()
            results = self.search_engine.search(query, top_k=5)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        else:
            results = []

        page = render_page(query, results, elapsed_ms)
        data = page.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None
        """Silence default logging to keep console clean."""
        return


def run_server() -> None:
    if not INDEX_DIR.exists():
        raise SystemExit(
            f"ERROR: Index directory does not exist: {INDEX_DIR}. "
            "Run the index builder first."
        )

    server = HTTPServer((HOST, PORT), SearchRequestHandler)
    print(f"Web search interface running at http://{HOST}:{PORT}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")
    finally:
        # Cleanly close search engine if it was created
        if SearchRequestHandler.search_engine is not None:
            SearchRequestHandler.search_engine.close()
        server.server_close()


if __name__ == "__main__":
    run_server()


