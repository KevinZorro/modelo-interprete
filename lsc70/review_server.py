"""Local web page for the T-14 review queue (runs on the reviewer's machine only).

    python -m lsc70.review_server --out data/scraper --refs landmarks_cache.npz

Serves one page on 127.0.0.1. The video is the platform's own embedded player at the
sampled second; nothing is downloaded. Decisions go to reviews.jsonl via review_queue.
"""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from lsc70 import review_queue as rq

PAGE = Path(__file__).with_name("review_ui.html")
EXTRA_LABELS = [str(d) for d in range(10)]


def make_handler(out: Path, labels: list):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/queue":
                payload = {"items": rq.review_items(out), "labels": labels}
                self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            else:
                self._send(404, b"{}")

        def do_POST(self):
            if self.path != "/api/review":
                return self._send(404, b"{}")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                rec = rq.record_review(out, body["sample_id"], body["decision"], body.get("label"),
                                       body.get("reviewer", ""), body.get("notes", ""))
                self._send(200, json.dumps(rec, ensure_ascii=False).encode("utf-8"))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                self._send(400, json.dumps({"error": str(e)}).encode("utf-8"))

        def log_message(self, *args):
            pass

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local review page for scraped landmarks")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refs", type=Path, help="npz for labels and (re)building the queue")
    ap.add_argument("--rebuild", action="store_true", help="recompute review_queue.jsonl first")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    if a.refs and (a.rebuild or not (a.out / "review_queue.jsonl").exists()):
        print({"queued": rq.build(a.out, a.refs)})
    labels = list(EXTRA_LABELS)
    if a.refs:
        labels = rq.load_refs(a.refs)[1] + labels
    srv = HTTPServer(("127.0.0.1", a.port), make_handler(a.out, labels))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"Review page: {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
