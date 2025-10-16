# file: wp_rest_enum.py
import argparse
import csv
import json
import logging
import os
import sys
import threading
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings()

try:
    import pandas as pd  # for CSV/ODS
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# status bar
try:
    from tqdm import tqdm  # status bar
    from tqdm.contrib.logging import TqdmLoggingHandler, logging_redirect_tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False
    tqdm = None  # type: ignore
    TqdmLoggingHandler = None  # type: ignore
    # no-op context manager when tqdm is absent
    def logging_redirect_tqdm():  # type: ignore
        return contextlib.nullcontext()

try:
    from colorama import Fore, Style, init as colorama_init  # colors
    colorama_init(autoreset=True)
    HAS_COLOR = True
except Exception:
    HAS_COLOR = False

HEADERS = {"User-Agent": "WordPress REST Enumerator"}
DEFAULT_TIMEOUT = 10

# Colored Logging
class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not HAS_COLOR:
            return msg
        if record.levelno >= logging.ERROR:
            return f"{Fore.RED}{msg}{Style.RESET_ALL}"
        if record.levelno >= logging.WARNING:
            return f"{Fore.YELLOW}{msg}{Style.RESET_ALL}"
        if record.levelno == logging.INFO:
            return f"{Fore.CYAN}{msg}{Style.RESET_ALL}"
        return msg

def setup_logging(level: str) -> None:
    """
    Use a TQDM-aware logging handler so log lines don't break the single progress bar.
    Falls back to a normal StreamHandler if tqdm is unavailable.
    """
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    if HAS_TQDM and TqdmLoggingHandler is not None:
        handler = TqdmLoggingHandler()  # writes via tqdm.write
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)

# HTTP 
def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s = requests.Session()
    s.headers.update(HEADERS)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

# Input normalization
def normalize_url(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return None
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    if not parsed.netloc:
        return None
    path = parsed.path.rstrip("/")
    norm = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return norm.rstrip("/")

def read_txt(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n = normalize_url(line)
            if n:
                out.append(n)
    return out

def _detect_url_column(headers: List[str]) -> Optional[str]:
    lowered = [str(h).lower() for h in headers]
    for cand in ("url", "urls", "website", "site", "host", "domain"):
        if cand in lowered:
            return headers[lowered.index(cand)]
    return None

def read_csv_builtin(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return urls
        col = _detect_url_column(reader.fieldnames) or reader.fieldnames[0]
        for row in reader:
            n = normalize_url(str(row.get(col, "")).strip())
            if n:
                urls.append(n)
    return urls

def read_tabular_with_pandas(path: str) -> List[str]:
    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, engine=None)  # ODS needs odfpy installed
    except Exception as e:
        logging.error("Failed to read '%s' via pandas: %s", path, e)
        return []
    if df.empty:
        return []
    col = None
    for c in df.columns:
        if str(c).lower() in ("url", "urls", "website", "site", "host", "domain"):
            col = c
            break
    if col is None:
        col = df.columns[0]
    urls: List[str] = []
    for val in df[col].astype(str).tolist():
        n = normalize_url(val)
        if n:
            urls.append(n)
    return urls


def read_targets(website: Optional[str], input_file: Optional[str]) -> List[str]:
    targets: List[str] = []
    if website:
        n = normalize_url(website)
        if n:
            targets.append(n)
    if input_file:
        if not os.path.exists(input_file):
            logging.error("Input file not found: %s", input_file)
            sys.exit(2)
        ext = os.path.splitext(input_file)[1].lower()
        if ext in (".txt", ".list"):
            targets.extend(read_txt(input_file))
        elif ext == ".csv":
            targets.extend(read_tabular_with_pandas(input_file) if HAS_PANDAS else read_csv_builtin(input_file))
        elif ext in (".ods",):
            if not HAS_PANDAS:
                logging.error("Reading .ods requires pandas + odfpy. Install: pip install pandas odfpy")
                sys.exit(2)
            targets.extend(read_tabular_with_pandas(input_file))
        else:
            logging.error("Unsupported input extension: %s", ext)
            sys.exit(2)
    # dedupe, preserve order
    seen = set()
    unique = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique

# WP enumeration
def is_wordpress(base_url: str, session: requests.Session) -> Tuple[bool, Optional[dict]]:
    try:
        resp = session.get(f"{base_url}/wp-json/", timeout=DEFAULT_TIMEOUT, verify=False)
    except requests.RequestException:
        return False, None
    if resp.status_code != 200:
        return False, None
    try:
        data = resp.json()
    except ValueError:
        return False, None
    routes = data.get("routes") or {}
    has_v2 = any("/wp/v2" in k for k in routes.keys())
    return bool(routes) and has_v2, data

def paginate(session: requests.Session, url: str) -> List[dict]:
    per_page = 100
    page = 1
    out: List[dict] = []
    while True:
        try:
            resp = session.get(
                f"{url}?per_page={per_page}&page={page}",
                timeout=DEFAULT_TIMEOUT,
                verify=False,
            )
        except requests.RequestException:
            break
        if resp.status_code != 200:
            break
        try:
            payload = resp.json()
        except ValueError:
            break
        if not isinstance(payload, list) or not payload:
            break
        out.extend(payload)
        page += 1
    return out

def extract_extension(url: Optional[str], mime: Optional[str]) -> Optional[str]:
    """Infer a file extension from source_url or mime_type; returns lowercase ext without dot."""
    if url:
        try:
            p = urlparse(url)
            path = p.path
            if "." in os.path.basename(path):
                ext = os.path.basename(path).split(".")[-1].lower()
                ext = ext.split("?")[0].split("#")[0]
                alias = {
                    "jpeg": "jpg",
                    "tif": "tiff",
                    "htm": "html",
                }
                return alias.get(ext, ext)
        except Exception:
            pass
    if mime:
        mime = mime.lower()
        mapping = {
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp",
            "image/svg+xml": "svg",
            "image/tiff": "tiff",
            "application/pdf": "pdf",
            "application/zip": "zip",
            "application/x-zip-compressed": "zip",
            "application/json": "json",
            "application/msword": "doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.ms-excel": "xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "application/vnd.ms-powerpoint": "ppt",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
            "audio/mpeg": "mp3",
            "audio/mp4": "m4a",
            "audio/ogg": "ogg",
            "video/mp4": "mp4",
            "video/quicktime": "mov",
            "video/x-msvideo": "avi",
            "text/plain": "txt",
            "text/html": "html",
            "application/x-shockwave-flash": "swf",
        }
        return mapping.get(mime)
    return None


def fetch_media(base_url: str, session: requests.Session) -> Tuple[List[str], Dict[str, int]]:
    """Return list of media URLs plus extension counts."""
    api_base = f"{base_url}/wp-json/wp/v2/media"
    rows = paginate(session, api_base)
    urls: List[str] = []
    counts: Dict[str, int] = {}
    for row in rows:
        src = row.get("source_url") or row.get("guid", {}).get("rendered") or row.get("link")
        if src:
            urls.append(src)
        mime = row.get("mime_type")
        ext = extract_extension(src, mime)
        if ext:
            counts[ext] = counts.get(ext, 0) + 1
    return urls, counts


def fetch_collection(base_url: str, coll: str, session: requests.Session) -> List:
    api_base = f"{base_url}/wp-json/wp/v2/{coll}"
    rows = paginate(session, api_base)
    results: List = []
    if coll == "users":
        for row in rows:
            name = row.get("name")
            slug = row.get("slug")
            if name and slug:
                results.append({"name": name, "username": slug})
        return results
    for row in rows:
        guid = None
        if isinstance(row.get("guid"), dict):
            guid = row.get("guid", {}).get("rendered")
        if not guid:
            guid = row.get("link")
        if guid:
            results.append(guid)
    return results


def enumerate_site(
    base_url: str,
    do_posts: bool,
    do_pages: bool,
    do_comments: bool,
    do_media: bool,
    do_users: bool,
) -> Dict[str, List]:
    s = build_session()  # separate session per worker
    is_wp, _ = is_wordpress(base_url, s)
    if not is_wp:
        logging.warning("Not a detectable WordPress site (or blocked): %s", base_url)
        return {}
    logging.info("Enumerating: %s", base_url)
    out: Dict[str, List] = {}
    try:
        if do_posts:
            out["posts"] = fetch_collection(base_url, "posts", s)
        if do_pages:
            out["pages"] = fetch_collection(base_url, "pages", s)
        if do_comments:
            out["comments"] = fetch_collection(base_url, "comments", s)
        if do_media:
            media_urls, ext_counts = fetch_media(base_url, s)
            out["media"] = media_urls
            out["media_extension_counts"] = ext_counts  # per-site counts
        if do_users:
            out["users"] = fetch_collection(base_url, "users", s)
    except Exception as e:
        logging.error("Enumeration failed for %s: %s", base_url, e)
    return out


# State & IO
def atomic_write_json(path: str, data: dict, pretty: bool = True) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2 if pretty else None)
    os.replace(tmp, path)

def load_state(state_path: str) -> Tuple[Set[str], List[str]]:
    if not os.path.exists(state_path):
        return set(), []
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        completed = set(obj.get("completed", []))
        remaining = obj.get("remaining", [])
        return completed, remaining
    except Exception as e:
        logging.warning("Failed to load state '%s': %s; starting fresh.", state_path, e)
        return set(), []

def save_state(state_path: str, completed: Set[str], remaining: List[str]) -> None:
    payload = {
        "completed": sorted(list(completed)),
        "remaining": remaining,
        "version": 3,
    }
    atomic_write_json(state_path, payload, pretty=True)


#  Split output #
class SplitOutputWriter:
    """
    Thread-safe JSONL writer per collection.
    media.jsonl line: {"site": "<url>", "item": "<media_url>"}
    media_extension_counts.jsonl line: {"site": "<url>", "counts": {...}}
    """
    def __init__(self, directory: Optional[str]) -> None:
        self.dir = directory
        self.lock = threading.Lock()
        self.handles: Dict[str, 'io.TextIOWrapper'] = {}
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

    def _path(self, name: str) -> str:
        assert self.dir is not None
        return os.path.join(self.dir, f"{name}.jsonl")

    def _ensure_open(self, name: str):
        if self.dir is None:
            return None
        import io
        if name not in self.handles:
            self.handles[name] = open(self._path(name), "a", encoding="utf-8")
        return self.handles[name]

    def write(self, site: str, results: Dict[str, List]):
        if not self.dir:
            return
        with self.lock:
            for coll, items in results.items():
                if coll == "media_extension_counts":
                    h = self._ensure_open("media_extension_counts")
                    if h:
                        line = json.dumps({"site": site, "counts": items}, ensure_ascii=False)
                        h.write(line + "\n")
                        h.flush()
                    continue
                h = self._ensure_open(coll)
                if not h:
                    continue
                for item in items:
                    line = json.dumps({"site": site, "item": item}, ensure_ascii=False)
                    h.write(line + "\n")
                h.flush()

    def close(self):
        with self.lock:
            for h in self.handles.values():
                try:
                    h.close()
                except Exception:
                    pass
            self.handles.clear()


#  Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate WordPress via REST API from URL/TXT/CSV/ODS (concurrent, auto-resume, colored, status bar, split output, media counts)."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("-w", "--website", help="Single website (URL or host).")
    target.add_argument(
        "-i",
        "--input-file",
        help="Path to .txt/.csv/.ods with targets. Uses first column or URL-like column.",
    )
    parser.add_argument("--output", required=True, help="Path to write final combined JSON results.")
    parser.add_argument(
        "--split-output-dir",
        help="Optional directory to also write per-collection JSONL (e.g., media.jsonl, users.jsonl) and media_extension_counts.jsonl.",
    )
    parser.add_argument(
        "--state-file",
        default=".wp_rest_enum.state.json",
        help="Separate state file path (auto-resume). Default: .wp_rest_enum.state.json",
    )
    parser.add_argument("--workers", type=int, default=10, help="Max concurrent workers.")
    parser.add_argument("--log-level", default="INFO", type=str, help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("-m", "--media", action=argparse.BooleanOptionalAction, help="Fetch media.")
    parser.add_argument("-po", "--posts", action=argparse.BooleanOptionalAction, help="Fetch posts.")
    parser.add_argument("-pa", "--pages", action=argparse.BooleanOptionalAction, help="Fetch pages.")
    parser.add_argument("-u", "--users", action=argparse.BooleanOptionalAction, help="Fetch users.")
    parser.add_argument("-c", "--comments", action=argparse.BooleanOptionalAction, help="Fetch comments.")
    parser.add_argument("--pretty", action=argparse.BooleanOptionalAction, help="Pretty-print JSON output.")
    return parser.parse_args()

# Main
def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    targets = read_targets(args.website, args.input_file)
    if not targets:
        logging.error("No valid targets.")
        sys.exit(1)

    # feature flags default to "all" when none provided
    flags_provided = any(
        v is not None for v in [args.posts, args.pages, args.comments, args.media, args.users]
    )
    do_posts = args.posts if flags_provided else True
    do_pages = args.pages if flags_provided else True
    do_comments = args.comments if flags_provided else True
    do_media = args.media if flags_provided else True
    do_users = args.users if flags_provided else True

    # Auto resume via separate state file (results NOT stored here)
    state_path = args.state_file
    completed, _prev_remaining = load_state(state_path)
    completed &= set(targets)  # ignore stale entries
    remaining = [t for t in targets if t not in completed]

    lock = threading.Lock()
    split_writer = SplitOutputWriter(args.split_output_dir)

    if HAS_TQDM and tqdm is not None:
        bar = tqdm(
            total=len(targets),
            initial=len(completed),
            desc="Enumerating",
            unit="site",
            file=sys.stdout,                 # same stream as logging
            dynamic_ncols=True,              # adapt to terminal width
            mininterval=0.2,                 # throttle redraws
            leave=True,                      # keep final single line
            disable=not sys.stdout.isatty(), # avoid multiline when piped
        )
    else:
        bar = None
        logging.info("Progress: %d/%d completed.", len(completed), len(targets))

    # Combined output
    combined_results: Dict[str, Dict[str, List]] = {}

    def handle_done(site: str, site_result: dict) -> None:
        with lock:
            if site_result:
                combined_results[site] = site_result
                split_writer.write(site, site_result)
            completed.add(site)
            rem_list = [t for t in targets if t not in completed]
            save_state(state_path, completed, rem_list)
            if bar:
                bar.update(1)
                bar.set_postfix_str(site[:60])  # keep postfix concise
            else:
                logging.info("Progress: %d/%d completed. Last: %s", len(completed), len(targets), site)

    if remaining:
        # Route logging inside the loop through tqdm 
        with logging_redirect_tqdm():
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                future_map = {
                    pool.submit(
                        enumerate_site,
                        site,
                        do_posts,
                        do_pages,
                        do_comments,
                        do_media,
                        do_users,
                    ): site
                    for site in remaining
                }
                for fut in as_completed(future_map):
                    site = future_map[fut]
                    try:
                        site_result = fut.result()
                    except Exception as e:
                        logging.error("Unhandled error for %s: %s", site, e)
                        site_result = {}
                    handle_done(site, site_result)
    else:
        logging.info("Nothing to do; all %d target(s) already completed.", len(completed))

    if bar:
        bar.close()
    split_writer.close()

    # Final combined write
    atomic_write_json(args.output, combined_results, pretty=bool(args.pretty))

    # Cleanup state after success
    try:
        if os.path.exists(state_path):
            os.remove(state_path)
    except OSError:
        pass

    # Overall media summary
    overall_counts: Dict[str, int] = {}
    if do_media:
        for site_data in combined_results.values():
            for ext, n in site_data.get("media_extension_counts", {}).items():
                overall_counts[ext] = overall_counts.get(ext, 0) + n
        if overall_counts:
            logging.info(
                "Overall media extension counts: %s",
                dict(sorted(overall_counts.items(), key=lambda x: (-x[1], x[0]))),
            )
    logging.info("Wrote combined results to %s (%d site(s)).", args.output, len(combined_results))


if __name__ == "__main__":
    main()
