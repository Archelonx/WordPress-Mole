# WordPress-Mole
The all-in-one WordPress enumeration script.

[![License](https://img.shields.io/badge/license-AGPL--3.0-brightgreen)](/LICENSE)


**Concurrent WordPress REST API enumerator** with auto-resume, media type counting, colored logging, and per-collection output.  
Designed for reconnaissance, automation, and bug bounty research — **fast, resilient, and non-intrusive.**

---

## ⚙️ Features
- **Concurrent enumeration** with configurable worker threads  
- **Auto-resume** from state file after crash/interruption  
- **Per-collection split output** (`media.jsonl`, `users.jsonl`, etc.)  
- **Automatic URL normalization** (`http`, `https`, etc.)  
- **Colored logging** with progress bar (`tqdm`)  
- **Supports** `.txt`, `.csv`, and `.ods` input (via `pandas`)  
- **Media extension summary** (e.g., `jpg=12`, `pdf=4`, etc.)  
- **Atomic JSON writes** (no corrupt output on interruption)  

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/wordpress-rest-enum.git
cd wordpress-rest-enum
pip install -r requirements.txt
```

## 🚀 Usage
```
python wpmole.py -w example.com --output results.json
```

Or enumerate from a file:

```
python wpmole.py -i targets.txt --output out.json
```

### Supported input formats:
- .txt — line-separated URLs/domains
- .csv — auto-detects URL column or uses first column
- .ods — requires pandas + odfpy

## Example
```
python wpmole.py -i urls.csv --output combined.json --split-output-dir results --workers 20 --log-level INFO
```

## Arguments

| Flag                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `-w`, `--website`    | Single target (URL or host)                                  |
| `-i`, `--input-file` | Path to .txt/.csv/.ods with targets                          |
| `--output`           | Combined JSON output file                                    |
| `--split-output-dir` | Optional directory for per-collection JSONL files            |
| `--state-file`       | Auto-resume state file (default: `.wp_rest_enum.state.json`) |
| `--workers`          | Max concurrent threads (default: 10)                         |
| `--log-level`        | Logging level (DEBUG, INFO, WARNING, ERROR)                  |
| `-m`, `--media`      | Fetch media collection                                       |
| `-po`, `--posts`     | Fetch posts                                                  |
| `-pa`, `--pages`     | Fetch pages                                                  |
| `-u`, `--users`      | Fetch users                                                  |
| `-c`, `--comments`   | Fetch comments                                               |
| `--pretty`           | Pretty-print output JSON                                     |

