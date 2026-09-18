"""
Module 1 — replay_simulator.py
Streams replay_stream.csv (produced by module1_preprocessing.py) row by row,
in chronological-proxy order, to Module 4's /ingest endpoint on a timer —
this is what makes the live dashboard demo look real.

Usage:
    python replay_simulator.py --file ./output/replay_stream.csv \
        --url http://localhost:8000/ingest --min-delay 0.5 --max-delay 2.0

    # Loop forever once the file is exhausted (handy for a live demo booth):
    python replay_simulator.py --file ./output/replay_stream.csv --loop
"""

import argparse
import json
import random
import time

import pandas as pd
import requests


def stream(file_path: str, url: str, min_delay: float, max_delay: float, loop: bool):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} rows from {file_path}")
    print(f"Streaming to {url} (delay {min_delay}-{max_delay}s per row, loop={loop})")

    sent, failed = 0, 0
    while True:
        for _, row in df.iterrows():
            payload = row.where(pd.notnull(row), None).to_dict()
            try:
                resp = requests.post(url, json=payload, timeout=2)
                if resp.status_code >= 400:
                    failed += 1
                    print(f"  [{resp.status_code}] ingest rejected row flow_seq={payload.get('flow_seq')}")
                else:
                    sent += 1
            except requests.exceptions.RequestException as e:
                failed += 1
                print(f"  connection error on flow_seq={payload.get('flow_seq')}: {e}")

            if (sent + failed) % 100 == 0:
                print(f"  progress: {sent} sent, {failed} failed")

            time.sleep(random.uniform(min_delay, max_delay))

        print(f"Reached end of file. Total sent={sent}, failed={failed}")
        if not loop:
            break
        print("Looping back to the start of the stream...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="./output/replay_stream.csv")
    parser.add_argument("--url", default="http://localhost:8000/ingest",
                         help="Module 4's ingest endpoint")
    parser.add_argument("--min-delay", type=float, default=0.5)
    parser.add_argument("--max-delay", type=float, default=2.0)
    parser.add_argument("--loop", action="store_true",
                         help="Restart from the beginning once the file is exhausted")
    args = parser.parse_args()

    stream(args.file, args.url, args.min_delay, args.max_delay, args.loop)


if __name__ == "__main__":
    main()
