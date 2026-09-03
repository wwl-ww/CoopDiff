#!/usr/bin/env python
"""Write a deterministic prefix subset of a ternary JSON index."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', required=True, type=int)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit('--limit must be positive')

    source = json.loads(Path(args.index).read_text(encoding='utf-8'))
    entries = source.get('entries', source)
    if len(entries) < args.limit:
        raise SystemExit(f'index contains only {len(entries)} entries')
    payload = {
        'entries': entries[:args.limit],
        'metadata': {
            'source_index': args.index,
            'selection': 'prefix',
            'requested': args.limit,
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Wrote {args.limit} entries to {destination}')


if __name__ == '__main__':
    main()
