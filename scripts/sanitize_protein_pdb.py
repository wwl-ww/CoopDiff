#!/usr/bin/env python
"""Copy a protein PDB while excluding non-amino-acid ATOM records."""

import argparse
from pathlib import Path


AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE',
}


def sanitize(source):
    """Return PDB text and the excluded residue identifiers."""
    kept, excluded = [], set()
    for line in Path(source).read_text(encoding='utf-8').splitlines():
        if line.startswith(('ATOM  ', 'ANISOU')) and line[17:20].strip() not in AMINO_ACIDS:
            excluded.add((line[17:20].strip(), line[21:27].strip()))
            continue
        kept.append(line)
    return '\n'.join(kept) + '\n', sorted(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error('destination already exists')
    text, excluded = sanitize(args.source)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(text, encoding='utf-8')
    print({'destination': str(args.destination), 'excluded': excluded})


if __name__ == '__main__':
    main()
