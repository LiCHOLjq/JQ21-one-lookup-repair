"""Recover frozen title/abstract strings by checking the appended metadata verbatim."""
import argparse
import csv
import gzip
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def api(endpoint, params):
    url = 'https://api.openalex.org/' + endpoint + '?' + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)['results']
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def concepts(ids):
    return api('concepts', {'filter': 'ids.openalex:' + '|'.join(ids),
                            'per_page': len(ids), 'select': 'id,display_name'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('id_map', type=Path)
    parser.add_argument('works', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('names', type=Path, help='JSON field/concept names, downloaded if absent')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.id_map.open(encoding='utf8') as f:
        id_map = {int(row['id']): row['paper_id'] for row in csv.DictReader(f)}
    selected = [id_map[int(i)] for i in np.load(args.cache)['ids']]
    keep = set(selected)
    rows = {}
    with args.works.open(encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            if row['paper_id'] in keep:
                rows[row['paper_id']] = row
    assert len(selected) == len(rows) == 20000
    if not args.names.exists():
        terms = sorted({key for row in rows.values() for key in row['field_ids']
                        if key.startswith('C')})
        records = api('fields', {'per_page':100,'select':'id,display_name'})
        with ThreadPoolExecutor(4) as pool:
            for group in pool.map(concepts, [terms[i:i+90] for i in range(0,len(terms),90)]):
                records.extend(group)
        args.names.write_text(json.dumps({r['id'].rsplit('/',1)[-1]:r['display_name']
                                          for r in records},ensure_ascii=False),encoding='utf8')
    names = json.loads(args.names.read_text(encoding='utf8'))
    unmatched = []
    prepared = {}
    for ident in selected:
        row = rows[ident]
        suffix = ' ' + ' '.join(names.get(key, key) for key in row['field_ids']) + ' ' + row['venue']
        if row['text'].endswith(suffix):
            prepared[ident] = row['text'][:-len(suffix)]
        else:
            unmatched.append(ident)
    # When names have changed, accept a current abstract only if it matches the
    # beginning of the frozen string exactly; this never guesses a word boundary.
    for i in range(0,len(unmatched),75):
        group = unmatched[i:i+75]
        current = api('works', {'filter':'openalex:'+'|'.join(group),'per_page':len(group),
                                'select':'id,abstract_inverted_index'})
        for r in current:
            ident = r['id'].rsplit('/',1)[-1]
            words = sorted((pos,word) for word,positions in (r['abstract_inverted_index'] or {}).items()
                           for pos in positions)
            text = rows[ident]['title'] + ' ' + ' '.join(word for _,word in words)
            if words and rows[ident]['text'].startswith(text+' '):
                prepared[ident] = text
    remaining = sorted(keep-set(prepared))
    assert not remaining, f'Could not verify {len(remaining)} text prefixes: {remaining[:6]}'
    with gzip.open(args.output,'wt',encoding='utf8') as f:
        for index, ident in enumerate(selected):
            assert prepared[ident].startswith(rows[ident]['title']+' ')
            f.write(json.dumps([index,prepared[ident],rows[ident]['text']],ensure_ascii=False)+'\n')
    print(f'{len(selected)} frozen texts mapped; {len(unmatched)} used abstract-prefix verification')


if __name__ == '__main__':
    main()
