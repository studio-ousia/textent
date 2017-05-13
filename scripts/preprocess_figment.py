# -*- coding: utf-8 -*-

import click
import freebase
import gzip
import os
from utils.description_db import DescriptionDB
from utils.embedding_reader import EmbeddingReader


@click.group()
def cli():
    pass


@cli.command()
@click.argument('fb_dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File(mode='w'), default='fb_wiki_mapping.tsv')
def create_fb_wiki_mapping(fb_dump_file, out_file):
    mapping = {}
    with gzip.open(fb_dump_file) as f:
        target_pred = u'<http://rdf.freebase.com/key/wikipedia.en_title>'

        for (n, line) in enumerate(f, 1):
            line = line.rstrip().decode('utf-8')
            (s, p, o, _) = line.split('\t')

            if p == target_pred:
                mid = _extract_freebase_id(s)
                if mid is None:
                    continue

                title = freebase.api.mqlkey.unquotekey(o[1:-1])
                title = title.replace('_', ' ')
                mapping[mid] = title

    for (mid, title) in mapping.iteritems():
        out_file.write('%s\t%s\n' % (mid.encode('utf-8'), title.encode('utf-8')))


@cli.command()
@click.argument('dataset_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('out_file', type=click.File(mode='w'), default='figment_title_list.txt')
@click.argument('fb_wiki_mapping_file', type=click.File(), default='fb_wiki_mapping.tsv')
def create_wiki_title_list(dataset_dir, out_file, fb_wiki_mapping_file):
    dataset_files = ['Etrain', 'Edev', 'Etest']

    mapping = {}
    for line in fb_wiki_mapping_file:
        (mid, title) = line.rstrip().decode('utf-8').split('\t')
        mapping[mid] = title

    titles = set()
    for name in dataset_files:
        with open(os.path.join(dataset_dir, name)) as f:
            for line in f:
                mid = line.rstrip().decode('utf-8').split('\t')[0]
                if mid in mapping:
                    titles.add(mapping[mid])

    for title in titles:
        out_file.write('%s\n' % title.encode('utf-8'))


@cli.command()
@click.argument('dataset_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('description_db_file', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path(exists=True, file_okay=False), default='.')
@click.argument('fb_wiki_mapping_file', type=click.File(), default='fb_wiki_mapping.tsv')
@click.option('--suffix', default='')
def create_filtered_dataset(dataset_dir, description_db_file, out_dir, fb_wiki_mapping_file, suffix):
    description_db = DescriptionDB(description_db_file)
    valid_titles = frozenset(description_db.keys())

    dataset_files = ['Etrain', 'Edev', 'Etest']

    valid_mid_set = set()
    for line in fb_wiki_mapping_file:
        (mid, title) = line.rstrip().decode('utf-8').split('\t')
        if title in valid_titles:
            valid_mid_set.add(mid)

    for name in dataset_files:
        total = 0
        found = 0
        with open(os.path.join(out_dir, name + suffix), mode='w') as f:
            with open(os.path.join(dataset_dir, name)) as in_file:
                for line in in_file:
                    total += 1
                    if line.split('\t')[0] in valid_mid_set:
                        f.write(line)
                        found += 1
        print '%s: found %d of %d instances (%.3f%%)' % (name, found, total, float(found) / total)


@cli.command()
@click.argument('dataset_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('embedding_file', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('fb_wiki_mapping_file', type=click.File(), default='fb_wiki_mapping.tsv')
@click.option('--suffix', default='')
def create_filtered_dataset_by_embedding(dataset_dir, embedding_file, out_dir, fb_wiki_mapping_file, suffix):
    embedding = EmbeddingReader.load(embedding_file)
    valid_titles = frozenset(embedding.entity_vocab)

    dataset_files = ['Etrain', 'Edev', 'Etest']

    valid_mid_set = set()
    for line in fb_wiki_mapping_file:
        (mid, title) = line.rstrip().decode('utf-8').split('\t')
        if title in valid_titles:
            valid_mid_set.add(mid)

    for name in dataset_files:
        total = 0
        found = 0
        with open(os.path.join(out_dir, name + suffix), mode='w') as f:
            with open(os.path.join(dataset_dir, name)) as in_file:
                for line in in_file:
                    total += 1
                    if line.split('\t')[0] in valid_mid_set:
                        f.write(line)
                        found += 1
        print '%s: found %d of %d instances (%.3f%%)' % (name, found, total, float(found) / total)


def _extract_freebase_id(s):
    if not s.startswith('<http://rdf.freebase.com/ns/m'):
        return None

    return u'/' + s[28:-1].replace(u'.', u'/')


if __name__ == '__main__':
    cli()
