# -*- coding: utf-8 -*-

import click
import freebase
import gzip
import os


@click.group()
def cli():
    pass


@cli.command()
@click.argument('freebase_dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File(mode='w'), default='freebase2wikipedia.tsv')
def create_mapping(freebase_dump_file, out_file):
    mapping = {}
    with gzip.open(freebase_dump_file) as f:
        target_pred = '<http://rdf.freebase.com/key/wikipedia.en_title>'

        for (n, line) in enumerate(f, 1):
            (s, p, o, _) = line.rstrip().split('\t')

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
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path(exists=True, file_okay=False), default='.')
@click.argument('description_db_file', type=click.Path(exists=True))
@click.argument('mapping_file', type=click.File(), default='freebase2wikipedia.tsv')
def create_dataset(input_dir, output_dir, description_db_file, mapping_file):
    from utils.description_db import DescriptionDB

    description_db = DescriptionDB(description_db_file)
    valid_titles = frozenset(description_db.keys())

    dataset_files = ['Etrain', 'Edev', 'Etest']

    mapping = {}
    for line in mapping_file:
        (mid, title) = line.rstrip().decode('utf-8').split('\t')
        if title in valid_titles:
            mapping[mid] = title

    label_set = set()
    for name in dataset_files:
        with open(os.path.join(output_dir, name[1:] + '.tsv'), mode='w') as f:
            with open(os.path.join(input_dir, name)) as in_file:
                f.write('freebase_id\twikipedia_title\ttypes\n')

                for line in in_file:
                    parsed_line = [s.strip() for s in line.rstrip().decode('utf-8').split()]
                    mid = parsed_line[0]
                    if mid in mapping:
                        title = mapping[mid]
                        labels = []
                        for s in parsed_line[1:]:
                            if s.startswith('-'):
                                label = s.rstrip()[1:]
                                labels.append(label)
                                label_set.add(label)

                        f.write('%s\t%s\t%s\n' % (mid.encode('utf-8'), title.encode('utf-8'),
                                                  ','.join(labels).encode('utf-8')))

    with open(os.path.join(output_dir, 'labels.txt'), mode='w') as f:
        for label in label_set:
            f.write('%s\n' % label.encode('utf-8'))


@cli.command()
@click.argument('dataset_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('out_file', type=click.File(mode='w'), default='titles.txt')
@click.argument('mapping_file', type=click.File(), default='freebase2wikipedia.tsv')
def create_title_list(dataset_dir, out_file, mapping_file):
    dataset_files = ['Etrain', 'Edev', 'Etest']

    mapping = {}
    for line in mapping_file:
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


def _extract_freebase_id(s):
    if not s.startswith('<http://rdf.freebase.com/ns/m'):
        return None

    return u'/' + s[28:-1].replace(u'.', u'/')


if __name__ == '__main__':
    cli()
