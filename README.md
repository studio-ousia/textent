TextEnt
=======

TextEnt is a software to obtain vector representations of entities and documents from Wikipedia.

## Installation

You can install this software by:

```
% git clone https://github.com/studio-ousia/textent.git
% cd textent
% pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
% pip install -r requirements.txt
```

Currently, the code is based on Python 2.

## Reproducing Experiments

The experimental results presented in the paper can be reproduced by running the commands described below.
We conducted experiments using Python 2.7, Ubuntu Linux 16.04, and NVIDIA GTX GPUs.

### Resources

- [TextEnt Embeddings](http://textent.s3.amazonaws.com/pub/coling2018/textent_300d.tar.bz2)
- [Entity DB](http://textent.s3.amazonaws.com/pub/coling2018/entity_db_20160601.pkl.bz2): A database that contains information of Wikipedia entities
- [TAGME Cache Data](http://textent.s3.amazonaws.com/pub/coling2018/tagme_cache.pkl.bz2): A serialized dict file that stores the results returned by the [TAGME entity linker](https://tagme.d4science.org/tagme/)

Please unarchive these files before proceeding to the next step.

**Entity Typing experiment**:

```
% python cli.py evaluate_entity_typing textent_300d entity_db_20160601.pkl
```
The dataset used in this experiment is contained in the [dataset/entity_typing](dataset/entity_typing) directory.

**Text classification expeiment on the 20 newsgroups dataset**:

```
% python cli.py evaluate_text_classification textent_300d entity_db_20160601.pkl --tagme-cache=tagme_cache.pkl -t 20ng
```

**Text classification experiment on the R8 dataset**:

```
% python cli.py evaluate_text_classification textent_300d entity_db_20160601.pkl --tagme-cache=tagme_cache.pkl -t r8
```
