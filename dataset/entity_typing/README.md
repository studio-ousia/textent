# Fine-grained Entity Typing Dataset for Wikipedia Entities

This directory contains a dataset for fine-grained entity typing task of Wikipedia entities.
The aim of this task is to assign each entity with one or more entity types such as *musician* and *film*.

This dataset is based on the dataset proposed in [Yaghoobzadeh and Schütze, 2015](https://arxiv.org/abs/1606.07901).
The original dataset contains 201,933 Freebase entities mapped to 102 fine-grained entity types that were proposed in [Ling and Weld, 2012](https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewPaper/5152).
Because the dataset contains entity type annotations of Freebase entities and we wanted to conduct an experiment using Wikipedia entities, we modified the original dataset by mapping each entity to the corresponding entry in Wikipedia and excluded those entities that did not exist in Wikipedia.
We mapped approximately 92% of the entities in the original dataset to Wikipedia, and obtained training set (train.tsv), development set (dev.tsv), and test set (test.tsv) containing 93,350, 37,036, and 55,715 entities, respectively.
The dataset was created using [process.py](process.py).
