# ATOM
ATOM: Commit Message Generation Based onAbstract Syntax Tree and Hybrid Ranking

A repository with the code for [the paper](https://arxiv.org/pdf/1912.02972.pdf). It consists the code for preprocessing and training the commit message generation model. The data is also available at [dataset](https://drive.google.com/drive/folders/1Qw06U462e1nFUXEzXsclziMBpFrulykM).

Requirements:
* Python 3.6
* Tensorflow 1.14.0
* GitPython 2.1.11
* Pandas 0.23.4
* Pygment 2.3.1
* NlTK 3.4.5
* multiprocess 0.70.7
* Ctags 5.8
* scikit-learn 0.20.2
* Numpy 1.14.5
* Pickle 0.7.5

## Pre-Processing Stage

The preprocessing contains three steps, gitproject.py function_extractor.py and ast_extractor_java.py respectively.

Firstly, you need to download the needed repos locally and run gitproject.py to save the raw data into csv files. The output will be saved in 'root_project/ast_process/data/java_relevant_data/raw_commits_from_github/'

```python gitproject.py --project_dir <Repo Dir```

Based on the raw csv files, you need to extract the positive and negative functions. So function_extractor.py serves as this.

```python function_extractor.py --repo_dir <repo dir> --multi ```

repo dir is the repo directory you have download and multi is ture or false to multiprocess the data and default is true.

Finally, we need to extract related asts based on these functions. Run 
```python ast_extractor_java.py -l <max_path_length> -w <max_path_width>```
where max_path_length is the clipped length of AST extracted and default is 8.
The max_path_width is the maximun width of ast path, default is 2.

After the preprocessing, we can get the preprocessed AST data save in the "project_root/ast_process/data/java_relevant_data/ast_commits"

## Train Stage
The training stage contains two steps, first, we need to train AST2seq model. Once AST2seq model is done, we can use it for hybrid ranking model to prioritize the best commit messages.

### AST2seq
Based on the processed AST data, we need to build the dataset for train, and test. 
Run the script in commit2seq "python preporoces.py --dataset_name <dataset name> --max_contexts <max path number> --message_lengths <max message length>" where dataset is the name of your dataset constructed, max_context is the max ast paths and message path will filter
too long sequences. The constructed dataset will be saved in "project_root/commit2seq/training_data/translation".

Based on the dataset, we can train our AST2seq model by running the shell train.sh. In this shell, you need to specify the dataset name, data_dir. 

### Hybrid Ranking
We provides different variants of hybird ranking models e.g., CNN, LSTM, GRU or attention. To train the hybird ranking model, 
you need to run trainer.py file in the dir commit2seq/ranking/trainer.py and save the model to ranking_models directory.

We also provide the implementation of [Commit2Vec: Learning Distributed Representations of Code Changes](https://arxiv.org/abs/1911.07605) in the commt2vec directionary. The difference between commit2vec and ast2seq is the part of encoding part. In ast2seq, we use Bi-lSTM to encode ast paths, however, commit2seq chose to use fully-collected layer for path representation.


## Citation
If you find this code or our paper relevant to your work, please cite our arXiv paper:

```
@article{liu2019atom,
  title={ATOM: Commit Message Generation Based on Abstract Syntax Tree and Hybrid Ranking},
  author={Liu, Shangqing and Gao, Cuiyun and Chen, Sen and Nie, Lun Yiu and Liu, Yang},
  journal={arXiv preprint arXiv:1912.02972},
  year={2019}
}
```

