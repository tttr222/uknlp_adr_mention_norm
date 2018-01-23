# ADR Mention Normalization

A supervised classification model for concept normalization of adversed drug reactions (ADRs) mentions. Given a concept mention (without context), this model returns a MEDDRA Preferred Term (PT) code.

Packages required:
~~~
numpy
sklearn
tensorflow 1.0.0 with tensorflow-fold
~~~

# Training and Testing

This model is trained on examples from files in the `data_train` folder and labels unlabeled instances from `data_test/task_3_test1_amiaformat_to_release.txt`. The output is printed to `test_hiercharlstm.txt`.

`python run_charlstm_test.py`

# Author

> Tung Tran  
> tung.tran **[at]** uky.edu  
> <http://tttran.net/>

