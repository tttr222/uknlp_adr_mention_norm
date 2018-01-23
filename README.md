# ADR Mention Normalization

A supervised classification model for concept normalization of adversed drug reactions (ADRs) mentions. Given a concept mention (without contextual information), this model predicts the corresponding MEDDRA Preferred Term (PT) code.

The deep architecture is a hierarchical such that a forward character-level LSTM composes word representations. These word representations are then composed via a bi-directional LSTM to form the phrase (corresponding to the mention). The model is designed such that there are as many softmax output units as there are unique MEDDRA codes appearing in the training set. 

Packages required:
~~~
numpy
sklearn
tensorflow 1.0.0 with tensorflow-fold
~~~

# Data Format

Format of the training data:

`ID <-tab-> Text <-tab-> MEDDRA PT`


Format of the unlabeled testing data:

`ID <-tab-> Text`

# Training and Testing

The model is trained using annotated data located the `data_train` directory. The program will train 10 such models as part of an ensemble and store them in a local `tmp` directory. The program will then annotate unlabeled instances from `data_test` and print corresponding results to a new file named `test_hiercharlstm.txt`.

To execute the file, simply run:

`python main.py`

See the `sample_output.txt` file for the output of this model on this particular dataset.

The output format is:

`ID <-tab-> MEDDRA PT`

For example:

~~~
44675	10041349
40103	10000125
41585	10016322
41834	10019211
46301	10061920
41783	10019133
41184	10016322
45250	10047700
45856	10048010
45048	10044565
40652	10011469
...
~~~

To evaluate the test predictions, you must have the official test labels which are not provided in this repository.

# Author

> Tung Tran  
> tung.tran **[at]** uky.edu  
> <http://tttran.net/>

