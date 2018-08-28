# TopJudge

This is source code of TopJudge which is used to solve the Legal Judgment Prediction(LJP). The details of this model can be found from <a href="http://thunlp.org/~tcc/publications/emnlp2018_dag.pdf" target="_blank">http://thunlp.org/~tcc/publications/emnlp2018_dag.pdf</a>

## Usage

For training, you can run like:

```
python train.py --config CONFIG_FILE --gpu GPU_ID
```

The ``gpu`` config is optional.

For testing, you can run like:

```
python test.py --config CONFIG_FILE --gpu GPU_ID --model MODEL_PATH
```

If you want to use our data setting, you need to promise that there are the frequency count result in the data folder. You can run ``python counter.py`` to generate the count result. This version of ``conuter.py`` has not been written in OOP form yet.

If your data format is different from this project's setting, you need to modify ``net/data_formatter.py``. You need to implement two functions, ``check`` and ``parse``. The function ``check`` has two parameters: ``data`` and ``config``, and it will return whether this data is suitable for the config. The function ``parse`` has three parameters: ``data``, ``config`` and ``transformer``. The ``transformer`` is the Word2Vec object, you can view the function ``net/data_formatter.py/load`` for usage. The ``parse`` function should return three things. The first is the documents embedding with size $sentence\_num\times sentence\_len\times vec\_size$. The second is the length of every sentence with total size $sentence\_num$. The third is the label whose size should be fixed for your tasks.

## Config File

To run our project, a config file is needed. If your config file is ``config``, we will first query the config setting in ``config``, and then if failed, query the config setting in ``default_config``. You can find ``default_config`` in the directory ``config``. If you want to learn the details of the config file, you can read the following list:

* Field ``net``
    * ``name``: The type of model. There are some possible models listing following:
        * ``CNN``: The model using CNN as encoder.
        * ``LSTM``: The model using LSTM as encoder.
        * ``MultiLSTM``: The model using hierarchical LSTM as encoder.
        * ``ArtFact``: The model in ``Luo, Bingfeng, et al. "Learning to Predict Charges for Criminal Cases with Legal Basis." (2017).``
        * ``ArtFactSeq``: Combine the model metioned before with SeqJudge.
        * ``CNNSeq``: Combine CNN encoder with SeqJudge.
        * ``MultiLSTMSeq``: Combine hierarchical LSTM encoder with SeqJudge.
        * ``Article``: Use artilce information as known knowledge to improve the results of SeqJudge.
    * ``max_gram``: The max gram of CNN encoder.
    * ``min_gram``: The min gram of CNN encoder.
    * ``fc1_feature``: The num of features of fc layer.
    * ``filters``: The num of filters of CNN.
    * ``more_fc``: Whether to add a more fc for every tasks or not.
    * ``hidden_size``: The hidden size of LSTM cell.
    * ``attention``: Whether to use attention or not.
    * ``num_layers``: The number of layers in LSTM.
    * ``method``: The method of LSTM to generate embedding. The possible options are:
        * ``MAX``: Using max-pooling on the LSTM output.
        * ``LAST``: Using the last output.
* Field ``data``
    * ``data_path``: The path of data. Under the data path.
    * ``dataset``: The dataset under the path of data. There should be two files named as ``crit.txt`` and ``law.txt``, containing the frequency information.
    * ``train_data``: The list of train data filenames.
    * ``test_data``: The list of test data filenames.
    * ``type_of_label``: The list of tasks.
    * ``type_of_loss``: The list of loss of every task. The posssble loss function are:
        * ``single_classification``: Single class classifcation loss, with cross entropy.
        * ``multi_classification``: Multi classes classifcation loss, with cross entropy.
        * ``log_regression``: $\log^2\left(\left|y_{label}-y_{predict}\right|+1\right)$, unimplemented yet.
    * ``graph``: The DAG structure of SeqJudge.
    * ``batch_size``: The batch size.
    * ``shuffle``: Whether to shuffle the data or not.
    * ``vec_size``: The vector size of every word embedding.
    * ``sentence_num``: The maximum number of sentences in documents.
    * ``sentence_len``: The maximum length of a sentence.
    * ``min_frequency``: The min frequency of a label in the data.
    * ``word2vec`` : The word2vec path.
    * ``top_k``: The top-k relevant articles, see the paper we mentioned.
* Field ``train``
    * ``epoch``: The maximum training epoches.
    * ``learning_rate``: The learning rate.
    * ``momentum``: The momentum of sgd.
    * ``weight_decay``: The weight decay.
    * ``optimizer``: The optimizer, now you can use ``adam`` or ``sgd``.
    * ``dropout``: The dropout ratio.
    * ``train_num_process``: The num of processes to read training data.
    * ``test_num_process``: The num of processes to read testing data.
* Field ``debug``
    * ``output_time``: The num of batches to output debug information.
    * ``model_path``: The basic path to save models.
    * ``test_path``: The basic path to save test results,
    * ``model_name``: The name of this config model. We will store all the things in the directory with name basic path add model name.
    * ``test_time``: The num of epoches to run a test.

## Reference

BibTex:

```
@InProceedings{zhong2018topjudge,
  Title                    = {Legal Judgment Prediction via Topological Learning},
  Author                   = {Zhong, Haoxi and Guo, Zhipeng and Tu, Cunchao and Xiao, Chaojun and Liu, Zhiyuan and Sun, Maosong},
  Booktitle                = {Proceedings of EMNLP},
  Year                     = {2018}
}
```



