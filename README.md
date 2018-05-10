# Law_pre

Law_pre is the shortcut for law prediction. This is a project uses the method named as SeqJudge to solve the Legal Judgment Prediction(LJP).

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
If your data format is different from this project's setting, you need to modify ``net/data_formatter.py``. You need to implement two functions, ``check`` and ``parse``. The function ``check`` has two parameters: ``data`` and ``config``, and it will return whether this data is suitable for the config. The function ``parse`` has three parameters: ``data``, ``config`` and ``transformer``. The ``transformer`` is the Word2Vec object, you can view the function ``net/data_formatter.py/load`` for usage. The ``parse`` function should return three things. The first is the documents embedding with size $sentence\_num\times sentence\_len\times vec\_size$. The second is the length of every sentence with total size $sentence\_num$. The third is the label whose size should be fixed for your tasks.

## Config File

To run our project, a config file is needed. If your config file is ``config``, we will first query the config setting in ``config``, and then if failed, query the config setting in ``default_config``. You can find ``default_config`` in the directory ``config``. If you want to learn the details of the config file, you can read the following list:

* Field ``net``
    * ``name``:
    * 
* Field ``data``
* Field ``train``
* Field ``debug``

Under construction.

## Reference

Under construction.

