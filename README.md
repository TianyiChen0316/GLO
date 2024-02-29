# GLO: Towards Generalized Learned Query Optimization

## Requirements

Before running GLO, please install [PyTorch](https://pytorch.org/get-started/locally/) and [Deep Graph Library (DGL)](https://www.dgl.ai/pages/start.html) following the instructions on the pages, and run the command `pip install -r requirements.txt` to install the required packages. (We use `torch==2.0.1` and `dgl==1.1.1+cu118` in our experiments.) In addition, we use [psqlparse](https://github.com/alculquicondor/psqlparse) to parse queries. Since we found that installing psqlparse with `pip` may result in errors, we recommend building it directly with setup.py.

## Running

Please use the following command to train GLO.

```shell
python glo.py
```

GLO provides a list of settings as follows.

- `-F FILE_ID`: To change the output file ID of experiment results and checkpoints. The default value is `(default)`.
- `-d TRAIN TEST`: To train with the specified training and testing workloads. `TRAIN` is the folder of training workload, and `TEST` is for testing workload.
  - The specified path is relative to `./GLO/`. For example, when the training workload path is `./GLO/dataset/train_tpcds`, please use `dataset/train_tpcds` instead.
- `-e EPOCHS`: To train with the specified number of epochs. The default value is `200`.
- `--seed VALUE`: To set the random seed of training.
- `-D DATABASE`: To specify the PostgreSQL database name. By default, we set the name to `database`.
- `-U USER`: To specify the PostgreSQL user name. The default value is `postgres`.
- `-P PASSWORD`: To specify the password of the PostgreSQL user. By default, the password is not used.
- `--port PORT`: To specify the port of PostgreSQL.
- `--host HOST`: To specify the server host name of PostgreSQL.
- `--reset`: To ignore the previous checkpoint and retrain the model.
- `--warm-up ITER`: To execute the train and test workload queries for `ITER` iterations before training. This argument can be used to warm up the shared buffers.
