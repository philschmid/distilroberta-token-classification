# Fine-tune `distilroberta-base` for token-classification

This repository contains the code to train [distilroberta-base](https://huggingface.co/distilroberta-base) for token-classiciaton using either `conl++`, `wikiANN` or both of them for training. Training can be launched either on a local machine or Amazon SageMaker. 


Features:
- [X] Load best model at the end
- [x] Hyperparameter search
- [ ] early stopping

## Training

https://colab.research.google.com/drive/1-gZS6YNLdLPBGXzW8305BNhxE8LgFz7u#scrollTo=dBoBfjJ6vRBp

### Launch training

**SageMaker**

You need to define your Hyperparameters, instance_type, instance count in `sagemaker_launcher.py`

```bash
python sagemaker_launcher.py \
  --model_name_or_path distilroberta-base \
  --dataset conll2003 \
  --learning_rate 3e-05 \
  --num_train_epochs 4 \
  --weight_decay 0.0 \
  --use_auth_token api_Uxxxx 
```

**Local Machine**


```bash
python src/train.py \
  --model_name_or_path distilroberta-base \
  --dataset conll2003 \
  --output_dir /content/model \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-05 \
  --num_train_epochs 4 \
  --use_auth_token api_Uxxxx 
```

### Launch hyperparameter search with optuna

Inspired/Guided from the [Hugging Face Blog Post "Hyperparameter Search with Transformers and Ray Tune"](https://huggingface.co/blog/ray-tune). Here is another example from ray [pbt_transformers_example](https://docs.ray.io/en/master/tune/examples/pbt_transformers.html)

**Local Machine**


```bash
python src/optuna_hyperparameter_search_train.py \
  --model_name_or_path distilroberta-base \
  --dataset conll2003 \
  --output_dir /content/hp \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 16 \
  --n_trials 25 
```

**SageMaker**
```bash
python src/sagemaker_launcher.py \
  --model_name_or_path distilroberta-base \
  --dataset conll2003 \
  --run_hpo \
  --n_trials 25 
```



## Results

```json
{'learning_rate': 4.568387981745284e-05,
 'num_train_epochs': 5, 
 'weight_decay': 0.06449331205217038}. 
 ```
 Best is trial 2 with `eval_f1=0.9541` and `test_f1=0.9108`.

```bash
 BestRun(run_id='16', objective=0.954160391566265, hyperparameters={'learning_rate': 3.97877689893475e-05, 'num_train_epochs': 5, 'weight_decay': 0.05250677574551765})
```

```json
 {'learning_rate': 4.9902376275441704e-05, 'num_train_epochs': 6, 'weight_decay': 0.1270276949568118}. Best is trial 10 with value: 0.9545005411000801.

{'learning_rate': 4.187762151532122e-05, 'num_train_epochs': 4, 'weight_decay': 0.15885868332736558}. Best is trial 22 with value: 0.9557580311854217.
```