# Fine-tune `distilroberta-base` for token-classification

This repository contains the code to train [distilroberta-base](https://huggingface.co/distilroberta-base) for token-classiciaton using either `conl++`, `wikiANN` or both of them for training. Training can be launched either on a local machine or Amazon SageMaker. 


Features:
- [X] Load best model at the end
- [x] Hyperparameter search
- [ ] early stopping

```json
{'learning_rate': 4.568387981745284e-05,
 'num_train_epochs': 5, 
 'weight_decay': 0.06449331205217038}. 
 ```
 Best is trial 2 with `eval_f1=0.9541` and `test_f1=0.9108`.

## Training

https://colab.research.google.com/drive/1-gZS6YNLdLPBGXzW8305BNhxE8LgFz7u#scrollTo=dBoBfjJ6vRBp

### Launch training

**SageMaker**

You need to define your Hyperparameters, instance_type, instance count in `sagemaker_launcher.py`

```bash
python sagemaker_launcher.py
```

**Local Machine**


```bash
python src/train.py --foo bar
```

### Launch hyperparameter search with ray tune

Inspired/Guided from the [Hugging Face Blog Post "Hyperparameter Search with Transformers and Ray Tune"](https://huggingface.co/blog/ray-tune). Here is another example from ray [pbt_transformers_example](https://docs.ray.io/en/master/tune/examples/pbt_transformers.html)

**Local Machine**


```bash
python src/hyperparameter_search_train.py
```

## Results