import argparse
import os
import boto3

from sagemaker import Session
from sagemaker.huggingface import HuggingFace


ROLE_NAME = "sagemaker_execution_role"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.p3.xlarge"
PROFILE = "hf-sm"
REGION = "us-east-1"


def launch(args):

    os.environ["AWS_PROFILE"] = args.profile  # setting aws profile for our boto3 client
    os.environ["AWS_DEFAULT_REGION"] = args.region  # current DLCs are only in us-east-1 available
    iam_client = boto3.client("iam")
    role = iam_client.get_role(RoleName=args.role_name)["Role"]["Arn"]

    entry_point = "train.py"

    # hyperparameters, which are passed into the training job
    hyperparameters = {
        "model_name_or_path": args.model_name_or_path,
        "task": args.task,
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
    }

    if args.run_hpo:
        entry_point = "optuna_hyperparameter_search_train.py"
        hyperparameters["n_trials"] = args.n_trials
    else:
        entry_point = "train.py"
        hyperparameters["use_auth_token"] = args.use_auth_token
        hyperparameters["weight_decay"] = args.weight_decay
        hyperparameters["learning_rate"] = args.learning_rate
        hyperparameters["num_train_epochs"] = args.num_train_epochs
        hyperparameters["use_auth_token"] = args.use_auth_token

    huggingface_estimator = HuggingFace(
        entry_point=entry_point,
        source_dir="./scripts",
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        role=role,
        transformers_version="4.4",
        pytorch_version="1.6",
        py_version="py36",
        hyperparameters=hyperparameters,
    )

    # starting the train job with our uploaded datasets as input
    print(huggingface_estimator.hyperparameters())
    # huggingface_estimator.fit()


def parse_args():
    parser = argparse.ArgumentParser()

    # transformers base config
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--task", type=str, default="ner")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output_dir", type=str, default="opt/ml/model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    # aws config
    parser.add_argument("--instance_type", type=str, default=INSTANCE_TYPE)
    parser.add_argument("--instance_count", type=int, default=INSTANCE_COUNT)
    parser.add_argument("--profile", type=str, default=PROFILE)
    parser.add_argument("--region", type=str, default=REGION)
    parser.add_argument("--role_name", type=str, default=ROLE_NAME)

    # hyperparameter config
    parser.add_argument("--run_hpo", action="store_true")
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_auth_token", type=str, default="")

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    launch(args)
