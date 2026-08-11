import argparse
import yaml

from data_division import save_splits



def main(args) -> None:
    """Fine-tune the model. use mlflow to log the training process and save the model."""
    # read training config yaml
    with open("src/training_config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    if args.data_dir is not None:
        config_dict["data_dir"] = args.data_dir
    
    if args.include_data_download:
        # run data preparation script to download and prepare the dataset
        import src.data_collection as dc
        caltech_root = dc.download_data(config_dict["data_dir"])
        train_splits, val_split = dc.divide_dataset(caltech_root)
        dc.save_splits(train_splits, val_split,dc.get_label_map(caltech_root))

    if args.exclude_train:
        print("Data preparation completed. Exiting as --exclude_train flag is set.")
        return
    
    else:
        from src.train import train
        train(config_dict["data_dir"])



if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="Train a MobileNetV2 model on the Caltech101 dataset."
    )
    argument_parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the training data directory.",
    )
    argument_parser.add_argument(
        "--include_data_download",
        action="store_true",
        help="Include data download step.",
    )
    print("Starting training")
    args = argument_parser.parse_args()
    main(args)
