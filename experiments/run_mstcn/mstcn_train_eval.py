import csv
import sys
import os
import numpy as np
import torch
import random
import json
from src.jigsaws_metrics import get_jg_metrics, get_jg_metrics2

sys.path.append('./backbones/ms-tcn')

from batch_gen import BatchGenerator, read_dataset_config
from model import Trainer

from src.utils import load_meta, eval_txts
from src.predict import predict_backbone
import configs.mstcn_config as cfg

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
model_name = "mstcn"
data_name = sys.argv[1]
cuda = sys.argv[2]
device = f"cuda:{int(cuda)}"


def train_eval_model(dataset, split, dataset_config=None):
    (
        actions_dict,
        num_actions,
        gt_path,
        features_path,
        vid_list_file,
        vid_list_file_val,
        vid_list_file_tst,
        sample_rate,
        model_dir,
        result_dir,
        record_dir,
    ) = load_meta(
        cfg.dataset_root,
        cfg.model_root,
        cfg.result_root,
        cfg.record_root,
        dataset,
        split,
        model_name,
        dataset_config["epoch_level_augmentation"],
    )
    if dataset_config["epoch_level_augmentation"]:
        features_path, epoch_path = features_path
        gt_path, prob_path = gt_path
    cfg.features_dim = dataset_config["input_dim"]
    in_dim = dataset_config["input_dim"]
    
    if dataset_config["epoch_level_augmentation"]:
        batch_gen = BatchGenerator(num_actions, actions_dict, prob_path, epoch_path, sample_rate, epoch_level_augmentation=dataset_config["epoch_level_augmentation"], epoch_variational_gt=dataset_config["epoch_variational_gt"])
    else:
        batch_gen = BatchGenerator(num_actions, actions_dict, gt_path, features_path, sample_rate, epoch_level_augmentation=dataset_config["epoch_level_augmentation"], epoch_variational_gt=dataset_config["epoch_variational_gt"])    
    batch_gen.read_data(vid_list_file)
    trainer = Trainer(cfg.num_stages, cfg.num_layers, cfg.num_f_maps, cfg.features_dim, num_actions)
    epochs = dataset_config["num_epochs"]
    print("Using config: ", dataset_config)
    if "restore_weights" in dataset_config.keys() and dataset_config["restore_weights"] == True:
        print("Loading model from checkpoint: MS-TCN")
        trainer.model.load_state_dict(torch.load(dataset_config['restore_path'][model_name][split], map_location=device))
        print("Loaded model from checkpoint: MS-TCN")
    trainer.train(save_dir=model_dir, 
                batch_gen=batch_gen, 
                num_epochs=epochs, 
                batch_size=cfg.batch_size, 
                learning_rate=cfg.lr, 
                device=device,
                dataset_config=dataset_config)

    #### Training Loop ####
    f = open(os.path.join(record_dir, "split_{}_training.csv".format(split)), "w")

    writer = csv.writer(f, delimiter="\t")
    writer.writerow(
        [
            "epoch",
            "accu",
            "edit",
            "F1@{}".format(cfg.iou_thresholds[0]),
            "F1@{}".format(cfg.iou_thresholds[1]),
            "F1@{}".format(cfg.iou_thresholds[2]),
        ]
    )
    for epoch in range(1, epochs + 1):
        print("======================EPOCH {}=====================".format(epoch))
        predict_backbone(
            model_name,
            trainer.model,
            model_dir,
            result_dir,
            epoch_path if (dataset_config["epoch_level_augmentation"] or dataset_config["epoch_variational_gt"]) else features_path,
            vid_list_file,
            epoch,
            actions_dict,
            device,
            sample_rate,
            mode="training",
            epoch_level_augmentation=dataset_config["epoch_level_augmentation"]
        )
        results = eval_txts(
            cfg.dataset_root, result_dir, dataset, split, model_name, "training", epoch
        )
        writer.writerow(
            [
                epoch,
                "%.4f" % (results["accu"]),
                "%.4f" % (results["edit"]),
                "%.4f" % (results["F1@%0.2f" % (cfg.iou_thresholds[0])]),
                "%.4f" % (results["F1@%0.2f" % (cfg.iou_thresholds[1])]),
                "%.4f" % (results["F1@%0.2f" % (cfg.iou_thresholds[2])]),
            ]
        )
    f.close()

    #### Validation Loop ####
    max_epoch = -1
    max_val = 0.0
    max_results = dict()
    total_patience=dataset_config["total_patience"]

    f = open(os.path.join(record_dir, "split_{}_validation.csv".format(split)), "w")
    f2 = open(os.path.join(record_dir, "split_{}_colin_validation.csv".format(split)), "w")

    writer = csv.writer(f, delimiter="\t")
    writer.writerow(
        [
            "epoch",
            "accu",
            "edit",
            "F1@{}".format(cfg.iou_thresholds[0]),
            "F1@{}".format(cfg.iou_thresholds[1]),
            "F1@{}".format(cfg.iou_thresholds[2]),
        ]
    )
    writer2 = csv.writer(f2, delimiter="\t")
    writer2.writerow(
        [
            "epoch",
            "accu",
            "edit",
            "F1@{}".format(cfg.iou_thresholds[0]),
            "F1@{}".format(cfg.iou_thresholds[1]),
            "F1@{}".format(cfg.iou_thresholds[2]),
        ]
    )

    for epoch in range(1, epochs + 1):
        print("======================EPOCH {}=====================".format(epoch))
        predictions = predict_backbone(
            model_name,
            trainer.model,
            model_dir,
            result_dir,
            features_path,
            vid_list_file_val,
            epoch,
            actions_dict,
            device,
            sample_rate,
            mode="validation",
            epoch_level_augmentation=False
        )
        results = eval_txts(
            cfg.dataset_root, result_dir, dataset, split, model_name, "validation", epoch
        )
        writer.writerow(
            [
                epoch,
                "%.4f" % (results["accu"]),
                "%.4f" % (results["edit"]),
                "%.4f" % (results["F1@%0.2f" % (cfg.iou_thresholds[0])]),
                "%.4f" % (results["F1@%0.2f" % (cfg.iou_thresholds[1])]),
                "%.4f" % (results["F1@%0.2f" % (cfg.iou_thresholds[2])]),
            ]
        )

        with open(vid_list_file_val, "r") as f:
            file_names = f.readlines()
        file_names = [x.replace("\n", "") for x in file_names]
        groundTruth = {}
        for file_name in file_names:
            file_path = os.path.join(cfg.dataset_root, dataset, "groundTruth", file_name)
            with open(file_path, "r") as f:
                gt = f.readlines()
            gt = [actions_dict[x.replace("\n", "")] for x in gt]
            groundTruth[file_name.split(".txt")[0]] = gt

        file_names = [x.replace(".txt", "") for x in file_names]
        test_output = {}
        for file_name in file_names:
            test_output[file_name] = {
                "prediction": predictions[file_name].tolist(),
                "ground_truth": groundTruth[file_name],
            }
        results = get_jg_metrics2(test_output, len(list(actions_dict.keys())))
        writer2.writerow(
            [
                epoch,
                "%.4f" % (results["accu"]),
                "%.4f" % (results["edit"]),
                "%.4f" % (results["F1@0.1"]),
                "%.4f" % (results["F1@0.25"]),
                "%.4f" % (results["F1@0.5"]),
            ]
        )

        if results["F1@0.5"] > max_val:
            max_epoch = epoch
            max_val = results["F1@0.5"]
            max_results = results
            total_patience=dataset_config["total_patience"]
        elif not ("min_patience" in dataset_config.keys()) or epoch >= dataset_config["min_patience"] :
            total_patience-=1
        
        if "cutoff_epoch" in dataset_config.keys():
            if epoch >= dataset_config['cutoff_epoch'] and total_patience==0 and max_epoch != -1:
                break
        else:
            if total_patience==0 and max_epoch != -1:
                break


    f.close()
    f2.close()

    print("EARNED MAXIMUM PERFORMANCE IN EPOCH {}".format(max_epoch))

    f = open(os.path.join(record_dir, "split_{}_colin_best.csv".format(split)), "w")
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(
        [
            "epoch",
            "accu",
            "edit",
            "F1@{}".format(cfg.iou_thresholds[0]),
            "F1@{}".format(cfg.iou_thresholds[1]),
            "F1@{}".format(cfg.iou_thresholds[2]),
        ]
    )
    writer.writerow(
        [   
            max_epoch,
            "%.4f" % (max_results["accu"]),
            "%.4f" % (max_results["edit"]),
            "%.4f" % (max_results["F1@0.1"]),
            "%.4f" % (max_results["F1@0.25"]),
            "%.4f" % (max_results["F1@0.5"]),
        ]
    )
    f.close()

    #### Testing Loop ####
    if "split_strategy" in dataset_config.keys() and dataset_config["split_strategy"] == 3:
        f = open(os.path.join(record_dir, "split_{}_test.csv".format(split)), "w")

        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "accu",
                "edit",
                "F1@{}".format(cfg.iou_thresholds[0]),
                "F1@{}".format(cfg.iou_thresholds[1]),
                "F1@{}".format(cfg.iou_thresholds[2]),
            ]
        )

        predictions = predict_backbone(
            model_name,
            trainer.model,
            model_dir,
            result_dir,
            features_path,
            vid_list_file_tst,
            max_epoch,
            actions_dict,
            device,
            sample_rate,
            mode="testing",
            epoch_level_augmentation=False,
        )

        results = eval_txts(
            cfg.dataset_root, result_dir, dataset, split, model_name, "testing", max_epoch
        )
        with open(vid_list_file_tst, "r") as f:
            file_names = f.readlines()
        file_names = [x.replace("\n", "") for x in file_names]
        groundTruth = {}
        for file_name in file_names:
            file_path = os.path.join(cfg.dataset_root, dataset, "groundTruth", file_name)
            with open(file_path, "r") as f:
                gt = f.readlines()
            gt = [actions_dict[x.replace("\n", "")] for x in gt]
            groundTruth[file_name.split(".txt")[0]] = gt

        file_names = [x.replace(".txt", "") for x in file_names]
        test_output = {}
        for file_name in file_names:
            test_output[file_name] = {
                "prediction": predictions[file_name].tolist(),
                "ground_truth": groundTruth[file_name],
            }
        results = get_jg_metrics2(test_output, len(list(actions_dict.keys())))
        writer.writerow(
            [
                "%.4f" % (results["accu"]),
                "%.4f" % (results["edit"]),
                "%.4f" % (results["F1@0.1"]),
                "%.4f" % (results["F1@0.25"]),
                "%.4f" % (results["F1@0.5"]),
            ]
        )
        f.close()
        with open(os.path.join(record_dir, f"split_test_output_{split}.json"), "w") as f:
            json.dump(test_output, f)
        json_path = os.path.join(record_dir, f"split_test_output_{split}.json")
        save_path = os.path.join(record_dir, f"split_jigsaws_metrics_{split}.txt")
        get_jg_metrics(json_path, save_path, len(list(actions_dict.keys())))

### Run Loop ###
run_enabled = []
datasets = read_dataset_config()
for dataset, dataset_value in datasets.items():
    if "enable_run" in dataset_value.keys() and dataset_value["enable_run"] and dataset == data_name:
        run_enabled.append((dataset, dataset_value))

for item in run_enabled:
    dataset, dataset_config = item
    splits = dataset_config["splits"]
    for split in range(1, splits + 1):
        train_eval_model(dataset, split, dataset_config)