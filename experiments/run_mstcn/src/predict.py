import torch
import numpy as np
import os
            

def predict_backbone(name, model, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, mode, epoch_level_augmentation = False):
    model.eval()
    final_predictions = {}
    with torch.no_grad():
        model.to(device)
        model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            vid_name = vid.replace(".txt", "")
            if mode == "testing" or mode == "validation":
                features = np.load(features_path + vid_name + '.npy')
            elif not epoch_level_augmentation:
                features = np.load(features_path + vid_name + '.npy')
            else:
                features = np.load(os.path.join(features_path, str(epoch), vid_name + '.npy'))
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            predictions = model(input_x, torch.ones(input_x.size(), device=device))
            _, predicted = torch.max(predictions[-1].data, 1)
                
            predicted = predicted.squeeze()
            recognition = []
            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
            f_name = vid.split('/')[-1].split('.txt')[0]
            if not os.path.exists(os.path.join(result_dir, mode, str(epoch))):
                os.makedirs(os.path.join(result_dir, mode, str(epoch)))
            f_ptr = open(result_dir + "/" + mode + "/" + str(epoch) + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
            final_predictions[vid.split(".txt")[0]] = predicted
        

    return final_predictions