# VTS (DSH with ViT-cls Backbone - ICME 2022)
# paper [Vision Transformer Hashing for Image Retrieval, ICME 2022](https://arxiv.org/pdf/2109.12564.pdf)
# DSH basecode considered from https://github.com/swuxyj/DeepHash-pytorch

from utils.tools import *

import torch
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as TF

from TransformerModel.modeling_cls import VisionTransformer, VIT_CONFIGS

from PIL import Image, ImageOps

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import os
import time
import numpy as np

import getopt, sys

torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        "dataset": "cifar10",
        # "dataset": "cifar10-2",
        #"dataset": "coco",
        # "dataset": "nuswide_21",
        # "dataset": "imagenet",
        #"net": AlexNet, "net_print": "AlexNet",
        #"net":ResNet, "net_print": "ResNet",
        "net": VisionTransformer, "net_print": "ViT-B_32", "model_type": "ViT-B_32", "pretrained_dir": "/content/drive/MyDrive/pre-train/ViT-B_32.npz",
        # "net": VisionTransformer, "net_print": "ViT-B_16", "model_type": "ViT-B_16", "pretrained_dir": "/content/drive/MyDrive/pre-train/ViT-B_16.npz",
        
        "bit_list": [64,16],
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5}},
        "device": torch.device("cuda"), 
        "save_path": "Checkpoints_Results",
        "epoch": 150, "test_map": 30, "batch_size": 128, "resize_size": 256, "crop_size": 224,
        "info": "DSHcls", "alpha": 0.1,
    }
    config = config_dataset(config)
    return config

def retrieve_top_best_case(config, top_k=10, match_threshold=5, total_best_case = 200):
    dataset_binary = None
    dataset_image_path = None

    with open(config["data"]["database"]["list_path"]) as database_info:
        list_database_path = database_info.readlines()
        dataset_binary = np.array([np.array([int(la) for la in val.split()[1:]]) for val in list_database_path])
        dataset_image_path = [str(val.split()[0])  for val in list_database_path]
        dataset_label = [str(val.split()[0].split("/")[0])  for val in list_database_path]

    test_binary = None
    test_image_path = None
    with open(config["data"]["test"]["list_path"]) as test_info:
        list_test_path = test_info.readlines()
        test_binary = np.array([np.array([int(la) for la in val.split()[1:]]) for val in list_test_path])
        test_image_path = [str(val.split()[0])  for val in list_test_path]
        test_label = [str(val.split()[0].split("/")[0])  for val in list_test_path]
    # print(test_label)
    top_n_best_result = 0
    m_query = 5
    curr_index = 0
    plt.figure(figsize=(40, 20), dpi=50)
    font_size = 30
    # print(len(dataset_binary[0]), len(test_binary[0]))
    for test_index, sample in enumerate(test_binary):
        hamm = CalcHammingDist(sample,  dataset_binary)
        ind = np.argsort(hamm)[:top_k]
        return_image_list = np.array(dataset_label)[ind].tolist()
        return_image_list_path = np.array(dataset_image_path)[ind].tolist()
        c1 = Counter([test_label[test_index]])
        c2 = Counter(return_image_list)
        common = set(c1).intersection(set(c2))
        total_match = sum(max(c1[n], c2[n]) for n in common)
        if total_match > match_threshold:
            top_n_best_result +=1
            full_test_image_path = os.path.join(test_image_path[test_index])
            plt.subplot(m_query, top_k + 1, curr_index * (top_k+1) + 1)
            full_test_image_path = os.path.join(os.path.join(config["data_path"], "test"), test_image_path[test_index])
            img = Image.open(full_test_image_path).convert('RGB').resize((128, 128))
            plt.imshow(img)
            plt.axis('off')
            plt.text(5, 145, 'query image', size=font_size)
            # print(test_index, " ",c1, " ", c2, " total match:", total_match)
            print(curr_index, test_image_path[test_index], ". Result", return_image_list)
            for index, database_image_path in enumerate(return_image_list_path):
                plt.subplot(m_query, top_k + 1, curr_index * (top_k+1) + index + 2)
                full_database_image_path = os.path.join(os.path.join(config["data_path"], "train"), database_image_path)
                img = Image.open(full_database_image_path).convert('RGB').resize((120, 120))
                plt.axis('off')
                plt.imshow(img)
            curr_index += 1
        if curr_index == m_query:
            curr_index = 0
            viz_file_name = str(int(total_best_case/top_n_best_result)) + ".png"
            dataset_viz_folder = "/content/drive/MyDrive/Hashing/visualization/" + config["dataset"]
            plt.savefig(os.path.join(dataset_viz_folder, viz_file_name))
        if top_n_best_result >= total_best_case:
            break
        
def retrieve_image(config, bit, query_image, top_k = 5, ):
    step = [transforms.CenterCrop(config["crop_size"])]
    transform_norm = transforms.Compose([transforms.Resize(config["resize_size"])]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])
    m = 5
    n = 8
    plt.figure(figsize=(40, 20), dpi=50)
    font_size = 30
    num_classes = config["n_class"]
    hash_bit = bit
    device = config["device"]
    if "ViT" in config["net_print"]:
        vit_config = VIT_CONFIGS[config["model_type"]]
        net = config["net"](vit_config, config["crop_size"], zero_head=True, num_classes=num_classes, hash_bit=hash_bit).to(device)
    else:
        net = config["net"](bit).to(device)
    dataset_binary = None
    dataset_image_path = None
    with open(config["data"]["database"]["list_path"]) as database_info:
        list_database_path = database_info.readlines()
        dataset_binary = np.array([np.array([int(la) for la in val.split()[1:]]) for val in list_database_path])
        dataset_image_path = [str(val.split()[0])  for val in list_database_path]
    net.eval()
    bs = []
    image = Image.open(open(query_image,'rb'))
    transform_img = transform_norm(image).float()
    img_normalized = transform_img.unsqueeze_(0)
    output = net(img_normalized.to(device)).data.cpu()
    query_binary = torch.Tensor.int(output.sign()).numpy()[0]
    hamm = CalcHammingDist(query_binary,  dataset_binary )
    ind = np.argsort(hamm)[:top_k]
    return_img_list = np.array(dataset_image_path)[ind].tolist()
    return return_img_list

def inference_dataset(config, dataset_loader, bit):
    dataset_path = "/content/drive/MyDrive/Hashing/data/cifar10"
    num_classes = config["n_class"]

    
    
    config["num_train"] = num_train

    hash_bit = bit
    device = config["device"]
   

    if "ViT" in config["net_print"]:
        vit_config = VIT_CONFIGS[config["model_type"]]
        net = config["net"](vit_config, config["crop_size"], zero_head=True, num_classes=num_classes, hash_bit=hash_bit).to(device)
    else:
        net = config["net"](bit).to(device)
    
    if not os.path.exists(config["save_path"]): 
            os.makedirs(config["save_path"])
    best_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-BestModel.pt")
    trained_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-IntermediateModel.pt")
    results_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + ".txt")
    f = open(results_path, 'a')

    if os.path.exists(trained_path):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(trained_path)
        net.load_state_dict(checkpoint['net'])
        Best_mAP = checkpoint['Best_mAP']
        start_epoch = checkpoint['epoch'] + 1
    else:
        if "ViT" in config["net_print"]:
            print('==> Loading from pretrained model..')
            net.load_from(np.load(config["pretrained_dir"]))
    
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)



def extract_database(config, bit):
    step = [transforms.CenterCrop(config["crop_size"])]
    transform_norm = transforms.Compose([transforms.Resize(config["resize_size"])]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])
    num_classes = config["n_class"]
    hash_bit = bit
    device = config["device"]

    if "ViT" in config["net_print"]:
        vit_config = VIT_CONFIGS[config["model_type"]]
        net = config["net"](vit_config, config["crop_size"], zero_head=True, num_classes=num_classes, hash_bit=hash_bit).to(device)
    else:
        net = config["net"](bit).to(device)

    if not os.path.exists(config["save_path"]): 
            os.makedirs(config["save_path"])
    best_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-BestModel.pt")
    trained_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-IntermediateModel.pt")
    results_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + ".txt")
    f = open(results_path, 'a')

    data_path = config["data_path"]
    dataset_path = os.path.join(data_path, "test")
    dataset_list_path = config["data"]["test"]["list_path"]
    print("dataset_list_path", dataset_list_path, dataset_path)
    index = 0

    if os.path.exists(trained_path):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(trained_path)
        net.load_state_dict(checkpoint['net'])
        Best_mAP = checkpoint['Best_mAP']
        print("Best mAP received", Best_mAP)
        start_epoch = checkpoint['epoch'] + 1
    else:
        if "ViT" in config["net_print"]:
            print('==> Loading from pretrained model..')
            net.load_from(np.load(config["pretrained_dir"]))
            
    net.eval()

    with open(dataset_list_path, "w") as dataset_file:
        for (root, labels, _) in os.walk(dataset_path, topdown=True):
            for label in tqdm(labels, desc = 'dirs'):
                label_path = os.path.join(root, label)
                for images in os.listdir(label_path):
                    
                    image_path = os.path.join(label, images)
                    full_image_path = os.path.join(dataset_path, image_path)
                    img = Image.open(full_image_path)
                    transform_img = transform_norm(img).float()
                    img_normalized = transform_img.unsqueeze_(0)
                    output = net(img_normalized.to(device)).data.cpu()
                    binary = torch.Tensor.int(torch.sign(output)).numpy()[0]
                   
                    binary_str = str(' '.join([str(binary[bit]) for bit in range(len(binary))]))
                   
                    line = image_path + " " + binary_str
                    dataset_file.write(line + "\n")
                    index += 1
                    

def train_val(config, bit):
    start_epoch = 1
    Best_mAP = 0
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    
    num_classes = config["n_class"]
    hash_bit = bit
    
    if "ViT" in config["net_print"]:
        vit_config = VIT_CONFIGS[config["model_type"]]
        net = config["net"](vit_config, config["crop_size"], zero_head=True, num_classes=num_classes, hash_bit=hash_bit).to(device)
    else:
        net = config["net"](bit).to(device)
    
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    best_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-BestModel.pt")
    trained_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-IntermediateModel.pt")
    results_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + ".txt")
    f = open(results_path, 'a')
    
    if os.path.exists(trained_path):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(trained_path)
        net.load_state_dict(checkpoint['net'])
        Best_mAP = checkpoint['Best_mAP']
        start_epoch = checkpoint['epoch'] + 1
    else:
        if "ViT" in config["net_print"]:
            print('==> Loading from pretrained model..')
            net.load_from(np.load(config["pretrained_dir"]))
    
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = DSHLoss(config, bit)

    for epoch in range(start_epoch, config["epoch"]+1):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["net_print"], epoch, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        train_loss = 0
        for image, label, ind in train_loader:

            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            u = net(image)
            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        f.write('Train | Epoch: %d | Loss: %.3f\n' % (epoch, train_loss))

        if (epoch) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])
            
            if mAP > Best_mAP:
                Best_mAP = mAP
                P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())
                print(f'Precision Recall Curve data:\n"DSH":[{P},{R}],')
                f.write('PR | Epoch %d | ' % (epoch))
                for PR in range(len(P)):
                    f.write('%.5f %.5f ' % (P[PR], R[PR]))
                f.write('\n')
            
                print("Saving in ", config["save_path"])
                state = {
                    'net': net.state_dict(),
                    'Best_mAP': Best_mAP,
                    'epoch': epoch,
                }
                torch.save(state, best_path)
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch, bit, config["dataset"], mAP, Best_mAP))
            f.write('Test | Epoch %d | MAP: %.3f | Best MAP: %.3f\n'
                % (epoch, mAP, Best_mAP))
            print(config)
        
            state = {
            	'net': net.state_dict(),
            	'Best_mAP': Best_mAP,
            	'epoch': epoch,
            }
            torch.save(state, trained_path)
    f.close()


class DSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = config["alpha"] * (1 - u.sign()).abs().mean()

        return loss1 + loss2


if __name__ == "__main__":
    argumentList = sys.argv[1:]
    try:
        options = "q:"
        long_options = ["query_image="]
        config = get_config()
        query_image = ""
        opts, args = getopt.getopt(argumentList, options, long_options)
        for opt, arg in opts: 
            if opt in ['-q', '--query_image']: 
                query_image = arg
        dataset_loader, index = get_database(config)
        list_image_retrieved = retrieve_image(config,query_image= query_image, bit=config["bit_list"][0], top_k = 10)
        sys.stdout.write(str(list_image_retrieved))
        sys.stdout.flush()
        sys.exit(0)
    except getopt.error as err:
        print(str(err))

