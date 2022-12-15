# importing the module 
from xmlrpc.client import boolean 
import ase
import numpy as np
from ase import Atoms, Atom
from amptorch.trainer import AtomsTrainer
import sys
import torch
import os
import csv
import time
import ase.io

def log(log_filename, message):
    f = open(log_filename, "a")
    f.write(message)
    f.close()
    return

def load_training_data(training_filename, test_filename):

    training_list = ase.io.read(training_filename, index=":")
    test_list = ase.io.read(test_filename, index=":")

    return training_list, test_list


def load_linear_fit_result(linear_fit_result_filename):

    correction_dict = {}
    with open(linear_fit_result_filename) as fp: 
        Lines = fp.readlines() 
        for line in Lines: 
            temp = line.split()
            print(line.split())
            correction_dict[temp[0]]=float(temp[1])
    return correction_dict

def predict_data(trainer, test_images, folder_name, image_type = "test"):
    cwd = os.getcwd()
    
    predictions = trainer.predict(test_images, disable_tqdm = False)
    true_energies = np.array([image.get_potential_energy() for image in test_images])
    pred_energies = np.array(predictions["energy"])
    print(true_energies.shape)
    print(pred_energies.shape)

    # pickle.dump( true_energies, open( "{}_true_energies.p".format(image_type), "wb" ) )
    # pickle.dump( pred_energies, open( "{}_pred_energies.p".format(image_type), "wb" ) )

    mae_result = np.mean(np.abs(true_energies - pred_energies))
    print("Energy MAE:", mae_result)
    os.chdir(folder_name)
    list_of_error_per_atom = []
    with open('{}_prediction_result.csv'.format(image_type), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i, image in enumerate(test_images):
            num_atoms = len(image.get_atomic_numbers())
            total_energy_pred = pred_energies[i]
            total_energy_true = true_energies[i]
            
            error = pred_energies[i] - true_energies[i]
            per_atom_error = error / num_atoms
            list_of_error_per_atom.append(per_atom_error)
            writer.writerow([i, num_atoms,true_energies[i], pred_energies[i], total_energy_true, total_energy_pred,
                             error, per_atom_error, abs(error), abs(per_atom_error)])
    os.chdir(cwd)
    return mae_result


dataset_name = sys.argv[1]
nsigmas = int(sys.argv[2])
MCSHs_index = int(sys.argv[3])
rs = 1.0
NN_index = int(sys.argv[4])
isSH = True
seed = 1
activation_fx = str(sys.argv[5])

num_gpu = torch.cuda.device_count()
print("****\n Found {} GPUs \n****\n\n".format(num_gpu))

# train and test filenames in format supposted by ase
train_filename = <intput_dataset>
test_filename = <intput_dataset>


cwd = os.getcwd()
folder_name = "./trial_{}/test_ordernorm_sigma{}linwise_MCSH{}_NN{}_constrs{}_sh{}".format(dataset_name, nsigmas, MCSHs_index,NN_index,rs,int(isSH))
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)
train_images, test_images = load_training_data(train_filename, test_filename)

# sigmas = np.logspace(-1.0,0.3,nsigmas,endpoint=True)
sigmas = np.linspace(0,2.0,nsigmas+1,endpoint=True)
sigmas = sigmas[1:]
print(sigmas)
MCSHs_dict = {
0: { "orders": [0], "sigmas": sigmas,},
1: { "orders": [0,1], "sigmas": sigmas,},
2: { "orders": [0,1,2], "sigmas": sigmas,},
3: { "orders": [0,1,2,3], "sigmas": sigmas,},
4: { "orders": [0,1,2,3,4], "sigmas": sigmas,},
5: { "orders": [0,1,2,3,4,5], "sigmas": sigmas,},
6: { "orders": [0,1,2,3,4,5,6], "sigmas": sigmas,},
7: { "orders": [0,1,2,3,4,5,6,7], "sigmas": sigmas,},
8: { "orders": [0,1,2,3,4,5,6,7,8], "sigmas": sigmas,},
9: { "orders": [0,1,2,3,4,5,6,7,8,9], "sigmas": sigmas,},
}

MCSHs = MCSHs_dict[MCSHs_index]


GMP = {   "MCSHs": MCSHs,
            "atom_gaussians": {
                        "H": "./H_pseudodensity_2.g",
                        "C": "./C_pseudodensity_4.g",
                        "O": "./O_pseudodensity_4.g",
                  },
            "cutoff": 10.0,
            "rs_setup": {"setup": "constant", "rs":rs}, 
            "square":False,
            "solid_harmonics": isSH,
}


elements = ["H","C","O"]

NN_dict = {
0:[256,128],
1:[32,16,16,8],
2:[64,32,32,8],
3:[128,64,64,32,32,8],
4:[256,128,128,64,64,32,32,8],
5:[512,256,256,128,128,64,64,32,32,8],
11: [32,32,32],
12: [64,64,64],
13: [128,128,128],
14: [256,256,128,128],
15: [512,256,256,128,128,128],

21:[32,16,8],
22:[64,32,16,8],
23:[128,64,32,16,8],
24:[256,128,64,32,16,8],
25:[512,256,128,64,32,16,8],
26:[1024,512,256,128,64,32,16,8],

30: [16,16,16],
31: [32,32,32],
32: [64,64,64],
33: [128,64,64],
34: [256,128,64],
35: [512,256,128,64],
36: [1024,512,128,64],

212: [64, 32, 64],
}

hidden_layers = NN_dict[NN_index]
#hidden_layers = [256, 128, 64, 8]
#hidden_layers = [512,512,256, 128, 64, 8]
#hidden_layers = [256, 512, 256,128, 128, 64, 8]

activation_fx_dict = {
    "GELU": torch.nn.GELU,
    "Tanh": torch.nn.Tanh,
}


config = {
    "model": {"name":"singlenn",
                  "get_forces": False, 
                  "hidden_layers": hidden_layers, 
                  "activation":activation_fx_dict[activation_fx],
                  "batchnorm": True,
                  "initialization":"xavier",
                  },
    "optim": {
            "gpus":num_gpu,
            "force_coefficient": 0.0,
            "lr": 5e-4,
            "batch_size": 256,
            "epochs": 3500,
            "loss": "mae",
            "scheduler": {"policy": "StepLR", "params": {"step_size": 100, "gamma": 0.7}}
    },
    "dataset": {
            "raw_data": train_images,
            "val_split": 0.1,
            "elements": elements,
            "fp_scheme": "gmpordernorm",
            "fp_params": GMP,
            "save_fps": True,
            "scaling": {"type": "normalize", "range": (-1, 1),"elementwise":False}
        },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "data{}-ordernorm_sigma{}logwide_MCSH{}_NN{}_constrs{}_sh{}".format(dataset_name, nsigmas, MCSHs_index,NN_index,rs,int(isSH)),
        "verbose": True,
        "logger": False,
        "dtype": torch.DoubleTensor,
        "early_stopping":True,
        "early_stoppping_patience": 100
    },
}


trainer = AtomsTrainer(config)
print("training")
os.chdir(cwd)
tr_start = time.time()
trainer.train()
training_time = time.time() - tr_start

print("end training")

pr_start = time.time()
train_mae = predict_data(trainer, train_images, folder_name, image_type = "train")
test_mae = predict_data(trainer, test_images, folder_name, image_type = "test")
predict_time = time.time() - pr_start

os.chdir(folder_name)
message = "ordernorm sqrt\t{}linwide\t{}\t{}\t{}\t{}\t{}\tconstrs{}\ttraining:{}\tpred:{}\tcp_dir:{}\n".format(
                    nsigmas, 
                    MCSHs_index, 
                    hidden_layers,
                    activation_fx,
                    train_mae,
                    test_mae,
                    rs,
                    training_time,
                    predict_time,
                    trainer.cp_dir
                    )
log("../pseudo_train_result.dat",message)
os.chdir(cwd)
