# CertPHash: Towards Certified Perceptual Hashing via Robust Training


<!-- CERTPHASH is the first certified perceptual hashing (PHash) system designed to provide provable robustness against both evasion and collision attacks. Using verifier tools, CERTPHASH is trained on a robust method that incorporates anti-evasion, anti-collision and functionality optimization terms, and is able to withstand adversarial perturbations while maintaining functional utility. Being the first certified PHash system with robust training, it is a powerful tool for detecting illicit content in real-world applications.  -->

<!-- CertPhash is the first perceptual hashing (PHash) system designed with provable robustness against both evasion and collision attacks. -->

CertPhash is the first certified perceptual hashing (PHash) system with robust training. CertPhash includes three different optimization terms, anti-evasion, anti-collision, and functionality. The anti-evasion term establishes an upper bound on the hash deviation caused by input perturbations, the anti-collision term sets a lower bound on the distance between a perturbed hash and those from other inputs, and the functionality term ensures that the system remains reliable and effective throughout robust training.
This artifact includes the source code, dataset, setup, and instructions to implement and evaluate CertPHash. Our artifacts require a Linux machine with 64GB of RAM and a GPU with 40 GB of graphics memory.


## Experiment Setup
All experiments are run on a single NVIDIA A100-PCIE-40GB GPU with CUDA Version 12.1 and Driver Version 530.30.02.

### 1. Setting up Conda Environment
To ensure a consistent and isolated environment for running CERTPHASH, we recommend using Conda. 

If you don't have Conda installed, please follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Once Conda is installed, create and activate a new virtual environment named `certphash`:

```bash
conda create -n certphash python=3.10 -y
conda activate certphash
```

- **Install basic dependencies**:
   - Install the required dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```
- **Install auto_LiRPA dependencies**:
   - Clone the [auto_LiRPA repository](https://github.com/Verified-Intelligence/auto_LiRPA) and install its dependencies by running:
     ```bash
     git clone git@github.com:Verified-Intelligence/auto_LiRPA.git
     cd auto_LiRPA
     python setup.py install
     ```
     
This will install all the necessary packages for training and evaluation. 

### 2. Preparing datasets
- We provide the [download link](https://drive.google.com/file/d/11LBjSRM-tqhJOKcRbxO8_8ZEabPdepUD/view?usp=sharing) for the datasets (images and phashes) we used. 

- After downloading the above link, you will need to unzip and named the folder as `data`, then replace the current folder `train_verify/data` with the downloaded one.

- Note that for the COCO dataset, you will need to perform an *additional* step as described below.

- The file hierarchy is specified by `./train_verify/data/put_data_here_follow_this.txt`, such as:

```
-coco100x100
    -.jpg
-coco100x100_val
    -.jpg
-coco-train.csv
-coco-val.csv
-mnist
    -testing
        -.jpg
    -training
        -.jpg
    -mnist_test.csv
    -mnist_train.csv
```


We use the following four datasets:

- **COCO**: 
  - Download images from this [link](https://github.com/anishathalye/ribosome/releases/download/v1.0.0/coco100x100.tar.gz) and unzip the file (named `coco100x100`).
  - Place the folder under `train_verify/data`.

- **MNIST**: 
  - A folder named "mnist" containing images and hashes is required.

- **CelebA**: 
  - A random selection of 50,000 samples is used for training, and 6,000 samples are used for verification.

- **NSFW-56K**: 
  - The NSFW-56K dataset contains 56,000 images depicting explicit and unsafe content, all resized to 64x64 pixels. 
  - This dataset is used in combination with CelebA to evaluate CertPhash's ability to detect NSFW content in real-world scenarios.
  - Due to ethical concerns, we do not provide a download link for this dataset. Instead, follow this [repo](https://github.com/alex000kim/nsfw_data_scraper) to scrape images labeled as `porn`, and randomly select 56,000 images to form the NSFW dataset.
  - Name the folder containing these 56,000 images as `porn_resized` and place it under `train_verify/data`.
  - Randomly split the dataset into 50,000 for training and 6,000 for testing (See the `nsfw_train.csv` and `nsfw_val.csv` files downloaded from the first link for reference on the format. You will need to replace the NSFW part after collecting your own datasets).
  - Optionally, you can place a copy of the 6,000 testing images in the `porn_val` folder.


### 3. Downloading our trained model
We have released our certified robust trained model, adversarial trained model, and non-robust trained model in this [Download link.](https://drive.google.com/drive/folders/1b7RbO-uDdlvsxgsxE4H-tjrdGVx7pVRu?usp=sharing)
Please download this folder named `saved_models` and replace it with the existing `./train_verify/saved_models`.


## Environment and Functionality Test
### 1. Testing the environment
To test whether the environment has been set up correctly, please run the following for a demo robust training:
```bash
bash ./train_verify/test.sh
```
The expected output for `./train_verify/test.sh` should be like `./train_verify/test/train_log.txt`, a.k.a. something like the following:
```angular2html
Epoch 1, learning rate [0.0005], dir one_epoch_test
[ 1]: eps=0.00000048 active=0.3465 inactive=0.6532 Loss=0.4681 Rob_Loss=0.4681 Err=1.0000 Rob_Err=1.0000 L_tightness=0.5494 L_relu=0.0023 L_std=0.8852 loss_reg=0.5518 grad_norm=15.2293 wnorm=13.1858 Time=0.0517
...
Test without loss fusion
[ 1]: eps=0.00000048 active=0.3509 inactive=0.6491 Loss=0.4642 Rob_Loss=0.4642 Err=1.0000 Rob_Err=1.0000 L_tightness=0.0000 L_relu=0.0000 L_std=0.8729 loss_reg=0.0000 wnorm=14.8170 Time=0.0218
```

### 2. Testing the functionality

- To test if our certified robust trained model functions as expected, please first run:
```bash
python ./attack/benign0_func_check.py --dataset='coco_val' --target='photodna_nn_cert_ep1' --model='../train_verify/saved_models/coco_photodna_ep1/ckpt_best.pth'
```
The expected output from console contains the name of transformations and a progress bar, for example:
```angular2html
Original: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 22310.13it/s]
Rotate: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 32099.27it/s]
...
```
It will take around five minutes to calculate the PHash from different levels of different transformations. The generated hashes will be saved under folder `./attack/func_logs/coco_val_photodna_nn_cert_ep1`.

- Then calculate the ROC-AUC via:
```bash
python ./attack/benign0_func_AUC.py --dataset='coco_val' --target='photodna_nn_cert_ep1'
```
The expected output from console contains ROC-AUC of transformations tested in our papers RQ1. For instance, the output for the Hue transformation should be as follows:
```angular2html
...
hue:  -180      -150      -120      -90       -60       -30       0         30        60        90        120       150       
ROC AUC: 1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    
Mean ROC AUC: 1.0000, Std ROC AUC: 0.0000
...
```
The full results will be saved in `./attack/func_logs/coco_val_photodna_nn_cert_ep1/coco_val_results.txt`. These results should align with our paper's RQ1.




# Other Experiments

## Training and verification

Instead of using our models, you can also train and verify from scratch.

### Normal and robust training
Both normal and robust training is conducted by executing `./train_verify/train.py`. Different datasets require different configuration files, and separate configuration files are used for normal and robust training. We provide the training script in `./train_verify/train.sh`, which includes command lines for both **Robust Training** and **Normal Training** using the COCO and MNIST dataset. 

To illustrate, below are some examples on how to perform robust and normal training on different datasets.

To perform robust training with a perturbation epsilon of 0.0078, you may run the command:
```bash
python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0078 --dir=saved_models/coco_photodna_ep2 --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160 --model='resnet_v5' --lr=5e-4
```

To perform normal training on the MNIST dataset, you may run the command:
```bash
python train.py --config=config/mnist.normal.json --dir=saved_models/mnist_pdq_ep0 --lr-decay-milestones=15,18 --lr-decay-factor=0.2 --num-epochs=20  --model='resnet' --lr=5e-4 --natural
```

### Performing robust verification
Robust verification involves verifying the CNER(Certified No Evasion Rate) and CNCR(Certified No Collision Rate) of the trained model. To derive the CNER, we run the `./train_verify/verify_evasion.py`, specifying the perturbation epsilon and model path of the assessed model in the parameters. 

An example of verifying the CNER for a perturbation epsilon of 0.0312 for a certified robust trained on the COCO dataset could be like this:
```bash
python verify_evasion.py --data=coco --epsilon=0.0312 --model='saved_models/coco_photodna_ep8/ckpt_best.pth'
```

To derive the CNCR, we run `verify_preimage.py` under `./train_verify`. An example of verifying the CNCR for a model trained on the MNIST dataset could be like this:
```bash
python verify_preimage.py
```
You may find more scripts for model verification in `./train_verify/verify.sh`.



## PHash Functionality Evaluation
We perform benign non-adversarial transformations on images such as rotation and brightness alterations and verify the ROC-AUC score. 
To perform the verification, follow the steps below. The process involves two key scripts: 
1. `./attack/benign0_func_check.py` – Generates hashes for transformed images.
2. `./attack/benign0_func_AUC.py` – Calculates the ROC-AUC score based on the generated hashes.

The results will be saved under `./attack/func_logs/`. 

You will need to navigate to `./attack` folder first before running the below scripts. The detailed scripts can be found in `./attack/benign0.sh` with detailed comments. We also list the step-by-step instructions below.

### Phash models trained by us

Please run the following script one by one if you have downloaded our models or trained your own models.

#### Certified robust trained (CertPHash)
This script is the same as the previous one in *Testing the functionality*.
```bash
python benign0_func_check.py --dataset='coco_val' --target='photodna_nn_cert_ep1' --model='../train_verify/saved_models/coco_photodna_ep1/ckpt_best.pth'
python benign0_func_AUC.py --dataset='coco_val' --target='photodna_nn_cert_ep1'
```

#### Normal trained (BasePHash)
```bash
python benign0_func_check.py --dataset='coco_val' --target='photodna_nn_ep0' --model='../train_verify/saved_models/coco_photodna_ep0/ckpt_best.pth'
python benign0_func_AUC.py --dataset='coco_val' --target='photodna_nn_ep0'
```

#### Adversarially trained (AdvPHash)
```bash
python benign0_func_check.py --dataset='coco_val' --target='photodna_nn_adv' --model='../train_verify/saved_models/base_adv/coco-pdq-ep8-photodna.pt'
python benign0_func_AUC.py --dataset='coco_val' --target='photodna_nn_adv'
```


### (Optional) Existing PHash algorithms/models from other sources
We also provide the implementation and evaluation scripts for existing PHash systems (PhotoDNA, PDQ, NeuralHash). However, we cannot directly provide the extracted algorithms/models (following existing repos) due to copyright issues.

#### PDQ

- Please follow the [repo](https://github.com/faustomorales/pdqhash-python) to set up the PDQ Hash Model. Specifically, you will need to first run `pip install pdqhash` and then execute `python get_pdq_hash.py`. 

- For evaluating PDQ on the COCO validation dataset, use these commands:
```bash
python benign0_func_check.py --dataset='coco_val' --target='pdq'
python benign0_func_AUC.py --dataset='coco_val' --target='pdq'
```

#### PyPhotoDNA
- We follow the [repo](https://github.com/jankais3r/pyPhotoDNA) to set up the PhotoDNA model. To set up this model, you will have to:
  - Navigate to folder via `cd ../generate_phash` (assuming you are in the `attack` folder)
  - Install `wine64` if you are on a Linux via `sudo apt-get install wine-stable-dev`
  - Run `install.bat` if you are on Windows, or `install.sh` if you are on a Linux. Then you should have a file with suffix `.dll` which contains the PhotoDNA.
  - (Skip this step for now) Once the setup is complete, you can run `WINEDEBUG=-all wine64 python-3.9.12-embed-amd64/python.exe get_photodna_hash.py` to generate hashes. Remember to modify the image folder name and saved .csv file name in `get_photodna_hash.py` if you want to generate PHashes for your own datasets (Note: this step is needed if you want to generate the phash labels for NSFW datasets you collect).

- For PhotoDNA evaluation on the COCO validation dataset, follow these steps:
  - First, navigate back to `attack` folder via `cd ../attack`, generate and save the transformed images, then use generate_phash to get the .csv file.
  ```bash
  python benign0_func_check.py --dataset='coco_val' --target='photodna'
  ```
  - Second, navigate to `generate_phash` via `cd ../generate_phash` and then run
  ```bash
  main_root="../attack/func_logs/coco_val_photodna"
  for subroot in "$main_root"/*; do
    if [ -d "$subroot" ]; then
        echo "Processing folder: $subroot"
        WINEDEBUG=-all wine64 python-3.9.12-embed-amd64/python.exe get_photodna_hash.py --root="$subroot" --benign_func_test
        if [ $? -eq 0 ]; then
            echo "Completed successfully: $subroot"
        else
            echo "Error processing: $subroot" >&2
        fi
    fi
  done
  ```
  - Third, navigate back to `attack` via `cd ../attack` and then run
  ```bash
  python benign0_func_AUC.py --dataset='coco_val' --target='photodna'
  ```



#### NeuralHash
Please follow the guide provided by this [repo](https://github.com/ml-research/Learning-to-Break-Deep-Perceptual-Hashing), utilizing [AppleNeuralHash2ONNX](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX) to extract the NeuralHash model. For copyright reasons, we are unable to provide any NeuralHash files or models. Hash generation is under the `attack` folder.

To extract the NeuralHash model from a recent macOS or iOS build, please follow the conversion guide provided by AppleNeuralHash2ONNX. We will not provide any NeuralHash files or models, neither this repo nor by request. After extracting the onnx model, put the file model.onnx into `attack/models/`. Further, put the extracted Core ML file neuralhash_128x96_seed1.dat into `attack/models/coreml_model/`.

To convert the onnx model into PyTorch, run the following command after creating the `models` folder and placing model.onnx into the folder. The converted files will be stored at `attack/models/model.pth`:

```bash
python utils/onnx2pytorch.py
```

After getting and correctly placing the two model files, to evaluate Neuralhash with the COCO validation dataset, run the following commands:
```bash
python benign0_func_check.py --dataset='coco_val' --target='neuralhash_nn'
python benign0_func_AUC.py --dataset='coco_val' --target='neuralhash_nn'
```


### Empirical attacks
#### Empirical collision attacks
To perform empirical collision attacks, first derive the hash values of a dataset of choice using `utils/compute_dataset_hashes.py` (you only need to do so once), and then run `attack/adv1_collision_attack.py` to perform empirical collision attacks. 

Please refer to the scripts provided by `attack/adv1.sh` for more detailed command lines.

#### Empirical evasion attacks
Run `attack/adv2_evasion_attack.py` to perform empirical evasion attacks.

Again, please refer to the scripts provided by `attack/adv2.sh` for more detailed examples.

### Real-world evaluations on NSFW data
In our work, we adopt the NSFW dataset together with the CelabA dataset to assess our model's ability in NSFW detection. Please follow the instructions in the data preparation section for both datasets. 

After getting the datasets, please run the following script to train the model:

```bash
python train.py --method=fast --config=config/nsfw.crown-ibp.json --eps=0.0039 --dir=saved_models/NSFW_photodna_ep1 --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160  --model='resnet_v5' --lr=5e-4
```

And run the following script to test the functionality:
```angular2html
python benign0_func_check.py --dataset='nsfw' --target='photodna_nn_cert_ep1' --model='../train_verify/saved_models/NSFW_photodna_ep1/ckpt_best.pth'
python benign0_func_AUC.py --dataset='nsfw' --target='photodna_nn_cert_ep1'
```


<!-- ## Citation
Please don't forget to cite us if you use our code! -->

## Implementation Credits
Some of our implementations are built upon other repositories. We sincerely appreciate their contributions, and acknowledge these sources below. 
- AppleNeuralHash2ONNX: https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX
- PyPhotoDNA: https://github.com/jankais3r/pyPhotoDNA
- PDQHash: https://github.com/faustomorales/pdqhash-python
- Learning to Break Deep Perceptual Hashing: https://github.com/ml-research/Learning-to-Break-Deep-Perceptual-Hashing
- Fast Certified Robust Training: https://github.com/shizhouxing/Fast-Certified-Robust-Training
- Auto-LiRPA: https://github.com/Verified-Intelligence/auto_LiRPA
- Alpha-beta-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- Ribosome: https://github.com/anishathalye/ribosome?tab=readme-ov-file