# CERTPHASH: Towards Certified Perceptual Hashing via Robust Training
**Overview:**

<!-- CERTPHASH is the first certified perceptual hashing (PHash) system designed to provide provable robustness against both evasion and collision attacks. Using verifier tools, CERTPHASH is trained on a robust method that incorporates anti-evasion, anti-collision and functionality optimization terms, and is able to withstand adversarial perturbations while maintaining functional utility. Being the first certified PHash system with robust training, it is a powerful tool for detecting illicit content in real-world applications.  -->

CERTPHASH is the first perceptual hashing (PHash) system designed with provable robustness against both evasion and collision attacks. Unlike traditional PHash systems, CERTPHASH incorporates robust training methods that optimize for both security and functionality. It is specifically trained to withstand adversarial perturbations, making it resilient to content manipulation. By incorporating techniques for anti-evasion and anti-collision, CERTPHASH ensures that hash values remain stable and accurate, even under attack. Extensive experiments have shown that it is a powerful tool for detecting tampered images in digital forensics and identifying illicit content on social media platforms, providing a powerful and dependable solution for modern challenges.


## Installing environment
Follow the steps below to set up the environment:

### 1. **Setting up Conda Environment**
To ensure a consistent and isolated environment for running CERTPHASH, we recommend using Conda. 

If you don't have Conda installed, please follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Once Conda is installed, create and activate a new virtual environment named `certphash`:

```bash
conda create -n certphash python=3.10 -y
conda activate certphash
```

2. **Install basic dependencies**:
   - Install the required dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Install auto_LiRPA dependencies**:
   - Clone the [auto_LiRPA repository](https://github.com/Verified-Intelligence/auto_LiRPA) and install its dependencies by running:
     ```bash
     git clone git@github.com:Verified-Intelligence/auto_LiRPA.git
     cd auto_LiRPA
     python setup.py install
     ```

This will install all the necessary packages for training and evaluation. 

## Preparing datasets
The file hierarchy is specified by `./train_verify/data/put_data_here_follow_this.txt`.

We provide the [download link](https://drive.google.com/file/d/11LBjSRM-tqhJOKcRbxO8_8ZEabPdepUD/view?usp=sharing) for the datasets (images and phashes) we used. 

After downloading the above link, you will need to unzip and named the folder as `data`, then replace the current folder `train_verify/data` with the downloaded one.



Note that for the COCO dataset, you will need to perform an additional step as described below:

For robust training and evaluation, we use the following four datasets:

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
  - Randomly split the dataset into 50,000 for training and 6,000 for testing (See the `nsfw_train.csv` and `nsfw_val.csv` files downloaded from the first link for reference on the format).
  - Optionally, you can place a copy of the 6,000 testing images in the `porn_val` folder.


## Downloading our trained model
We have released our certified robust trained model, adversarial trained model, and non-robust trained model in this [Download link.](https://drive.google.com/drive/folders/1b7RbO-uDdlvsxgsxE4H-tjrdGVx7pVRu?usp=sharing)
Please download this folder named `saved_models` and replace it with the existing `./train_verify/saved_models`.

## Environment functional test

To Qichang: Please write a one epoch training script for mnist, remember to rename the saved folder as test. Then describe the expected output in log.


## Environment functional test
To test whether the environment has been set up correctly, please run `./train_verify/test.sh` and `./attack/test.sh`. The expected output for `./train_verify/test.sh` should be like `./train_verify/test/train_log.txt`.


## Setting up existing PHash models
The code for hash generation with existing PHash models is under `generate_phash` folder. Please first navigate to this folder via:

```bash
cd generate_phash
```

### PyPhotoDNA setup
We follow the [repo](https://github.com/jankais3r/pyPhotoDNA) to set up the PhotoDNA model. To set up this model, you will have to
1) Run `install.bat` if you are on Windows, or `install.sh` if you are on a Mac or Linux. Then you should have a file with suffix `.dll` which is the PhotoDNA.
2) (You can skip this step at this time) Once the setup is complete, you can run `WINEDEBUG=-all wine64 python-3.9.12-embed-amd64/python.exe get_photodna_hash.py` to generate hashes.

### PDQ setup
Please follow the [repo](https://github.com/faustomorales/pdqhash-python) to set up the PDQ Hash Model. Specifically, you will need to first run `pip install pdqhash` and then execute `python get_pdq_hash.py`.

### NeuralHash setup
Please follow the guide provided by this [repo](https://github.com/ml-research/Learning-to-Break-Deep-Perceptual-Hashing), utilizing [AppleNeuralHash2ONNX](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX) to extract the NeuralHash model. For copyright reasons, we are unable to provide any NeuralHash files or models. Hash generation is under the `attack` folder.

To extract the NeuralHash model from a recent macOS or iOS build, please follow the conversion guide provided by AppleNeuralHash2ONNX. We will not provide any NeuralHash files or models, neither this repo nor by request. After extracting the onnx model, put the file model.onnx into `attack/models/`. Further, put the extracted Core ML file neuralhash_128x96_seed1.dat into `attack/models/coreml_model/`.

To convert the onnx model into PyTorch, run the following command after creating folder models and putting model.onnx into it. The converted files will be stored at `attack/models/model.pth`:

```bash
python utils/onnx2pytorch.py
```


# Experiment

## Functional evaluation
After downloading the datasets and models, we can do functional evaluation (RQ1 in our paper) using our trained model. We perform benign non-adversarial transformations on images such as rotation and brightness alterations and verify the ROC-AUC score. To perform such verification, you will first need to execute `attack/benign0_func_check.py` to generate hashes for transformed images and then execute `attack/benign0_func_AUC.py` to derive the ROC-AUC score. 

After downloading the datasets and models, you can perform functional evaluation (RQ1 in our paper) using the trained model. The goal of this evaluation is to assess how the model performs under benign, non-adversarial transformations. These transformations include operations like rotation and brightness alterations, which help verify the model's performance in terms of the ROC-AUC score.

### Steps for Functional Verification

To perform the verification, follow the steps below. The process involves two key scripts: 
1. `benign0_func_check.py` – Generates hashes for transformed images.
2. `benign0_func_AUC.py` – Calculates the ROC-AUC score based on the generated hashes.

### 1. Neuralhash

To evaluate **Neuralhash** with the **COCO validation dataset**, run the following commands:

```bash
python benign0_func_check.py --dataset='coco_val' --target='neuralhash_nn'
python benign0_func_AUC.py --dataset='coco_val' --target='neuralhash_nn'
```
### 2. PDQ
For evaluating PDQ on the COCO validation dataset, use these commands:
```bash
python benign0_func_check.py --dataset='coco_val' --target='pdq'
python benign0_func_AUC.py --dataset='coco_val' --target='pdq'
```

### 3. PhotoDNA
For PhotoDNA, the process involves three steps:

#### Step 1: Generate hashes for transformed images
Run the following command to generate and save the transformed images for PhotoDNA evaluation:

```bash
python benign0_func_check.py --dataset='coco_val' --target='photodna'
```
This will save the images, and you will need to use the generate_phash script to get the corresponding .csv file.

#### Step 2: Generate PhotoDNA hashes using Wine (for Windows compatibility)
Change to the generate_phash directory and run the following command to generate the PhotoDNA hashes:
```bash
cd ../generate_phash
WINEDEBUG=-all wine64 python-3.9.12-embed-amd64/python.exe get_photodna_hash.py --root="../attack/func_logs/coco_val_photodna/pepperSalt" --benign_func_test
```

#### Step 3: Calculate the ROC-AUC score for PhotoDNA
Finally, return to the attack directory and run the following command to calculate the ROC-AUC score for PhotoDNA:

```bash
cd ../attack
python benign0_func_AUC.py --dataset='coco_val' --target='photodna'
```
This will generate the ROC-AUC score for the PhotoDNA evaluation.

## Training and verification

Instead of using our models, you can also train and verify from scratch.

### Normal and robust training
Both normal and robust training is conducted by executing `train_verify/train.py`. Different datasets require different configuration files, and normal versus robust training also each have their own specific configuration files. We provide the training script in `train_verify/train.sh`, which includes command lines for both **Robust Training** and **Normal Training** using the COCO and MNIST dataset. 

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
Robust verification involves verifying the CNER(Certified No Evasion Rate) and CNCR(Certified No Collision Rate) of the trained model. To derive the CNER, we run the `train_verify/verify_evasion.py`, specifying the perturbation epsilon and model path of the assessed model in the parameters. 

An example of verifying the CNER for a perturbation epsilon of 0.0312 for a certified robust trained on the COCO dataset could be like this:
```bash
python verify_evasion.py --data=coco --epsilon=0.0312 --model='saved_models/coco_photodna_ep8/ckpt_best.pth'
```

And on a PHash model trained using a non-robust method, the command line may be as follows:
```bash
python verify_evasion.py --data=coco --epsilon=0.0312 --model='saved_models/coco_photodna_ep0/ckpt_best.pth'
```

To derive the CNCR, we run `verify_preimage.py` under `./train_verify`. An example of verifying the CNCR for a model trained on the MNIST dataset could be like this:
```bash
python verify_preimage.py
```

You may find more scripts for model verification in `train_verify/verify.sh`.


### Empirical attacks
#### Empirical collision attacks
To perform empirical collision attacks, first derive the hash values of a dataset of choice using `utils/compute_dataset_hashes.py` (only need once), and then run `attack/adv1_collision_attack.py` to perform empirical collision attacks. 

Please refer to the scripts provided by `attack/adv1.sh` for more detailed command lines.

#### Empirical evasion attacks
Run `attack/adv2_evasion_attack.py` to perform empirical evasion attacks.

Again, please refer to the scripts provided by `attack/adv2.sh` for more detailed examples.

### Real-world evaluations on NSFW data
In our work, we adopt the NSFW dataset together with the CelabA dataset to assess our model's ability in NSFW detection. Please follow the data preparation section for both datasets.

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