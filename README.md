# CERTPHASH: Towards Certified Perceptual Hashing via Robust Training
**Overview:**

CERTPHASH is the first certified perceptual hashing (PHash) system designed to provide provable robustness against both evasion and collision attacks. Using verifier tools, CERTPHASH is trained on a robust method that incorporates anti-evasion, anti-collision and functionality optimization terms, and is able to withstand adversarial perturbations while maintaining functional utility. Being the first certified PHash system with robust training, it is a powerful tool for detecting illicit content in real-world applications. 

<!-- Perceptual hashing (PHash) systems—e.g., Apple’s NeuralHash, Microsoft’s PhotoDNA, and Facebook’s PDQ—are widely employed to screen illicit content. Such systems generate hashes of image files and match them against a database of known hashes linked to illicit content for filtering. One important drawback of PHash systems is that they are vulnerable to adversarial perturbation attacks leading to hash evasion or collision. It is desirable to bring provable guarantees to PHash systems to certify their robustness under evasion or collision attacks. However, to the best of our knowledge, there are no existing certified PHash systems, and more importantly, the training of certified PHash systems is challenging because of the unique definition of model utility and the existence of both evasion and collision attacks. 

In this paper, we propose CERTPHASH, the first certified PHash system with robust training. CERTPHASH includes three different optimization terms, anti-evasion, anti-collision, and functionality. The anti-evasion term establishes an upper bound on the hash deviation caused by input perturbations, the anti-collision term sets a lower bound on the distance between a perturbed hash and those from other inputs, and the functionality term ensures that the system remains reliable and effective throughout robust training. Our results demonstrate that CERTPHASH not only achieves non-vacuous certification for both evasion and collision with provable guarantees but is also robust against empirical attacks. Furthermore, CERTPHASH demonstrates strong performance in real-world illicit content detection tasks. -->


## Environment Setup
Follow the steps below to set up the environment:

1. **Install basic dependencies**:
   - Install the required dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```

2. **Install auto_LiRPA dependencies**:
   - Clone the [auto_LiRPA repository](https://github.com/Verified-Intelligence/auto_LiRPA) and install its dependencies by running:
     ```bash
     git clone git@github.com:Verified-Intelligence/auto_LiRPA.git
     cd auto_LiRPA
     python setup.py install
     ```

This will install all the necessary packages for training and evaluation. 

## Preparing Datasets
We provide the [download link](https://drive.google.com/file/d/11LBjSRM-tqhJOKcRbxO8_8ZEabPdepUD/view?usp=sharing) for the datasets (images and phashes) we used. 

After downloading the above link, you will need to unzip and named the folder as `data`, then replace the current folder `train_verify/data` with the downloaded one.

The file hierarchy is specified by `train_verify/data/put_data_here_follow_this.txt`.

Note that COCO need additional step as decribed below:

For robust training and evaluation, we use the following four datasets:
- **COCO**: ALSO need to download images from this [Link](https://github.com/anishathalye/ribosome/releases/download/v1.0.0/coco100x100.tar.gz) and unzip (named coco100x100), and then put the folder under the `train_verify/data`.
- **MNIST**: A folder named "mnist" includes images and hashes.
- **CelabA**: Random Selection of 50,000 samples are used for training and 6,000 samples are used for verification.
- **NSFW-56K**: The NSFW-56K dataset includes 56,000 images depicting explicit and unsafe content, all resized to 64x64 pixels. This dataset is used together with CelabA to assess CertPhash's ability in real-world NSFW content detection. Due to ethical concerns, we do not attach a download link for this dataset. Please follow this [repo](https://github.com/alex000kim/nsfw_data_scraper) to scape the images labeled as `porn`, and randomly selected 56,000 images as the NSFW dataset. And named the folder with 56,000 images as `porn_resized` and put it under `train_verify/data`. Then random split it into 50,000 and 6,000 for training and testing (See the `nsfw_train.csv` and `nsfw_val.csv` downloaded from the first link in this section as a reference for format), you can also have a copy of 6000 testing images into folder `porn_val`


## Downloading our trained model
We have released our certified robust trained model, adversarial trained model, and non-robust trained model in this [Download link.](https://drive.google.com/drive/folders/1b7RbO-uDdlvsxgsxE4H-tjrdGVx7pVRu?usp=sharing)
Please download this folder named `saved_models` and replace it with the existing `train_verify/saved_models`.


## Setting up existing PHash models
The code for hash generation with existing PHash models is under `generate_phash` folder. Please first navigate to this folder via:

```bash
cd generate_phash
```

### PyPhotoDNA setup
We follow the [repo](https://github.com/jankais3r/pyPhotoDNA) to set up the PhotoDNA model. To set up this model, you will have to
1) Run `install.bat` if you are on Windows, or `install.sh` if you are on a Mac or Linux. Then you should get a file with suffix `.dll`
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

## Performing functional verification
After downloading the datasets and models, we can do functional evaluation (RQ1 in our paper) using our trained model. We perform benign non-adversarial transformations on images such as rotation and brightness alterations and verify the ROC-AUC score. To perform such verification, you will first need to execute `attack/benign0_func_check.py` to generate hashes for transformed images and then execute `attack/benign0_func_AUC.py` to derive the ROC-AUC score. 

A sample script for this two-step operation is enclosed in `attack/benign0.sh`.

To Qichang: please test the benign0.sh one by one for each model, and copy the script seperately here, and just make sure they can run is enough.

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

And on non-robust PHash model trained:
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