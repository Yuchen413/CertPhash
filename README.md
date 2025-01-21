# CERTPHASH: Towards Certified Perceptual Hashing via Robust Training
**Abstract:**
Perceptual hashing (PHash) systems—e.g., Apple’s NeuralHash, Microsoft’s PhotoDNA, and Facebook’s PDQ—are widely employed to screen illicit content. Such systems generate hashes of image files and match them against a database of known hashes linked to illicit content for filtering. One important drawback of PHash systems is that they are vulnerable to adversarial perturbation attacks leading to hash evasion or collision. It is desirable to bring provable guarantees to PHash systems to certify their robustness under evasion or collision attacks. However, to the best of our knowledge, there are no existing certified PHash systems, and more importantly, the training of certified PHash systems is challenging because of the unique definition of model utility and the existence of both evasion and collision attacks. 

In this paper, we propose CERTPHASH, the first certified PHash system with robust training. CERTPHASH includes three different optimization terms, anti-evasion, anti-collision, and functionality. The anti-evasion term establishes an upper bound on the hash deviation caused by input perturbations, the anti-collision term sets a lower bound on the distance between a perturbed hash and those from other inputs, and the functionality term ensures that the system remains reliable and effective throughout robust training. Our results demonstrate that CERTPHASH not only achieves non-vacuous certification for both evasion and collision with provable guarantees but is also robust against empirical attacks. Furthermore, CERTPHASH demonstrates strong performance in real-world illicit content detection tasks.


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

## Preparing datasets
For robust training and evaluation, we use the following four datasets:
- **COCO**: [Download Link](https://github.com/anishathalye/ribosome/releases/download/v1.0.0/coco100x100.tar.gz).
- **MNIST**: A folder named "mnist" includes images and hashes. [Download Link]()
- **CelabA**: Random Selection of 50,000 samples are used for training and 6,000 samples are used for verification. The dataset download link is TODO. 
- **NSFW-56K**: The NSFW-56K dataset includes 56,000 images depicting explicit and unsafe content, all resized to 64x64 pixels. This dataset is used together with CelabA to assess CERTPHASH’s ability in real-world NSFW content detection. Due to ethical concerns, we do not attach a download link for this dataset.

After downloading the necessary datasets, you will need to place the datasets under `train_verify/data` according to the file hierarchy specified by `train_verify/data/put_data_here_follow_this.txt`.

## Setting up existing PHash models
The code for hash generation with existing PHash models is under `generate_phash` folder. 

### PyPhotoDNA setup
We follow the [repo](https://github.com/jankais3r/pyPhotoDNA) to set up the PhotoDNA model. To set up this model, you will have to 
1)	Clone the repo.
2)	Run `install.bat` if you are on Windows, or `install.sh` if you are on a Mac or Linux.
3)	Once the setup is complete, run `WINEDEBUG=-all wine64 python-3.9.12-embed-amd64/python.exe get_photodna_hash.py` to generate hashes.

### PDQ setup
Please follow the [repo](https://github.com/faustomorales/pdqhash-python) to set up the PDQ Hash Model. Specifically, you will need to first run `pip install pdqhash` and then execute `python get_pdq_hash.py`.

### NeuralHash setup
Please follow the guide provided by this [repo](https://github.com/ml-research/Learning-to-Break-Deep-Perceptual-Hashing), utilizing [AppleNeuralHash2ONNX](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX) to extract the NeuralHash model. For copyright reasons, we are unable to provide any NeuralHash files or models. Hash generation is under the ./attack folder.

<!-- To extract the NeuralHash model from a recent macOS or iOS build, please follow the conversion guide provided by AppleNeuralHash2ONNX. We will not provide any NeuralHash files or models, neither this repo nor by request. After extracting the onnx model, put the file model.onnx into /models/. Further, put the extracted Core ML file neuralhash_128x96_seed1.dat into /models/coreml_model.

To convert the onnx model into PyTorch, run the following command after creating folder models and putting model.onnx into it. The converted files will be stored at models/model.pth:

python utils/onnx2pytorch.py -->

## Training and verification
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
Robust verification involves verifying the CNER(Certified No Evasion Rate) and CNCR(Certified No Collision Rate) of the trained model. To derive the CNER, we run the `train_verify/verify_evasion.py`, specifying the perturbation epsilon and model path of the assessed model in the parameters. An example of verifying the CNER for a perturbation epsilon of 0.0312 for a non-robust PHash model trained on the COCO dataset could be like this:
```bash
python verify_evasion.py --data=coco --epsilon=0.0312 --model='saved_models/base_adv/coco-pdq-ep8-pdg.pt'
```

To derive the CNCR, we run verify_preimage.py under ./train_verify. An example of verifying the CNCR for a model trained on the MNIST dataset could be like this:
```bash
python verify_preimage.py --data=mnist --verify_epsilon=0.0078 --model='mnist-pdq-ep8-pdg.pt'
```
You may find more scripts for model verification in `train_verify/verify.sh`.


### Performing functional verification
In functional verification, we perform benign non-adversarial transformations on images such as rotation and brightness alterations and verify the ROC-AUC score. To perform such verification, you will first need to execute `attack/benign0_func_check.py` to generate hashes for transformed images and then execute `attack/benign0_func_AUC.py` to derive the ROC-AUC score. A sample script for this two-step operation is enclosed in `attack/benign0.sh`.

### Empirical attacks
#### Empirical collision attacks
To perform empirical collision attacks, first derive the hash values of a dataset of choice using `utils/compute_dataset_hashes.py`, and then run `attack/adv1_collision_attack.py` to perform empirical collision attacks. 

For example, you could run the following command line to evaluate a model trained on the COCO dataset.
```bash
python adv1_collision_attack.py --data=coco --epsilon=0.1 --model=../train_verify/saved_models/base_adv/coco-pdq-ep8-pdg.pt --output_folder=collision_test100_coco --source=/home/yuchen/code/verified_phash/train_verify/data/coco100x100_val --sample_limit=100 --threads=10 --learning_rate=5
```

Please refer to the scripts provided by `attack/adv1.sh` for more detailed command lines.

#### Empirical evasion attacks
To perform empirical evasion attacks, first derive the hash values of a dataset of choice using `utils/compute_dataset_hashes.py` if hashes haven't been generated yet, and then run `attack/adv2_evasion_attack.py` to perform empirical evasion attacks.

Below is an example of conducting empirical evasion attacks on a model trained on the COCO dataset.
```bash
python adv2_evasion_attack.py --data=coco --epsilon=0.034 --model=../train_verify/saved_models/coco_photodna_ep0/last_epoch_state_dict.pth --output_folder=evasion_test100_coco --source=/home/yuchen/code/verified_phash/train_verify/data/coco100x100_val --sample_limit=100 --threads=10 --learning_rate=1
```
Again, please refer to the scripts provided by `attack/adv2.sh` for more detailed examples.

### Real-world evaluations on NSFW data
In our work, we adopt the NSFW dataset together with the CelabA dataset to assess our model's ability in NSFW detection. The dataset will be made available exclusively for research purposes, and both the dataset and the model will require a password for access.

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