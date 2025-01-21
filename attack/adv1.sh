###First need to run this, this just needs one time
#python utils/compute_dataset_hashes.py --source=../train_verify/data/porn_val

##Then run this
#python adv1_collision_attack.py --data=coco --epsilon=0.1 --model=../train_verify/saved_models/base_adv/coco-pdq-ep8-pdg.pt --output_folder=collision_test100_coco --source=/home/yuchen/code/verified_phash/train_verify/data/coco100x100_val --threads=10 --learning_rate=5

#under l2 norm
#python adv1_collision_attack.py --data=mnist --epsilon=0.72 --model=../train_verify/saved_models/base_adv/mnist-pdq-ep8-pdg.pt --output_folder=collision_test100_mnist --source=/home/yuchen/code/verified_phash/train_verify/data/mnist/testing  --threads=10 --learning_rate=5e-4
#python adv1_collision_attack.py --data=mnist --epsilon=0.72 --model=../train_verify/saved_models/mnist_pdq_ep2/ckpt_best.pth --output_folder=collision_test100_mnist --source=/home/yuchen/code/verified_phash/train_verify/data/mnist/testing  --threads=10 --learning_rate=5e-4



