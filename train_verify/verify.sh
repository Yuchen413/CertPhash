##Example for coco dataset

python verify_evasion.py --data=coco --epsilon=0.0312 --model='coco_photodna_ep1/ckpt_best.pth' # certphash trained with eps=1/255
#python verify_evasion.py --data=coco --epsilon=0.0312 --model='coco_photodna_ep0/ckpt_best.pth' # means non-robust trained
#python verify_evasion.py --data=coco --epsilon=0.0312 --model='saved_models/base_adv/coco-pdq-ep8-pdg.pt' # for adversarial trained with eps=8/255


##Example for other datasets

#python verify_evasion.py --data=mnist --epsilon=1 --model='saved_models/mnist_pdq_ep8/last_epoch_state_dict.pth'
#python verify_evasion.py --data=nsfw --epsilon=0.0312 --model='saved_models/nsfw_photodna_ep1/ckpt_best.pth'