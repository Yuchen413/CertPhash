### This is for CertPHash
# python benign0_func_check.py --dataset='coco_val' --target='photodna_nn_cert_ep1' --model='../train_verify/saved_models/coco_photodna_ep1/ckpt_best.pth'
#python benign0_func_AUC.py --dataset='coco_val' --target='photodna_nn_cert_ep1'


## This is for Non-robust Training
#python benign0_func_check.py --dataset='coco_val' --target='photodna_nn_ep0' --model='../train_verify/saved_models/coco_photodna_ep0/ckpt_best.pth'
#python benign0_func_AUC.py --dataset='coco_val' --target='photodna_nn_ep0'


## This is for adversarial Training
#python benign0_func_check.py --dataset='coco_val' --target='photodna_nn_adv' --model='../train_verify/saved_models/base_adv/coco-pdq-ep8-photodna.pt'
#python benign0_func_AUC.py --dataset='coco_val' --target='photodna_nn_adv'



### This is for Neuralhash
#python benign0_func_check.py --dataset='coco_val' --target='neuralhash_nn'
#python benign0_func_AUC.py --dataset='coco_val' --target='neuralhash_nn'



## This is for pdq
#python benign0_func_check.py --dataset='coco_val' --target='pdq'
#python benign0_func_AUC.py --dataset='coco_val' --target='pdq'



####This is for photodna
##Step 1:
# python benign0_func_check.py --dataset='coco_val' --target='photodna' ##This is actually save the images, then needs to use the generate_phash to get the .csv
#

##Step 2:
#cd ../generate_phash
#main_root="../attack/func_logs/coco_val_photodna"
#for subroot in "$main_root"/*; do
#    if [ -d "$subroot" ]; then
#        echo "Processing folder: $subroot"
#        WINEDEBUG=-all wine64 python-3.9.12-embed-amd64/python.exe get_photodna_hash.py --root="$subroot" --benign_func_test
#        if [ $? -eq 0 ]; then
#            echo "Completed successfully: $subroot"
#        else
#            echo "Error processing: $subroot" >&2
#        fi
#    fi
#done

###Step 3:
#cd ../attack
#python benign0_func_AUC.py --dataset='coco_val' --target='photodna'






