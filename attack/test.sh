# Functionality test with the COCO dataset and PDQ
python benign0_func_check.py --dataset='coco_val' --target='pdq' # Hash generation for transformed images
python benign0_func_AUC.py --dataset='coco_val' --target='pdq' # Calculating the ROC-AUC score