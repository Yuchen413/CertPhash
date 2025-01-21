#COCO + PhotoDNA
#Robust Train
#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0039 --dir=saved_models/coco_photodna_ep1_revise_final --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160  --model='resnet_v5' --lr=5e-4
#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0078 --dir=saved_models/coco_photodna_ep2 --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160  --model='resnet_v5' --lr=5e-4
#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0156 --dir=saved_models/coco_photodna_ep4 --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160  --model='resnet_v5' --lr=5e-4
#python train.py --method=fast --config=config/coco.crown-ibp.json --eps=0.0312 --dir=saved_models/coco_photodna_ep8 --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160  --model='resnet_v5' --lr=5e-4

#Normal Train
#python train.py --config=config/coco.normal.json --dir=saved_models/coco_photodna_ep0 --lr-decay-milestones=15,18 --lr-decay-factor=0.2 --num-epochs=40  --model='resnet_v5' --lr=5e-4 --natural

#MNIST + PQD
#Robust Train
#python train.py --method=fast --config=config/mnist.crown-ibp.json --eps=0.0039 --dir=saved_models/mnist_pdq_ep1 --scheduler_opts=start=1,length=20 --lr-decay-milestones=50,60 --lr-decay-factor=0.2 --num-epochs=70  --model='resnet' --lr=5e-4
#python train.py --method=fast --config=config/mnist.crown-ibp.json --eps=0.0078 --dir=saved_models/mnist_pdq_ep2 --scheduler_opts=start=1,length=20 --lr-decay-milestones=50,60 --lr-decay-factor=0.2 --num-epochs=70  --model='resnet' --lr=5e-4
#python train.py --method=fast --config=config/mnist.crown-ibp.json --eps=0.0627 --dir=saved_models/mnist_pdq_ep16 --scheduler_opts=start=1,length=20 --lr-decay-milestones=50,60 --lr-decay-factor=0.2 --num-epochs=70  --model='resnet' --lr=5e-4

#Normal Train
#python train.py --config=config/mnist.normal.json --dir=saved_models/mnist_pdq_ep0_test --lr-decay-milestones=15,18 --lr-decay-factor=0.2 --num-epochs=20  --model='resnet' --lr=5e-4 --natural


#NSFW + PhotoDNA
#Robust
#python train.py --method=fast --config=config/nsfw.crown-ibp.json --eps=0.0039 --dir=saved_models/NSFW_photodna_ep1_test --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --lr-decay-factor=0.2 --num-epochs=160  --model='resnet_v5' --lr=5e-4

#NSFW + PhotoDNA
#Normal
#python train.py --method=fast --config=config/nsfw.normal.json --dir=saved_models/NSFW_photodna_ep0_test --lr-decay-milestones=15,18 --lr-decay-factor=0.2 --num-epochs=40  --model='resnet_v5' --lr=5e-4 --natural
