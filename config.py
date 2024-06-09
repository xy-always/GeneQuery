

#commented out variables are handled by argparse in main.py
debug = True
# batch_size = 128
# num_workers = 0
# lr = 0.005
# weight_decay = 1e-4
patience = 2
factor = 0.5
# epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_name = 'resnet50'
# image_embedding = 1000
# spot_embedding = 3559 #number of shared hvgs (change for each dataset)

pretrained = True
trainable = True 
temperature = 1

# image size
size = 224

# for projection head; used for both image and text encoders
# num_projection_layers = 1
# projection_dim = 256
# dropout = 0.1


#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',
