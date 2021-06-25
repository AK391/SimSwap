import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_mutil import Face_detect_crop
from util.videoswap import video_swap
from util.add_watermark import watermark_image
import gradio as gr
import torchtext

torchtext.utils.download_from_url("https://drive.google.com/uc?id=1TLNdIufzwesDbyr_nVTR7Zrx9oRHLM_N", root="./arcface_model/arcface_checkpoint.tar")
torchtext.utils.download_from_url("https://drive.google.com/uc?id=1PXkRiBUYbu1xWpQyDEJvGKeqqUFthJcI", root="./checkpoints.zip")
torchtext.utils.extract_archive("./checkpoints.zip", "./checkpoints")
torchtext.utils.download_from_url("https://drive.google.com/uc?id=1Ex4vpSB8E6juGG3kPMIhLRzx175rBdz", root="./antelope.zip")
torchtext.utils.extract_archive("./antelope.zip", "./insightface_func/models/")

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

def inference(image, video):
    opt = TestOptions()
    opt.initialize()
    opt.parser.add_argument('-f') ## dummy arg to avoid bug
    opt = opt.parse()
    opt.pic_a_path = image.name ## or replace it with image from your own google drive
    opt.video_path = video ## or replace it with video from your own google drive
    opt.output_path = './output/demo.mp4'
    opt.temp_path = './tmp'
    opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
    opt.isTrain = False

    crop_size = 224

    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()


    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))

    pic_a = opt.pic_a_path
    # img_a = Image.open(pic_a).convert('RGB')
    img_a_whole = cv2.imread(pic_a)
    img_a_align_crop, _ = app.get(img_a_whole,crop_size)
    img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
    img_a = transformer_Arcface(img_a_align_crop_pil)
    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

    # convert numpy to tensor
    img_id = img_id.cuda()

    #create latent id
    img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
    latend_id = model.netArc(img_id_downsample)
    latend_id = latend_id.detach().to('cpu')
    latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)
    latend_id = latend_id.to('cuda')

    video_swap(opt.video_path, latend_id, model, app, opt.output_path,temp_results_dir=opt.temp_path)
    return './output/demo.mp4'

title = "Anime2Sketch"
description = "demo for Anime2Sketch. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.05703'>Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis</a> | <a href='https://github.com/Mukosame/Anime2Sketch'>Github Repo</a></p>"

gr.Interface(
    inference, 
    [gr.inputs.Image(type="file", label="Input"),
     gr.inputs.Video(label="Input Video")], 
    gr.outputs.Video(label="Output Video"),
    title=title,
    description=description,
    article=article).launch(debug=True)






