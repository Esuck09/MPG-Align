import numpy as np
import os
from PIL import Image, ImageDraw
import numpy as np

"""
This code is primarily based on the MedRPG implementation from:

Chen, Zhihao et al. "Medical Phrase Grounding with Region-Phrase Context Contrastive Alignment."
MICCAI, 2023. https://arxiv.org/abs/2307.11767

Original code: https://github.com/openmedlab/MedRPG

Please refer to the original authors for core algorithmic contributions.
"""

def visualBBox(image_path, pred_box, bbox, output_dir):
    real_im = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(real_im)
    color_gt = (255, 0, 0) # 红色代表gt
    color_method1 = (0, 255, 0) # 绿色代表预测
    if bbox is not None:
        draw.rectangle(bbox, outline=color_gt, width=2)
    draw.rectangle(pred_box, outline=color_method1, width=2)
    save_path = os.path.basename(image_path)
    real_im.save('{}/{}.png'.format(output_dir, save_path))
    del draw



if __name__ == '__main__':

    pass
