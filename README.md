# MPG-Align: A Multimodal Cross-attention Alignment for Medical Phrase Grounding Task on Chest X-Ray Radiological Images.

![Image Text]()

## Training and Testing
1. Download the base model [DETR]() and [TransVG]() to folder ```pretrained``` 
2. Download [MS-CXR Dataset]() to folder ```ln_data```
3. Run script ```script_ours_MS_CXR.sh``` to train the model.

## Testing 
1. Download [trained checkpoint]() to folder ```released_checkpoint``` 
2. Or include your trained checkpoint to folder ```released_checkpoint```
3. Run ```eval.py``` for evaluation.