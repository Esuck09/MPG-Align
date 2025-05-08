# MPG-Align: A Multimodal Cross-attention Alignment for Medical Phrase Grounding Task on Chest X-Ray Radiological Images.

![Image Text](https://github.com/Esuck09/MPG-Align/blob/main/figures/models.png)

## Paper Link 
Paper is available by accessing the [PDF](https://drive.google.com/file/d/1GRO7CjuKpcsi9FDYucNDaCMZbhgXLq-A/view?usp=drive_link)

## Training
1. Download the base model [DETR](https://drive.google.com/file/d/1te77Wklb3_ayNJJvQmV1C0tnhmpheT9C/view?usp=drive_link) and [TransVG](https://drive.google.com/file/d/1CNjgvAFvXnmyTsxGohENbP2aTqgq_l1r/view?usp=drive_link) to folder ```pretrained``` 
2. Download [MS-CXR Dataset](https://drive.google.com/file/d/1VO2gnfw18MCiWUdVKjMvXXIkosf0flHh/view?usp=drive_link) to folder ```ln_data```
3. Run script ```script_ours_MS_CXR.sh``` to train the model.

## Testing 
1. Download [trained checkpoint](https://drive.google.com/file/d/1a156iF4_3j6PXwYdQh1fKSA9wJETfmKM/view?usp=drive_link) to folder ```released_checkpoint``` 
2. Or include your trained checkpoint to folder ```released_checkpoint```
3. Run ```eval.py``` for evaluation.

## Having issue accessing the files?
Simple! Send an email to ```qikai.work@gmail.com```