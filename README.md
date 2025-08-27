# CKCR
<img width="1623" height="931" alt="image" src="https://github.com/user-attachments/assets/19fa3088-4d9e-420b-b2a0-c362f3556334" />

## Installation and Requirements

Please note that our environment requirements are different from LLaVA's environment requirements. We strongly recommend you create the environment from scratch as follows.

1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/HVCL/HVCL.git
cd HVCL
```

2. Create a conda environment, activate it and install Packages
```Shell
conda create -n CKCR python=3.8
conda activate CKCR
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Install additional packages
```Shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.28.1
conda install -c pytorch faiss-gpu -y
pip install setuptools==59.5.0
pip install wandb pytorch-lightning==2.0.4 jsonnetbin easydict pandas scipy opencv-python fuzzywuzzy scikit-image matplotlib timm scikit-learn sentencepiece tensorboard datasets
pip install ujson evaluate GPUtil easydict peft==0.4.0
pip install bitarray spacy ujson gitpython ninja absl-py openai sacrebleu
cd third_party/ColBERT
pip install -e .
```

### COCO images
`data/ok-vqa/train2014`: [Train images](http://images.cocodataset.org/zips/train2014.zip)

`data/ok-vqa/val2014`: [Test images](http://images.cocodataset.org/zips/val2014.zip)

### OKVQA Dataset
`data/ok-vqa/mscoco_train2014_annotations.json`: [Training annotations](https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip)

`data/ok-vqa/mscoco_val2014_annotations.json`: [Testing annotations](https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip)

`data/ok-vqa/OpenEnded_mscoco_train2014_questions.json`: [Training questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip)

`data/ok-vqa/OpenEnded_mscoco_val2014_questions.json`: [Testing questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip)

### Google Search Corpus
[Official download link](https://drive.google.com/drive/folders/15uWx33RY5UmR_ZmLO6Ve1wyzbXsLxV6o?usp=sharing)

Data can be saved to `data/ok-vqa/pre-extracted_features/passages/okvqa_full_corpus.csv`.

Other codes will come soon！
## Contact
If you have any questions, feel free to either initiate an *Issue* .




## ❤️ Community efforts
* Our codebase is built upon the [RK-VQA](https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering) project. Great work!
