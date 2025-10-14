#from huggingface_hub import model_info; 
#print(model_info('gpt2'))
#requests.get("https://huggingface.co", timeout=5)
import transformers
print(transformers.__version__)
#update transformer via: pip install -U transformers --upgrade

import os
from getsysteminfo import getDeviceType
is_windows, is_hpc, is_hpc1gpu, is_hpc2gpu = getDeviceType()
if is_windows:
    #hfhome_dir=os.path.join('D:\\','Cache','huggingface')
    hfhome_dir=os.path.join('D:',os.sep, 'Cache','huggingface')
    os.makedirs(hfhome_dir, exist_ok=True)
    os.environ['HF_HOME'] = hfhome_dir
elif is_hpc:
    hfhome_dir="/data/cmpe249-fa23/Huggingfacecache"
    os.makedirs(hfhome_dir, exist_ok=True)
    #os.environ['TRANSFORMERS_CACHE'] = hfhome_dir
    os.environ['HF_HOME'] = hfhome_dir
    #os.environ['HF_HUB_CACHE'] = os.path.join(hfhome_dir, 'hub')
    os.environ['HF_DATASETS_CACHE'] = hfhome_dir
    #HF_HUB_OFFLINE=1
else:#other linux
    # hfhome_dir=os.path.join('./data/', 'huggingface')
    # os.makedirs(hfhome_dir, exist_ok=True)
    # os.environ['HF_HOME'] = hfhome_dir
    print("Using default HF Home")


from transformers import pipeline
classifier = pipeline("sentiment-analysis")
results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

from transformers import AutoModel, AutoImageProcessor
model = AutoModel.from_pretrained("google/vit-base-patch16-224")
image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224",
        trust_remote_code=True,
    )

from datasets import load_dataset
imdb_dataset = load_dataset("imdb")
print(imdb_dataset)

# eli5 = load_dataset("eli5")
# print(eli5)

import evaluate
metric = evaluate.load("sacrebleu") #pip install sacrebleu
metric = evaluate.load("accuracy") #save to /data/cmpe249-fa23/Huggingfacecache/metrics
metric = evaluate.load("squad")

# from huggingface_hub import delete_cache
# delete_cache()

