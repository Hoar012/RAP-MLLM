# from clip_model import model,processor
import faiss
import requests, json
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os
import torch
import numpy as np
from transformers import CLIPTextModel, CLIPVisionModel, CLIPModel, CLIPProcessor

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

clip_model = 'openai/clip-vit-large-patch14-336'
class ClipRetriever():
    def __init__(self, data_dir, index_path, embed_dim = 768, create_index = False, batch_size = 6, device = "cuda"):
        self.device = device
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.text_model= CLIPTextModel.from_pretrained(clip_model).to(self.device)
        self.feature_extractor = CLIPProcessor.from_pretrained(clip_model)
        self.index = faiss.IndexFlatL2(embed_dim)

        if create_index:
            print(f"Creating database from {data_dir}")

            file_paths = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        file_path = os.path.join(root, file)
                        file_paths.append(file_path)

            self.id2filename = {str(idx): x for idx,x in enumerate(file_paths)}
            with open(f'{data_dir}/id2filename.json', 'w') as json_file:
                json.dump(self.id2filename, json_file)

            for file_path in tqdm(file_paths, total = len(file_paths)):
                try:
                    image = Image.open(file_path)
                    inputs = self.feature_extractor(images=image, return_tensors="pt", padding = True)
                    image_features = self.clip_model.get_image_features(inputs["pixel_values"].to(self.device))
                    image_features = image_features / image_features.norm(p = 2, dim = -1, keepdim = True)  # normalize
                    image_features = image_features.detach().cpu().numpy()
                    self.index.add(image_features)
                    image.close()
                except Exception as e:
                    print(e)
                    print(file_path)

            faiss.write_index(self.index, f"{data_dir}/image.faiss")
        else:
            print(f"Loading database from {index_path}")
            with open(f'{data_dir}/id2filename.json', 'r') as json_file:
                self.id2filename = json.load(json_file)
            self.index = faiss.read_index(index_path)

    def text_search(self, text, k = 1):
        inputs = self.feature_extractor(text=text, images=None, return_tensors="pt", padding=True).to(self.device)
        text_features = self.clip_model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize
        text_features = text_features.detach().cpu().numpy()
        dists, I = self.index.search(text_features, k) #retrieval

        filenames = [[self.id2filename[str(j)] for j in i] for i in I]
        print(dists, filenames)
        return dists, filenames

    def image_search(self, image, k = 1):
        inputs = self.feature_extractor(images = image, return_tensors="pt")
        image_features = []
        dists, indexes = [], []
        for feature in torch.split(inputs['pixel_values'], self.batch_size, 0):
            image_features = self.clip_model.get_image_features(feature.to(self.device))
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
            image_features = image_features.detach().cpu().numpy()
            D, I = self.index.search(image_features, k) #retrieval

            dists += D.tolist()
            indexes += I.tolist()

        filenames = [[self.id2filename[str(j)] for j in i] for i in indexes]
        print(dists, filenames)
        return np.array(dists), filenames
    
    def retrieve(self, database, inp, queries, topK = 2):
        rag_images = dict()
        for concept in database["concept_dict"]:
            if concept in inp:
                rag_images[database["concept_dict"][concept]["image"]] = 0
        if len(queries) > 0:
            D, filenames = self.image_search(queries, k=2)
            ret_image_path = []
            for files in filenames:
                ret_image_path += files
            
            D = D.flatten()
            order = D.argsort()
            for i in order:
                if len(rag_images) >= topK:
                    break
                if ret_image_path[i] in rag_images:
                    continue
                
                rag_images[ret_image_path[i]] = D[i].tolist()
        
        extra_info = ""
        for i, ret_path in enumerate(rag_images):
            ret_path = ret_path.lstrip('./')
            tag = database["path_to_concept"][ret_path]
            name = database["concept_dict"][tag]["name"]
            info = database["concept_dict"][tag]["info"]
            extra_info += f"{i+1}.<image>\n Name: <{name}>, Info: {info}\n"
        
        return extra_info, rag_images
    