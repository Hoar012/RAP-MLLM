import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

import requests, json
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from detector import Detector
from retriever import ClipRetriever

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "phi" in model_name.lower():
        conv_mode = "phi3_instruct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"    
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    if args.retrieval:    
        detector = Detector()
        with open(f"{args.database}/database.json", "r") as f:
            database = json.load(f)
        
        # Set interested classes
        all_category = []
        for concept in database["concept_dict"]:
            cat = database["concept_dict"][concept]["category"]
            if cat not in all_category:
                all_category.append(cat)
        detector.model.set_classes(all_category)
        
        if args.index_path is None:
            retriever = ClipRetriever(data_dir = args.database, index_path = args.index_path, create_index = True)
        else:
            retriever = ClipRetriever(data_dir = args.database, index_path = args.index_path, create_index = False)
            
    image = load_image(args.image_file)
    image_sizes = [image.size]
    images = [image]

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if args.retrieval:
                crops = detector.detect_and_crop(image)
                extra_info, rag_images = retriever.retrieve(database, inp, queries = crops, topK = args.topK)
                
                for i, ret_path in enumerate(rag_images):
                    img = load_image(ret_path)
                    image_sizes.append(img.size)
                    images.append(img)
                
                inp = DEFAULT_IMAGE_TOKEN + f"\n[{extra_info}]" + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
                
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        
        # Similar operation in model_worker.py
        image_tensor = process_images(images, image_processor, model.config)

        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        prompt = conv.get_prompt()
        # print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--database", type=str, default=None)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--topK", type=int, default=2)

    args = parser.parse_args()
    main(args)
