## Data

### Download Links

<div class="two-columns">
    <div class="column">
        <table>
            <tr><th colspan="2" class="header">Full Dataset</th></tr>
            <tr>
                <td><a href="https://huggingface.co/datasets/Hoar012/RAP-260K/blob/main/rap_train_260k.json">rap_train_260k.json</a></td>
                <td><a href="https://huggingface.co/datasets/Hoar012/RAP-260K/blob/main/rap_train_210k.json">rap_train_210k.json</a></td>
            </tr>
            <tr><th colspan="2" class="header">Split</th></tr>
            <tr>
                <td><a href="https://huggingface.co/datasets/Hoar012/RAP-260K/blob/main/split/rap_grounding_100k.json">rap_grounding_100k.json</a></td>
                <td><a href="https://huggingface.co/datasets/Hoar012/RAP-260K/blob/main/split/rap_caption_30k.json">rap_caption_30k.json</a></td>
            </tr>
            <tr>
                <td><a href="https://huggingface.co/datasets/Hoar012/RAP-260K/blob/main/split/rap_recognition_40k.json">rap_recognition_40k.json</a></td>
                <td><a href="https://huggingface.co/datasets/Hoar012/RAP-260K/blob/main/split/rap_description_7k.json">rap_description_7k.json</a></td>
            </tr>
            <tr>
                <td><a href="https://huggingface.co/datasets/Hoar012/RAP-260K/blob/main/split/rap_qa_16k.json">rap_qa_16k.json</a></td>
                <td><a href="https://huggingface.co/datasets/Hoar012/RAP-260K/blob/main/split/llava_instruct_67k.json">llava_instruct_67k.json</a></td>
            </tr>
        </table>
    </div>
</div>


[`rap_train_260k.json`](https://huggingface.co/datasets/Hoar012/RAP-260K/blob/main/rap_train_260k.json) is the full dataset used for training RAP-Phi3-V; [`rap_train_210k.json`](https://huggingface.co/datasets/Hoar012/RAP-260K/blob/main/rap_train_210k.json) is a subset used for training RAP-LLaVA.

### Data Format
Each sample in the dataset contains the following keys: `id`, `image`, `conversations`, `extra` and `type`, where extra stores additional concepts related to the image and their information. For example, a sample may look like this:
```
{
    "id": 156,
    "image": "refcoco/image/94268.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nDoes <D> wear shoes?"
        },
        {
            "from": "gpt",
            "value": "Yes."
        }
    ],
    "extra": {
        "refcoco/crop/94268_0.jpg": {
            "name": "D",
            "info": "A man in a bright orange shirt is playing frisbee.",
            "category": "person"
        }
    },
    "type": "basic"
}
```

### Agreement
- The RAP dataset is available for non-commercial research purposes only, we do not own the rights to these images.
- You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.