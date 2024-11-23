from llava.train.rap_train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
