import torch
from models import S2S 
from _utils import tokenizerSetup  


def load_model(model_checkpoint_path):
    model = S2S.load_from_checkpoint(model_checkpoint_path)
    model.eval() # This will disable gradient calculations, as we do not need it during inference

    return model 

def inference(model, tokenizer, sentence, max_length=50):
    
    input_ids = tokenizer.encode(sentence, return_tensors="pt") #Encode the input sentence to tensor

    # Creating the target tensor and set its initial value to represent the start of a sequence
    target_ids = torch.tensor([tokenizer.bos_token_id], dtype=torch.long).unsqueeze(0) 


    for _ in range(max_length):
        output = model(input_ids, target_ids)
        next_token_id = output.argmax(-1)[:, -1].unsqueeze(-1)
        target_ids = torch.cat((target_ids, next_token_id), dim=-1)
        
        # Stop at the end of the sequence
        if next_token_id.item() == tokenizer.eos_token_id:            
            break

    #Decode and return the generated text
    return tokenizer.decode(target_ids.squeeze(), skip_special_tokens=True)  

if __name__ == "__main__":
    model_path = "output/best_model.ckpt"  # model checkpoint path
    model = load_model(model_path)
    tokenizer = tokenizerSetup()

    test_sentence = "For not use car ."
    corrected_sentence = inference(model, tokenizer, test_sentence)
    print(corrected_sentence)
