from typing import List
from beartype import beartype

import torch
import transformers 
from transformers import AutoTokenizer, AutoModel,AutoConfig
transformers.logging.set_verbosity_error()

 

class BGEAdapter():
    def __init__(
        self,
        name,
        text_embed_pad_value = 0.
    ):
        name = 'BAAI/bge-base-en-v1.5'
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        self.Config = AutoConfig.from_pretrained(name)
        
        if torch.cuda.is_available():
            model = model.to("cuda")  
            
        self.name =  name
        self.model = model
        self.tokenizer = tokenizer
        self.text_embed_pad_value = text_embed_pad_value

    @property
    def dim_latent(self):
        return self.Config.hidden_size

    @property
    def max_text_len(self):
        return 512

    @torch.no_grad()
    @beartype
    def embed_text(
        self,
        texts: List[str],
        return_text_encodings = False,
        output_device = None
    ):
        if output_device is None:
            output_device = self.model.device
        elif output_device != self.model.device:
            self.model = self.model.to(output_device) 
               
        encoded_input  = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(output_device)
 
        self.model.eval()
         
        with torch.no_grad():
            model_output = self.model(**encoded_input)  
            
        if not return_text_encodings: 
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings  # Return normalized CLS embedding
        
        encoded_text = model_output.last_hidden_state.to(output_device)
        attn_mask = encoded_input.attention_mask.bool()

        encoded_text = encoded_text.masked_fill_(~attn_mask[..., None], self.text_embed_pad_value)

        return encoded_text
