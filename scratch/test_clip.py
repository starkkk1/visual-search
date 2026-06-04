import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

def test_text():
    model_name = "openai/clip-vit-base-patch32"
    
    print("Loading text_model...")
    text_model = CLIPTextModelWithProjection.from_pretrained(model_name)
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    
    texts = ["a red square", "hello world"]
    inputs_t = tokenizer(texts, padding=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = text_model(**inputs_t)
        out_t1 = outputs.text_embeds
        
        print("out_t1 type:", type(out_t1))
        print("out_t1 shape:", out_t1.shape)
        
        # Norm and normalize
        arr = out_t1.cpu().numpy()
        norms = torch.linalg.norm(out_t1, dim=1, keepdim=True)
        print("norms:", norms)
        out_t1_normalized = out_t1 / norms
        print("normalized shape:", out_t1_normalized.shape)

if __name__ == "__main__":
    test_text()
