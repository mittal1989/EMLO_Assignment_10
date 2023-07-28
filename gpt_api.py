import torch
import tiktoken

from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
loaded_model = torch.jit.load("./Models/model.script.gpt.pt", map_location="cpu")
loaded_model.eval()

# tokenizer
cl100k_base = tiktoken.get_encoding("cl100k_base")

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
tokenizer = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)

def text_completion(text: str) -> str:
    input_enc = torch.tensor(tokenizer.encode(text))
    with torch.no_grad():
        out_gen = loaded_model.model.generate(input_enc.unsqueeze(0).long(), max_new_tokens=32)
    decoded = tokenizer.decode(out_gen[0].cpu().numpy().tolist())
    return decoded

class Query(BaseModel):
    text: str
    # max_new_tokens: int | None = None
    max_new_tokens: int 


@app.get("/infer")
async def infer(query: Query):
    text = query.text
    max_new_tokens = query.max_new_tokens

    input_enc = torch.tensor(tokenizer.encode(text))
    with torch.no_grad():
        out_gen = loaded_model.model.generate(input_enc.unsqueeze(0).long(), max_new_tokens)
    decoded = tokenizer.decode(out_gen[0].cpu().numpy().tolist())

    return {decoded}

@app.get("/health")
async def health():
    return {"message": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)