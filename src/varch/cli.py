from pathlib import Path
import os
import warnings
import torch
import typer
import open_clip
from varch.visual_archive import VisualArchive
from varch.encoder import MODEL as CLIP_MODEL
from varch.encoder import PRETRAINED
from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration)
from varch.vlm import MODEL as QWEN_MODEL
from huggingface_hub.constants import HF_HUB_CACHE


# silence warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*_check_is_size will be removed in a future PyTorch release.*"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

app = typer.Typer(help="varch - local multimodal image RAG system")


# how the retrived info is printed back to the user
def build_output(images_paths, scores, rag_answer=None):
    output_lines = ["\n--- Retrieved Images (Ctrl+Click to open) ---"]
    if rag_answer: output_lines = ["\n--- varch Answer ---", rag_answer] + output_lines
    for i, (path, score) in enumerate(zip(images_paths, scores)):
        output_lines.append(f"Image {i+1} [Score: {score:.4f}]: {path}")
    output_lines.append("-----------------------------------------------------------------\n")
    return "\n".join(output_lines)


# Commands

@app.command()
def init():
    typer.echo("initializing varch...")
    try:
        typer.echo("downloading open-clip...")
        _, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL,
            pretrained=PRETRAINED
        )
        _ = open_clip.get_tokenizer(CLIP_MODEL)
        typer.echo(f"open-clip saved to {HF_HUB_CACHE}")
        typer.echo("open-clip ready")

        typer.echo("downloading qwen-vl...")
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        AutoProcessor.from_pretrained(QWEN_MODEL)
        typer.echo(f"qwen-vl saved to {HF_HUB_CACHE}")
        typer.echo("qwen-vl ready")

        typer.echo("varch initialization completed successfully")

    except Exception as e:
        typer.echo(f"[ERROR] {e}")
        raise typer.Exit(code=1)


# observe all images in path and embedd into the DB
@app.command()
def observe(path: str = typer.Argument(...,help="path to image folder")):
    visual_archive = VisualArchive(path=Path.cwd(), device=device, load_db=False)
    visual_archive.observe(path)

# load & search the archive
@app.command()
def search(k: int = typer.Option(5, "-k", help="number of retrieved images"), fast_retrieval: bool = typer.Option(False, "--fr", help="use faster retrieval mode")):
    visual_archive = VisualArchive(path=Path.cwd(), device=device, load_db=True, load_vlm=not fast_retrieval)
    while True:
        query: str = input('query: ')
        if query == '~terminate': 
            break 
        relevant_paths, scores, rag_answer = visual_archive.search(query, k)
        answer = build_output(relevant_paths, scores, rag_answer)
        typer.echo(answer)




def main():
    app()
if __name__ == "__main__":
    main()
    