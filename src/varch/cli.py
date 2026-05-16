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


# silence warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*_check_is_size will be removed in a future PyTorch release.*"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

app = typer.Typer(help="varch - local multimodal image RAG system")


# Commands

# downloads and installs all necessary things
@app.command()
def init():

    typer.echo("initializing varch...")

    typer.echo(f"detected device: {"cpu" if device != 'cuda' else torch.cuda.get_device_name(0)}")

    try:
        typer.echo("downloading open-clip...")
        _, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL,PRETRAINED)
        _ = open_clip.get_tokenizer(CLIP_MODEL)
        typer.echo("open-clip cached at ~/.cache/huggingface/hub")
        typer.echo("open-clip ready")

        typer.echo("downloading qwen-vl...")
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa")
        AutoProcessor.from_pretrained(QWEN_MODEL)
        typer.echo("qwen-vl cached at ~/.cache/huggingface/hub")
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
def search(k: int = typer.Option(5,help="top-k retrieval count")):
    visual_archive = VisualArchive(path=Path.cwd(), device=device, load_db=True)
    while True:
        query: str = input('query: ')
        if query == '~terminate': 
            break 
        answer = visual_archive.search(query, k)
        typer.echo(answer)


def main():
    app()
if __name__ == "__main__":
    main()
    