import os
import typer
import logging
import warnings
from pathlib import Path

import torch
import open_clip
from transformers import utils
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub import scan_cache_dir

from varch.visual_archive import VisualArchive
from varch.encoder import MODEL as CLIP_MODEL
from varch.encoder import PRETRAINED
from varch.vlm import MODEL as QWEN_VLM_MODEL
from varch.slm import MODEL as QWEN_SLM_MODEL


# silence warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*_check_is_size will be removed in a future PyTorch release.*")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
utils.logging.set_verbosity_error()



device = "cuda" if torch.cuda.is_available() else "cpu"
app = typer.Typer(help="varch - VisualArchive, local image RAG system")


def build_output(images_paths, scores, rag_answer=None):
    """Constructs the relevant info and prints it back to the user"""

    output_lines = ["--- Relevant Images (Ctrl+Click to open) ---"]
    if rag_answer: 
        styled_rag_answer = typer.style(rag_answer, fg=typer.colors.BRIGHT_CYAN)
        output_lines = ["\n", styled_rag_answer] + output_lines
    for i, (path, score) in enumerate(zip(images_paths, scores)):
        output_lines.append(f"Image {i+1} [Score: {score:.4f}]: {path}")
    output_lines.append("-----------------------------------------------------------------\n")
    return "\n".join(output_lines)


# Commands

@app.command()
def status():
    """Prints the current device-chache-db status"""
    
    # device
    device_color = typer.colors.GREEN if device == 'cuda' else typer.colors.BLUE
    styled_device = typer.style(device, fg=device_color)

    # cache
    clip_repo = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    qwen_vlm_repo = QWEN_VLM_MODEL
    qwen_slm_repo = QWEN_SLM_MODEL

    try:
        cached_repos = {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        cached_repos = set()
    clip_cached = clip_repo in cached_repos
    qwen_vlm_cached = qwen_vlm_repo in cached_repos
    qwen_slm_cached = qwen_slm_repo in cached_repos

    styled_clip = typer.style(clip_repo, fg=typer.colors.GREEN if clip_cached else typer.colors.RED)
    styled_vlm_qwen = typer.style(qwen_vlm_repo, fg=typer.colors.GREEN if qwen_vlm_cached else typer.colors.RED)
    styled_slm_qwen = typer.style(qwen_slm_repo, fg=typer.colors.GREEN if qwen_slm_cached else typer.colors.RED)

    # db
    db_dir = Path(os.getcwd()) / "db"
    db_files = {"embeddings.index", "paths.npy"}
    db_status = db_dir.exists() and all((db_dir / file_name).exists() for file_name in db_files)

    styled_db = typer.style(f"ready" if db_status else f"missing", fg=typer.colors.GREEN if db_status else typer.colors.RED)

    typer.echo(f"device: {styled_device}")
    typer.echo(f"hf cache: {styled_clip}   {styled_vlm_qwen}   {styled_slm_qwen}")
    typer.echo(f"database: {styled_db}")
 
@app.command()
def init():
    """Downlaods all necessary models to hf cache"""

    typer.echo("initializing varch...")
    try:
        typer.echo("downloading open-clip...")
        _, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL,pretrained=PRETRAINED)
        _ = open_clip.get_tokenizer(CLIP_MODEL)
        typer.echo(f"open-clip saved to {HF_HUB_CACHE}")
        typer.echo("open-clip ready")

        typer.echo("downloading qwen-vl...")
        snapshot_download(repo_id=QWEN_VLM_MODEL, local_files_only=False)
        typer.echo(f"qwen-vl saved to {HF_HUB_CACHE}")
        typer.echo("qwen-vl ready")

        typer.echo("downloading qwen-lm...")
        snapshot_download(repo_id=QWEN_SLM_MODEL, local_files_only=False)
        typer.echo(f"qwen-lm saved to {HF_HUB_CACHE}")
        typer.echo("qwen-lm ready")

        typer.echo("varch initialization completed successfully")

    except Exception as e:
        typer.echo(f"[ERROR] {e}")
        raise typer.Exit(code=1)

@app.command()
def observe(path: str = typer.Argument(...,help="path to image folder")):
    """Call to observes all images in path"""

    visual_archive = VisualArchive(path=Path.cwd(), device=device, load_db=False)
    visual_archive.observe(path)

@app.command()
def search(
    k: int = typer.Option(5, "-k", help="number of retrieved images"), 
    fast_retrieval: bool = typer.Option(False, "--fr", help="use faster retrieval mode"),
    dual_modality: bool = typer.Option(False, "--dm", help="search with image and text")):
    """Call to a search on the archive"""

    visual_archive = VisualArchive(path=Path.cwd(), device=device, load_db=True, load_vlm=not fast_retrieval, load_slm=dual_modality)
    while True:
        image_path = None
        if dual_modality: 
            image_path: str = input('image: ')
            if image_path == '~terminate': break 
        query: str = input('query: ')
        if query == '~terminate': break 
        relevant_paths, scores, rag_answer = visual_archive.search(query, image_path=image_path, k=k)
        answer = build_output(relevant_paths, scores, rag_answer)
        typer.echo(answer)


def main():
    app()
if __name__ == "__main__":
    main()
    