from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="depth-anything/Depth-Anything-V2-Small",
    filename="depth_anything_v2_vits.pth",
    local_dir="training/depth/weights",
)

print("Downloaded.")