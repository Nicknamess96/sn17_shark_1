import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
from pipelines.models import TextToImageRequest
from torch import Generator
from DeepCache import DeepCacheSDHelper

def load_pipeline() -> StableDiffusionXLPipeline:
    # Load TinyVAE
    vae = AutoencoderTiny.from_pretrained(
        'madebyollin/taesdxl',
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Load the main pipeline with TinyVAE
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "./models/newdream-sdxl-20",
        torch_dtype=torch.float16,
        local_files_only=True,
        vae=vae,
    ).to("cuda")

    # Initialize DeepCache
    helper = DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()

    # Warm up the pipeline
    pipeline(prompt="")

    return pipeline

def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
    ).images[0]