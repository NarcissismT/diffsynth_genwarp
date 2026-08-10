import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from PIL import Image


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig("/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors"),
        ModelConfig("/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/text_encoder/model.safetensors"),
        ModelConfig("/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/text_encoder_2"),
        ModelConfig("/juicefs-algorithm/data/IPT/haonan_wu/Models/Qwen-Image/black-forest-labs/FLUX.1-Kontext-dev/ae.safetensors"),
    ],
)
pipe.load_lora(pipe.dit, "/juicefs-algorithm/data/IPT/junle_liu/slurm/slurm-result/lora/FLUX.1-Kontext-dev_lora/20250825/dehighlight_compareyuyi/step-4000.safetensors", alpha=1)

img_path="/juicefs-algorithm/data/IPT/junle_liu/滤镜/证件测试集采样/certificate/reflection_sample/驾驶证/12.jpg"
img = Image.open(img_path)
orig_width, orig_height = img.size

image = pipe(
    prompt="Enhance this image to high definition, remove reflections, and balance light and shadow.",
    kontext_images=Image.open(img_path).resize((768, 768)),
    width=768,
    height=768
)
image = image.resize((orig_width, orig_height))
image.save("image_FLUX.1-Kontext-dev_lora.jpg")
