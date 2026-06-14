import os
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration
)

MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def load_model():

    try:


        dtype = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        model.to("cuda")

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        model.eval()

        return model, processor

    except Exception as e:

        print(f"Error loading model: {e}")

        return None, None, None


def generate_caption(model, processor, image_path):

    try:

        image = Image.open(image_path).convert("RGB")

        prompt = (
            "USER: <image>\n"
            "Generate a concise diffusion training caption. "
            "If there is a boat, call it "
            "T_U_G_B_O_A_T. "
            "Mention important visual elements, lighting, "
            "environment, colors, and camera angle.\n"
            "ASSISTANT:"
        )

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                num_beams=1
            )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[-1]

        return caption.strip()

    except Exception as e:

        print(f"Error processing {image_path}: {e}")

        return ""


def main():

    model, processor, device = load_model()

    if model is None:
        return

    image_folder = "./training_flux/final_train"

    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        )
    ]

    print(f"Found {len(image_files)} images")

    for image_file in image_files:

        image_path = os.path.join(
            image_folder,
            image_file
        )

        base_name, _ = os.path.splitext(image_file)

        txt_path = os.path.join(
            image_folder,
            f"{base_name}.txt"
        )

        print(f"Processing {image_file}")

        caption = generate_caption(
            model,
            processor,
            image_path
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)

        print(caption)
        print("-" * 80)


if __name__ == "__main__":
    main()