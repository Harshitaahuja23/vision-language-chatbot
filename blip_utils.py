import torch
from PIL import Image
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration, 
    BlipForQuestionAnswering, 
    BlipForImageTextRetrieval
)

# Load BLIP models once
_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

_vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
_vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

_retrieval_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
_retrieval_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

def generate_caption(image: Image.Image, use_nucleus: bool = True) -> str:
    """Generate caption for image with nucleus sampling or beam search."""
    inputs = _caption_processor(image, return_tensors="pt")
    if use_nucleus:
        output = _caption_model.generate(
            **inputs, num_beams=3, num_return_sequences=1,
            max_length=20, top_p=0.9, temperature=0.7
        )
    else:
        output = _caption_model.generate(
            **inputs, num_beams=3, num_return_sequences=1,
            max_length=20
        )
    caption = _caption_processor.decode(output[0], skip_special_tokens=True)
    return caption

def answer_question(image: Image.Image, question: str) -> str:
    """Answer a question about the image."""
    inputs = _vqa_processor(image, question, return_tensors="pt")
    output = _vqa_model.generate(**inputs, max_length=20)
    answer = _vqa_processor.decode(output[0], skip_special_tokens=True)
    return answer

def check_image_text_match(image: Image.Image, text: str) -> float:
    """Check if the image and user-provided text match. Returns match probability."""
    inputs = _retrieval_processor(image, text, return_tensors="pt")
    outputs = _retrieval_model(**inputs)
    itm_score = torch.nn.functional.softmax(outputs.itm_score, dim=1)
    match_probability = itm_score[:, 1].item()
    return match_probability