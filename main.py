import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from openai import OpenAI
import gradio as gr
from google.colab import userdata

apikey = userdata.get('OPENAI_API_KEY')
client = OpenAI(api_key=apikey)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

if device == "cuda":
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
else:
    pipe.to(device)

def chat(message, history=None):
    """Enhanced prompt generation using OpenAI"""
    system_prompt = """You are to act as an image generation system that will effectively handle users prompt. Once you take in this prompt, properly dissect it to get:
    1. The main prompt - what the user actually wants
    2. The negative prompt - what the user will not want from the prompt given
    
    Make sure you handle this prompt critically and finetune it in such a way that the user gets the optimal needed response.
    
    Please return your response in this exact format:
    PROMPT: [your enhanced positive prompt here]
    NEGATIVE_PROMPT: [your deduced negative prompt here]"""
    
    if history is None:
        history = []
    
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content

def generate_image_from_input(message):
    
    print("🔄 Enhancing prompts with OpenAI...")
    result = chat(message, None)
    
    try:
        lines = result.split('\n')
        prompt = lines[0].replace('PROMPT:', '').strip()
        negative_prompt = lines[1].replace('NEGATIVE_PROMPT:', '').strip()
    except Exception as e:
        print(f"Error parsing prompts: {e}")
        prompt = message + ", high quality, detailed, professional"
        negative_prompt = "blurry, low quality, ugly, deformed, multiple people, duplicate body"
        result = f"PROMPT: {prompt}\nNEGATIVE_PROMPT: {negative_prompt}"
    
    
    print("🎨 Generating image...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=1024,
        width=768,
        generator=torch.Generator(device=device).manual_seed(42)
    ).images[0]
    
    
    return image, prompt, negative_prompt, result

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 AI Image Generator with Smart Prompt Enhancement
    *Describe your idea → AI optimizes the prompt → Get amazing images!*
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💬 Describe Your Image")
            user_input = gr.Textbox(
                label="What do you want to create?",
                placeholder="e.g., a cute cat wearing a hat, cartoon style",
                lines=3
            )
            generate_btn = gr.Button("Generate Image 🚀", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 🖼️ Generated Image")
            image_output = gr.Image(
                label="Your AI Creation",
                height=512,
                show_download_button=True
            )
            
            gr.Markdown("### 📋 Enhanced Prompts")
            with gr.Accordion("View Optimized Prompts", open=True):
                enhanced_prompt_display = gr.Textbox(
                    label="POSITIVE PROMPT",
                    lines=3,
                    interactive=False
                )
                negative_prompt_display = gr.Textbox(
                    label="NEGATIVE PROMPT", 
                    lines=3,
                    interactive=False
                )
                
            raw_prompts_display = gr.Textbox(
                label="Raw OpenAI Response",
                visible=True, 
                lines=3
            )
    
    gr.Markdown("### 💡 Try These Examples:")
    examples = gr.Examples(
        examples=[
            ["a beautiful sunset over mountains, digital art"],
            ["a cyberpunk cityscape at night with neon lights"],
            ["a cute panda eating bamboo in a forest"],
            ["an astronaut riding a horse on Mars, photorealistic"]
        ],
        inputs=user_input
    )
    
    generate_btn.click(
        fn=generate_image_from_input, 
        inputs=[user_input],
        outputs=[image_output, enhanced_prompt_display, negative_prompt_display, raw_prompts_display]
    )

demo.launch(share=True, debug=True)
