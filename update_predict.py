import re

# Leer el archivo
with open('img2img.py', 'r') as f:
    content = f.read()

# Nuevo método predict con lógica de strength
new_predict = '''    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Calcular t_index basado en strength (0-3 para SD-Turbo con 4 pasos)
        # strength 1.0 -> t_index 0 (máximo cambio)
        # strength 0.0 -> t_index 3 (mínimo cambio)
        t_index = int((1.0 - params.strength) * 3)
        
        # Si el t_index cambió, recrear el stream
        current_t_index = self.stream.t_index_list[0] if hasattr(self.stream, 't_index_list') else None
        
        if current_t_index != t_index:
            # Recrear stream con nuevo t_index
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=base_model,
                use_tiny_vae=self.args.taesd,
                device=self.device,
                dtype=self.torch_dtype,
                t_index_list=[t_index],
                frame_buffer_size=1,
                width=512,
                height=512,
                use_lcm_lora=False,
                output_type="pil",
                warmup=10,
                vae_id=None,
                acceleration=self.args.acceleration,
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="none",
                use_safety_checker=self.args.safety_checker,
                engine_dir=self.args.engine_dir,
            )
            
            self.stream.prepare(
                prompt=params.prompt,
                negative_prompt=default_negative_prompt,
                num_inference_steps=4,
                guidance_scale=1.2,
            )
        
        image_tensor = self.stream.preprocess_image(params.image)
        output_image = self.stream(image=image_tensor, prompt=params.prompt)
        return output_image'''

# Reemplazar el método predict
pattern = r'def predict\(self.*?\n        return output_image'
content = re.sub(pattern, new_predict.strip(), content, flags=re.DOTALL)

# Añadir atributos necesarios en __init__
init_additions = '''        self.args = args
        self.device = device
        self.torch_dtype = torch_dtype
        '''

# Insertar después de def __init__
content = content.replace('def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):\n', 
                         'def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):\n' + init_additions)

# Cambiar t_index_list inicial y num_inference_steps
content = content.replace('t_index_list=[0]', 't_index_list=[2]')  # Empezar con strength medio
content = content.replace('num_inference_steps=1', 'num_inference_steps=4')

# Escribir el archivo modificado
with open('img2img.py', 'w') as f:
    f.write(content)

print("Archivo actualizado exitosamente")
