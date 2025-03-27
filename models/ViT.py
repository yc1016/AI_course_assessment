import torch
from transformers import ViTForImageClassification, ViTConfig
from thop import profile

class ViT:
    def __init__(self, image_size):
        self.image_size = image_size
        self.config = self._create_config()
        self.model = self._create_model()

    def _create_config(self):
        config = ViTConfig(
            image_size=self.image_size,
            patch_size=8,
            num_channels=3,
            num_labels=15,
            hidden_size=256,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=512
        )
        return config

    def _create_model(self):
        model = ViTForImageClassification(self.config)
        return model
    
    def get_model(self):
        return self.model    
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    model_name = "CNN"

    model = ViT(image_size).to(device)
    input_tensor = torch.randn(1, 3, image_size, image_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs of {model_name} for {image_size}x{image_size}: {flops}")
    print(f"Params of {model_name} for {image_size}x{image_size}: {params}")