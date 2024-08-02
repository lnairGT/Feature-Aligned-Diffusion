import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPProcessor, CLIPModel
import torchvision


class PreTrainedDiffusion(torch.nn.Module):
    def __init__(self, model_name="segmind/tiny-sd"):
        super().__init__()
        self.model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

        # CLIP text encoder used to generate text embeddings of class labels
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()

        # Map text embeddings to image embeddings dimension
        self.text_to_cross_dim = torch.nn.Linear(512, self.model.config.cross_attention_dim)
        
        # Generate 1-channel grayscale images by replacing first and last conv of
        # the stable diffusion model
        self.model.conv_in = torch.nn.Conv2d(
            1, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.model.conv_out = torch.nn.Conv2d(
            320, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

    def get_text_features(self, inputs):
        with torch.no_grad():
            inputs = self.clip_model.get_text_features(**inputs)
        return inputs

    def forward(self, img, timestep, class_labels):
        inputs = self.processor(text=class_labels, return_tensors="pt", padding=True)
        inputs = {k: v.to(img.device) for k, v in inputs.items()}
        inputs = self.get_text_features(inputs)
        inputs = self.text_to_cross_dim(inputs)

        inputs = inputs.unsqueeze(1)
        return self.model(img, timestep, encoder_hidden_states=inputs).sample


class FeatureAlignedDiffusion(torch.nn.Module):
    def __init__(self, expert_ckpt, model_name="segmind/tiny-sd"):
        super().__init__()
        self.model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.expert_model = ExpertModel()

        # Some updates to checkpoint since we don't need the last classification layer
        ckpt = torch.load(expert_ckpt)
        layers = [
            'resnet.fc.0.weight',
            'resnet.fc.0.bias',
            'resnet.fc.2.weight',
            'resnet.fc.2.bias'
        ]
        for l in layers:
            del ckpt[l]
        self.expert_model.load_state_dict(ckpt)
        self.expert_model.eval()

        # Same as baseline model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        self.text_to_cross_dim = torch.nn.Linear(512, self.model.config.cross_attention_dim)
        self.model.conv_in = torch.nn.Conv2d(
            1, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.model.conv_out = torch.nn.Conv2d(
            320, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        # Attach Pytorch hooks that get intermediate feature maps
        self._attach_layer_hook()
        # Average pooling applied to unet downsample features
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # Mapping from expert feature dimension to diffusion dimensions
        self.map_expert_to_unet = torch.nn.Linear(2048, 1280)

    def get_text_features(self, inputs):
        with torch.no_grad():
            inputs = self.clip_model.get_text_features(**inputs)
        return inputs
    
    def get_expert_features(self, img):
        with torch.no_grad():
            self.expert_embeddings = self.expert_model(img)

    def _get_output_embedding_score(self, module, inputs, outputs):
        mapped_embeddings = self.map_expert_to_unet(self.expert_embeddings)
        self.align_loss = 1 - torch.nn.functional.cosine_similarity(
            mapped_embeddings, self.adaptive_avg_pool(outputs).squeeze()
        )
        self.align_loss = torch.mean(self.align_loss)  # Align loss
        

    def _attach_layer_hook(self):
        for n, m in self.model.named_modules():
            # Last layer in downsampling block
            if n == "down_blocks.2.resnets.0.conv_shortcut":
                m.register_forward_hook(self._get_output_embedding_score)

    def forward(self, img, timestep, class_labels, modality="text"):
        self.get_expert_features(img)
        inputs = self.processor(text=class_labels, return_tensors="pt", padding=True)
        inputs = {k: v.to(img.device) for k, v in inputs.items()}
        inputs = self.get_text_features(inputs)
        inputs = self.text_to_cross_dim(inputs)

        inputs = inputs.unsqueeze(1)
        return self.model(img, timestep, encoder_hidden_states=inputs).sample


class ExpertModel(torch.nn.Module):
    def __init__(self):
        super(ExpertModel, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   
        self.resnet.fc = torch.nn.Identity()

    def forward(self, images):
        output = self.resnet(images)
        return output


class ExpertEvaluator(torch.nn.Module):
    def __init__(self, model="resnet"):
        super(ExpertEvaluator, self).__init__()

        if model == "resnet":
            # get resnet model
            self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Replace last linear layer to predict 8 classes (not 1000)
            self.resnet.fc = torch.nn.Sequential(
                torch.nn.Linear(2048, 256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(256, 8)
            )
        else:
            self.resnet = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
            self.resnet.conv_proj = torch.nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
            # Replace last linear layer to predict 8 classes (not 1000)
            self.resnet.heads.head = torch.nn.Sequential(
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(256, 8)
            )
        self.softmax = torch.nn.Softmax()

    def forward(self, images):
        output = self.resnet(images)
        output = self.softmax(output)
        return output
