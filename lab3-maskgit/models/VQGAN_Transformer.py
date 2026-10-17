import torch 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs, mask_func):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma_type = mask_func if mask_func else configs['gamma_type']
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

        print(f"Gamma function type: {self.gamma_type}")

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        zq, z_indices, _ = self.vqgan.encode(x)
        z_indices = z_indices.view(zq.shape[0], -1)
        return zq, z_indices
    
##TODO2 step1-2:    
    def gamma_func(self, ratio, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            mask_ratio = 1.0 - ratio
        elif mode == "cosine":
            mask_ratio = np.cos(np.pi / 2 * ratio)
        elif mode == "square":
            mask_ratio = 1.0 - ratio ** 2
        else:
            raise NotImplementedError

        return np.clip(mask_ratio, 1e-6, 1.0)

##TODO2 step1-3:            
    def forward(self, x):
        zq, z_indices = self.encode_to_z(x)

        # Generate mask
        r = np.random.uniform(0, 1)
        mask_ratio = self.gamma_func(r, mode=self.gamma_type)
        v = torch.rand(z_indices.shape, device=z_indices.device)
        mask = (v < mask_ratio)
        
        # Prepare masked indices
        masked_indices = z_indices.clone()
        masked_indices[mask] = self.mask_token_id
        
        logits = self.transformer(masked_indices)
        
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, ratio, initial_mask=None):
        logits = self.transformer(z_indices)  # (B, num_tokens, codebook_size+1)
        logits_for_prob = logits[:, :, :self.mask_token_id]  # ignore the last token (mask token)
        
        # Sample tokens instead of argmax
        probs = F.softmax(logits_for_prob, dim=-1)
        B, N, K = probs.shape
        z_indices_predict = torch.multinomial(probs.view(-1, K), 1).view(B, N)
        
        # Use the probability of the sampled token as confidence (add gumbel noise)
        z_indices_predict_prob = torch.gather(probs, -1, z_indices_predict.unsqueeze(-1)).squeeze(-1)
        temperature = self.choice_temperature * (1 - ratio)
        g = torch.distributions.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices.device)
        confidence = z_indices_predict_prob + temperature * g
        
        confidence[~mask] = float('inf')  # Unmasked region
        
        sorted_indices = torch.argsort(confidence, dim=-1)  # (B, num_tokens) ascending
        
        # How many tokens to unmask
        mask_ratio = self.gamma_func(ratio, mode=self.gamma_type)
        n_mask = math.ceil(mask_ratio * initial_mask.sum())
        
        # New mask: lowest confidence tokens stay masked
        mask_bc = torch.zeros_like(mask)
        for i in range(z_indices.shape[0]):
            mask_bc[i, sorted_indices[i, :n_mask]] = True
        
        # Ensure mask_bc is a subset of the original mask
        mask_bc = mask_bc & initial_mask

        # Update z_indices for the next step
        z_indices_new = z_indices.clone()
        # Revealed tokens: the ones that were masked but are now unmasked
        revealed = mask & ~mask_bc
        z_indices_new[revealed] = z_indices_predict[revealed]

        return z_indices_new, mask_bc, z_indices_predict
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
