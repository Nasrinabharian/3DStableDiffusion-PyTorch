import torch
import torch.nn as nn
from models.blocks import DownBlock, MidBlock, UpBlock


class VQVAE(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        
        # Reduce the number of channels
        self.down_channels = [int(x // 2) for x in model_config['down_channels']]  # Reduced channels
        self.mid_channels = [int(x // 2) for x in model_config['mid_channels']]    # Reduced mid channels
        self.down_sample = model_config['down_sample']
        
        # Reduce number of layers
        self.num_down_layers = max(1, model_config['num_down_layers'] // 2)  # Reduced layers
        self.num_mid_layers = max(1, model_config['num_mid_layers'] // 2)    # Reduced mid layers
        self.num_up_layers = max(1, model_config['num_up_layers'] // 2)      # Reduced up layers
        
        self.attns = model_config['attn_down']
        
        # Latent dimension
        self.z_channels = max(16, model_config['z_channels'] // 2)   # Reduced latent channels
        self.codebook_size = max(128, model_config['codebook_size'] // 2)  # Reduced codebook size
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        self.up_sample = list(reversed(self.down_sample))
        
        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv3d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1, 1))
        
        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv3d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)
        
        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv3d(self.z_channels, self.z_channels, kernel_size=1)
        
        # Codebook
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)
        ####################################################
        
        ##################### Decoder ######################
        
        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv3d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv3d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1, 1))
        
        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i-1],
                                               norm_channels=self.norm_channels))
        
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv3d(self.down_channels[0], im_channels, kernel_size=3, padding=1)
    
    def quantize(self, x):
        B, C, D, H, W = x.shape
        
        # B, C, D, H, W -> B, D, H, W, C
        x = x.permute(0, 2, 3, 4, 1)
        
        # B, D, H, W, C -> B, D*H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))
        
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        quant_out = x + (quant_out - x).detach()
        
        quant_out = quant_out.reshape((B, D, H, W, C)).permute(0, 4, 1, 2, 3)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-3), quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices

    def encode(self, x):
        out = self.encoder_conv_in(x)

        #Nasrin
        #RuntimeError: Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [4] and input of shape [4, 64, 64, 64]
        
        '''if out.dim() == 4:
            print("unsqueezed1  *****")
            out = torch.unsqueeze(out, 0)'''
            
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        out, quant_losses, _ = self.quantize(out)
        return out, quant_losses
    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)
        
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out
    
    def forward(self, x):
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return out, z, quant_losses