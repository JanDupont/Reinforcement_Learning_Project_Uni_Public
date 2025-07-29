from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch
from ray.rllib.utils.annotations import override


class PistonParametricModel(TorchModelV2, nn.Module):
    """Custom CNN model for piston agents using PyTorch"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # CNN layers for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 457, 120) # Default PistonBall Observation Shape Dimensions
            conv_out_size = self.conv_layers(dummy_input).shape[1]
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )
        
        # Value function head
        self.value_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self._last_flat_in = None
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        # Ensure obs is float and normalized
        obs = obs.float() / 255.0
        
        # Convert from (B, H, W, C) to (B, C, H, W) for PyTorch # batch size, h height, w width, c channels
        if len(obs.shape) == 4:
            obs = obs.permute(0, 3, 1, 2)
        else:
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        
        # Pass through conv layers
        conv_out = self.conv_layers(obs)
        self._last_flat_in = conv_out
        
        # Get action logits
        action_logits = self.fc_layers(conv_out)
        
        return action_logits, state
    
    @override(TorchModelV2)
    def value_function(self):
        if self._last_flat_in is not None:
            return self.value_head(self._last_flat_in).squeeze(1)
        else:
            return torch.zeros(1)

class GovernanceModel(TorchModelV2, nn.Module):
    """Custom model for governance agent using PyTorch"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Simple fully connected network
        self.fc_layers = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self._last_flat_in = None
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        # Normalize and flatten observations
        if isinstance(obs, dict):
            # Handle dict observations
            flattened_inputs = []
            for key, value in obs.items():
                flattened_inputs.append(torch.flatten(value.float() / 255.0, start_dim=1))
            flattened = torch.cat(flattened_inputs, dim=1)
        else:
            flattened = torch.flatten(obs.float() / 255.0, start_dim=1)
        
        # Reduce to fixed size if needed
        if flattened.shape[1] > 1000:
            flattened = flattened[:, :1000]
        elif flattened.shape[1] < 1000:
            # Pad with zeros
            padding = torch.zeros(flattened.shape[0], 1000 - flattened.shape[1])
            if flattened.device != padding.device:
                padding = padding.to(flattened.device)
            flattened = torch.cat([flattened, padding], dim=1)
        
        self._last_flat_in = flattened
        action_logits = self.fc_layers(flattened)
        
        return action_logits, state
    
    @override(TorchModelV2)
    def value_function(self):
        if self._last_flat_in is not None:
            return self.value_head(self._last_flat_in).squeeze(1)
        else:
            return torch.zeros(1)