import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig
from labels import LABEL2ID, ID2LABEL

class EnhancedTokenClassifier(nn.Module):
    def __init__(self, model_name: str, dropout_prob: float = 0.1):
        super().__init__()
        
        # Load configuration and modify dropout if needed
        config = AutoConfig.from_pretrained(
            model_name, 
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        # Ensure hidden_dropout_prob is set (helpful for regularization on small data)
        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob

        # Load the base HF model
        self.automodel = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Standard forward pass.
        Returns:
            if labels provided: (loss, logits)
            if labels is None:  (pred_ids, logits)
        """
        outputs = self.automodel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        logits = outputs.logits

        if labels is not None:
            # Training mode: return loss and logits
            return outputs.loss, logits
        else:
            # Inference mode: return argmax predictions and logits
            # We apply argmax over the last dimension to get label IDs
            pred_ids = torch.argmax(logits, dim=-1)
            return pred_ids, logits

    def save_pretrained(self, save_directory):
        """Helper to save the underlying HF model"""
        self.automodel.save_pretrained(save_directory)

    @classmethod
    def from_pretrained_custom(cls, model_dir, device="cpu"):
        """
        Helper to load the model for inference.
        Since we wrapped the model, we instantiate the class and load weights.
        """
        # We assume the model_dir contains the standard HF config and pytorch_model.bin
        model = cls(model_dir)
        model.to(device)
        return model

# Function for simple instantiation (used in skeleton, kept for compatibility if needed)
def create_model(model_name: str):
    return EnhancedTokenClassifier(model_name)