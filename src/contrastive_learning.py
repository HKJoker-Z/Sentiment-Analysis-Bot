import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
# Fix the AdamW import - try different locations based on version
try:
    from transformers import AdamW
except ImportError:
    # For newer versions of transformers
    from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Web3ContrastiveTrainer:
    def __init__(
        self, 
        base_model="distilbert-base-uncased",
        output_dir="./fine_tuned_model",
        temperature=0.07,
        batch_size=16,
        epochs=3
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.temperature = temperature
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModel.from_pretrained(base_model)
        
        # Create projection head for contrastive learning
        model_dim = self.model.config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 128)  # Project to lower dimension for contrastive loss
        )
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def prepare_data(self, data_path):
        """Load and prepare data for contrastive learning"""
        print(f"Reading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Filter valid texts
        df = df[df['text'].notna() & (df['text'].str.strip() != '')].reset_index(drop=True)
        
        # For contrastive learning, we'll create pairs of similar sentences
        # For Web3 content, we'll consider sentences from the same post as similar
        processed_texts = []
        
        for text in tqdm(df['text'].tolist(), desc="Processing texts"):
            if not isinstance(text, str) or len(text.strip()) < 10:
                continue
                
            # Split into sentences - naive approach
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
            
            # If we have multiple sentences, add them as positive pairs
            if len(sentences) > 1:
                for i in range(len(sentences) - 1):
                    processed_texts.append({
                        'anchor': sentences[i],
                        'positive': sentences[i+1]
                    })
        
        print(f"Created {len(processed_texts)} contrastive pairs")
        return processed_texts
    
    class ContrastiveDataset(Dataset):
        def __init__(self, pairs, tokenizer, max_length=128):
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.pairs)
            
        def __getitem__(self, idx):
            pair = self.pairs[idx]
            
            # Tokenize anchor and positive text
            anchor_encoding = self.tokenizer(
                pair['anchor'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            positive_encoding = self.tokenizer(
                pair['positive'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Remove batch dimension added by tokenizer
            return {
                'anchor_input_ids': anchor_encoding['input_ids'].squeeze(),
                'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(),
                'positive_input_ids': positive_encoding['input_ids'].squeeze(),
                'positive_attention_mask': positive_encoding['attention_mask'].squeeze(),
            }
    
    def contrastive_loss(self, features):
        """InfoNCE loss for contrastive learning"""
        batch_size = features.size(0)
        labels = torch.arange(batch_size).to(self.device)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        
        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        # Mask out similarities of each example with itself
        mask = torch.eye(batch_size).bool().to(self.device)
        similarity_matrix.masked_fill_(mask, -float('inf'))
        
        # Compute loss - cross entropy with target being the positive pair
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text with the model and projection head"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        # Apply projection head
        projections = self.projection_head(embeddings)
        # Normalize for cosine similarity
        normalized = F.normalize(projections, p=2, dim=1)
        return normalized
    
    def train_model(self, train_loader):
        """Train the model with contrastive learning"""
        # Move model to device
        self.model.to(self.device)
        self.projection_head.to(self.device)
        
        # Setup optimizer
        optimizer = AdamW([
            {'params': self.model.parameters()},
            {'params': self.projection_head.parameters()}
        ], lr=2e-5)
        
        # Training loop
        self.model.train()
        self.projection_head.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                anchor_input_ids = batch['anchor_input_ids'].to(self.device)
                anchor_attention_mask = batch['anchor_attention_mask'].to(self.device)
                positive_input_ids = batch['positive_input_ids'].to(self.device)
                positive_attention_mask = batch['positive_attention_mask'].to(self.device)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass - encode both anchor and positive
                anchor_features = self.encode_text(anchor_input_ids, anchor_attention_mask)
                positive_features = self.encode_text(positive_input_ids, positive_attention_mask)
                
                # Concatenate features from anchor and positive for contrastive learning
                # Each anchor is contrasted with all other anchors and all positives except its own
                combined_features = torch.cat([anchor_features, positive_features], dim=0)
                
                # Compute loss
                loss = self.contrastive_loss(combined_features)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update progress
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
                
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(train_loader):.4f}")
    
    def save_model(self):
        """Save the fine-tuned model"""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save projection head
        torch.save(self.projection_head.state_dict(), os.path.join(self.output_dir, "projection_head.pt"))
        
        print(f"Model saved to {self.output_dir}")
        
    def finetune(self, data_path="./training_data_unlabeled.csv"):
        """Full contrastive learning workflow"""
        # 1. Prepare data
        pairs = self.prepare_data(data_path)
        
        if len(pairs) < 10:
            print("Error: Not enough text pairs for effective contrastive learning")
            return False
        
        # 2. Create dataset and dataloader
        dataset = self.ContrastiveDataset(pairs, self.tokenizer)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 3. Train model
        self.train_model(train_loader)
        
        # 4. Save model
        self.save_model()
        
        return True
    
    def create_sentiment_classifier(self):
        """Create a sentiment classifier from the contrastively learned model"""
        # For transferring to sentiment classification, we'll add a simple classifier head
        model_dim = self.model.config.hidden_size
        classifier = nn.Linear(model_dim, 2)  # 2 classes: positive/negative
        
        # Initialize with the pre-trained weights
        classifier_model = AutoModel.from_pretrained(self.output_dir)
        
        # Save classifier architecture info for later loading
        torch.save(classifier, os.path.join(self.output_dir, "classifier_head.pt"))
        
        print(f"Sentiment classifier head created and saved to {self.output_dir}")
        return classifier_model, classifier

if __name__ == "__main__":
    # Create trainer
    trainer = Web3ContrastiveTrainer(
        base_model="distilbert-base-uncased",
        output_dir="./fine_tuned_model",
        temperature=0.07,
        batch_size=16,
        epochs=3
    )
    
    # Run fine-tuning
    success = trainer.finetune()
    
    if success:
        # Create sentiment classifier
        classifier_model, classifier_head = trainer.create_sentiment_classifier()
        
        print("Contrastive learning completed successfully!")
        print("The model has learned Web3-specific language representations.")
        print("A sentiment classifier head has been initialized and saved.")
        print(f"You can now use the model at: {trainer.output_dir}")
    else:
        print("Fine-tuning did not complete due to insufficient data.")