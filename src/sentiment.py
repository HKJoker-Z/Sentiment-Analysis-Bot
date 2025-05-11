import pandas as pd
from transformers import pipeline
import torch
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class SentimentAnalyzer:
    def __init__(self, model_name=None):
        """
        Initialize sentiment analyzer
        
        Parameters:
            model_name (str): Pre-trained model name or path to fine-tuned model
        """
        # Check for contrastively fine-tuned model
        fine_tuned_path = "./fine_tuned_model"
        if model_name is None and os.path.exists(fine_tuned_path):
            print(f"Using contrastively fine-tuned Web3 model from {fine_tuned_path}")
            self.use_contrastive_model = True
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
            self.model = AutoModel.from_pretrained(fine_tuned_path)
            
            # Load classifier head - fix for PyTorch 2.6+ security changes
            classifier_path = os.path.join(fine_tuned_path, "classifier_head.pt")
            if os.path.exists(classifier_path):
                try:
                    # Add safe globals to allow torch.nn.modules.linear.Linear to be unpickled
                    torch.serialization.add_safe_globals(["torch.nn.modules.linear.Linear"])
                    self.classifier = torch.load(classifier_path)
                except Exception as e:
                    print(f"Failed to load classifier using default approach: {e}")
                    # Fallback method using weights_only=False (only if the file is trusted)
                    try:
                        self.classifier = torch.load(classifier_path, weights_only=False)
                    except Exception as inner_e:
                        print(f"Failed to load classifier with fallback: {inner_e}")
                        # Create a new classifier head if loading fails
                        print("Creating new classifier head")
                        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 2)
            else:
                # If no classifier head found, create a simple one
                print("No classifier head found, creating new one")
                self.classifier = torch.nn.Linear(self.model.config.hidden_size, 2)
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.classifier.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            self.classifier.eval()
            
        else:
            # Default: use Hugging Face pipeline
            model_name = model_name or "distilbert-base-uncased-finetuned-sst-2-english"
            self.use_contrastive_model = False
            # Force CPU usage to avoid GPU issues
            self.device = -1
            
            print(f"Loading sentiment analysis model: {model_name}...")
            # Initialize transformers sentiment analysis pipeline
            from transformers import pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=model_name, 
                device=self.device
            )
            
        print("Model loading complete!")
        
    def analyze_batch(self, texts, batch_size=16):
        """
        Batch analyze sentiment of texts
        
        Parameters:
            texts (list): List of texts to analyze
            batch_size (int): Batch processing size
            
        Returns:
            list: List containing sentiment analysis result for each text
        """
        # Avoid processing very long texts, Transformer models usually limit input length
        truncated_texts = [text[:512] if isinstance(text, str) else "" for text in texts]
        
        # Filter out empty strings
        valid_indices = [i for i, text in enumerate(truncated_texts) if text.strip()]
        valid_texts = [truncated_texts[i] for i in valid_indices]
        
        if not valid_texts:
            return []
            
        if self.use_contrastive_model:
            # Process with our contrastive model
            results = []
            
            with torch.no_grad():
                for i in range(0, len(valid_texts), batch_size):
                    batch_texts = valid_texts[i:i+batch_size]
                    
                    # Tokenize
                    encoded = self.tokenizer(
                        batch_texts,
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get model outputs
                    outputs = self.model(**encoded)
                    cls_embeds = outputs.last_hidden_state[:, 0, :]
                    
                    # Apply classifier
                    logits = self.classifier(cls_embeds)
                    probs = F.softmax(logits, dim=1)
                    
                    # Convert to expected format
                    for j in range(len(batch_texts)):
                        negative_prob = probs[j, 0].item()
                        positive_prob = probs[j, 1].item()
                        
                        if positive_prob > negative_prob:
                            results.append({
                                "label": "POSITIVE", 
                                "score": positive_prob
                            })
                        else:
                            results.append({
                                "label": "NEGATIVE", 
                                "score": negative_prob
                            })
        else:
            # Process with standard pipeline
            results = []
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i+batch_size]
                batch_results = self.sentiment_pipeline(batch_texts)
                results.extend(batch_results)
        
        # Map results back to original indices
        full_results = [{"label": "NEUTRAL", "score": 0.5} for _ in range(len(texts))]
        for i, result in zip(valid_indices, results):
            full_results[i] = result
            
        return full_results
        
    def analyze_dataframe(self, df, text_column):
        """
        Analyze sentiment of text in a specified DataFrame column
        
        Parameters:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Column name containing text to analyze
            
        Returns:
            pd.DataFrame: Original DataFrame with sentiment analysis results added
        """
        if df.empty:
            return df
            
        # Copy DataFrame to avoid modifying original data
        result_df = df.copy()
        
        # Get text column data
        texts = result_df[text_column].tolist()
        
        # Batch analyze sentiment
        sentiment_results = self.analyze_batch(texts)
        
        # Add sentiment labels and scores to DataFrame
        result_df["sentiment"] = [result["label"] for result in sentiment_results]
        result_df["sentiment_score"] = [result["score"] for result in sentiment_results]
        
        # Add sentiment category
        result_df["sentiment_category"] = result_df["sentiment"].apply(self._map_sentiment_label)
        
        return result_df
    
    def _map_sentiment_label(self, label):
        """
        Map model output labels to more readable sentiment categories
        """
        if label in ["POSITIVE", "LABEL_1"]:
            return "positive"
        elif label in ["NEGATIVE", "LABEL_0"]:
            return "negative"
        else:
            return "neutral"
            
# Test code
if __name__ == "__main__":
    # Create test data
    test_data = {
        "id": ["1", "2", "3", "4", "5"],
        "text": [
            "I love blockchain technology and web3 development!",
            "This cryptocurrency project seems like a scam.",
            "The market is very volatile right now, I'm not sure what to do.",
            "The new Ethereum upgrade will solve many scaling issues.",
            ""  # Empty string test
        ]
    }
    test_df = pd.DataFrame(test_data)
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze test data
    result_df = analyzer.analyze_dataframe(test_df, "text")
    
    # Print results
    for i, row in result_df.iterrows():
        print(f"Text: {row['text']}")
        print(f"Sentiment: {row['sentiment_category']} (Label: {row['sentiment']}, Score: {row['sentiment_score']:.4f})")
        print("-" * 80)