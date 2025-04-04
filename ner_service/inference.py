import torch
from transformers import AutoTokenizer
from src.models.transformer_model import TransformerModel
from src.utils.config_loader import load_config
from src.utils.label_mapping_regplans import label_to_id, id_to_label
from src.utils.logger import setup_logger
from pathlib import Path
from tqdm import tqdm
import spacy
import pdfplumber
import re
from typing import List
import os

logger = setup_logger(__name__)

# Global variables for models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

class ModelManager:
    """Manages model initialization and access"""
    def __init__(self):
        self.nlp = None
        self.tokenizer = None
        self.model = None
        self.config = None
        
    def initialize(self):
        """Initialize all required models"""
        try:
            logger.info("Starting model initialization")
            
            # Load configuration
            base_dir = Path(__file__).parent
            possible_config_paths = [
                base_dir / 'src' / 'model_params.yaml',
                Path('src/model_params.yaml')
            ]
            
            config_path = None
            for path in possible_config_paths:
                if path.exists():
                    config_path = path
                    break
                    
            if not config_path:
                paths_str = '\n'.join(str(p) for p in possible_config_paths)
                logger.error(f"Configuration file not found. Tried:\n{paths_str}")
                logger.error(f"Current directory: {os.getcwd()}")
                raise FileNotFoundError("Configuration file not found")
                
            logger.info(f"Loading configuration from {config_path}")
            self.config = load_config(config_path)
            
            # Load spaCy model
            logger.info("Loading spaCy model")
            try:
                self.nlp = spacy.load('nb_core_news_md')
            except OSError:
                logger.warning("spaCy model not found, downloading...")
                os.system('python -m spacy download nb_core_news_md')
                self.nlp = spacy.load('nb_core_news_md')
            
            # Load tokenizer
            logger.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['model_name'])
            
            # Initialize transformer model
            logger.info("Initializing transformer model")
            self.model = TransformerModel(
                model_name=self.config['model']['model_name'], 
                dropout=self.config['model']['dropout'],
                num_labels=len(label_to_id)
            )
            
            # Load model weights
            possible_model_paths = [
                base_dir / 'src' / 'models' / 'nb-bert-base.pth',
                Path('src/models/nb-bert-base.pth')
            ]
            
            model_path = None
            for path in possible_model_paths:
                if path.exists():
                    model_path = path
                    break
                    
            if not model_path:
                paths_str = '\n'.join(str(p) for p in possible_model_paths)
                logger.error(f"Model file not found. Tried:\n{paths_str}")
                logger.error(f"Current directory: {os.getcwd()}")
                raise FileNotFoundError("Model file not found")
                
            logger.info(f"Loading model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            
            logger.info("Model initialization completed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            # Reset variables on failure
            self.nlp = None
            self.tokenizer = None
            self.model = None
            self.config = None
            return False
            
    def is_initialized(self):
        """Check if models are initialized"""
        return self.nlp is not None and self.tokenizer is not None and self.model is not None

# Create a global instance
model_manager = ModelManager()

def get_text(pdf_path):
    # Extracts text from a PDF 
    text = []
    logger.info(f"Opening PDF: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                logger.error("PDF has no pages")
                raise ValueError("PDF has no pages")
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    # Removes big unnecessary whitespaces
                    clean_text = re.sub(r'\s+', ' ', page_text).strip()
                    text.append(clean_text) 
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise ValueError(f"Failed to process PDF: {str(e)}")

    if not text:
        logger.error("No text extracted from PDF")
        raise ValueError("No text could be extracted from PDF")

    return ' '.join(text) # Returns text as one big string

def process_text_with_spacy(text, nlp):
    # Splits text into sentences and tokenizes them
    logger.info("Processing text with spaCy")
    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        tokens = []
        for token in sent:
            # Further split tokens (punctuations) to separate symbols
            split_tokens = re.findall(r"\w+(?:[-/.]\w+)*|[^\w\s]", token.text, re.UNICODE)
            for sub_token in split_tokens:
                tokens.append(sub_token)

        sentences.append(tokens) # Each sent is a list of tokens

    logger.info(f"Processed {len(sentences)} sentences")
    return sentences # Returns a list of sentences

def get_predictions(sent_tokens, model, tokenizer):
    logger.info("Starting model predictions")
    final_preds = []

    for sent in tqdm(sent_tokens):
        if not sent:
            continue

        encoding = tokenizer(
            sent,
            is_split_into_words=True,
            return_offsets_mapping=True,
            truncation=True,
            return_tensors='pt',
            padding='max_length',
            max_length=model_manager.config['data']['max_seq_len']
        )

        inputs = encoding['input_ids'].to(device)
        masks = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(inputs, masks, labels=None)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(inputs.squeeze(0))
        pred_labels = [id_to_label.get(pred, 'O') for pred in preds] 

        aligned_preds = []
        aligned_words = []
        word_idx = -1

        offsets = encoding['offset_mapping'].squeeze(0)

        # Align predictions with words
        for j, (token, offset) in enumerate(zip(tokens, offsets)):
            if offset[0] == 0 and offset[1] != 0: # New word
                word_idx += 1
                if word_idx < len(sent):
                    aligned_words.append(sent[word_idx])
                    aligned_preds.append(pred_labels[j])

        # Only keep 'B-FELT' and 'I-FELT'
        felt_tokens = [word for word, label in zip(aligned_words, aligned_preds) if label in ['B-FELT', 'I-FELT']]
        
        # Combine 'B-FELT' and 'I-FELT' tokens
        combined_tokens = []
        for i, token in enumerate(felt_tokens):
            if i == 0 or felt_tokens[i-1] == 'B-FELT':
                combined_tokens.append(token)
            else:
                combined_tokens[-1] += ' ' + token
        
        final_preds.extend(combined_tokens)

        # TODO: Make a regex function that show all zones in 'BKS1-BKS6'
            
    logger.info(f"Found {len(final_preds)} unique predictions")
    return list(dict.fromkeys(final_preds)) # Only return unique tokens

def get_sent_tokens(text: str, nlp) -> List[str]:
    """Get tokens for each sentence using spaCy"""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def initialize_models():
    """Initialize models using the model manager"""
    return model_manager.initialize()