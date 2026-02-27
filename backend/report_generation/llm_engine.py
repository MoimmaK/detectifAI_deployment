"""
Local LLM Engine for Report Generation

Handles loading and inference with local GGUF models using llama-cpp-python.
Supports both Qwen2.5-3B-Instruct and Phi-3-mini as fallback.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import ReportConfig, LLMConfig

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Local LLM engine using llama-cpp-python.
    
    Provides deterministic, instruction-following text generation
    for structured report content.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize the LLM engine.
        
        Args:
            config: Report configuration (uses default if None)
        """
        self.config = config or ReportConfig()
        self.llm_config = self.config.llm
        self.model = None
        self._is_loaded = False
        self._model_type = None  # 'qwen' or 'phi'
        
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def download_model(self, use_alternative: bool = False) -> str:
        """
        Download model from HuggingFace Hub.
        
        Args:
            use_alternative: If True, download Phi-3 instead of Qwen
            
        Returns:
            Path to downloaded model
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models. "
                "Install with: pip install huggingface_hub"
            )
        
        if use_alternative:
            repo_id = self.llm_config.alt_hf_repo
            filename = self.llm_config.alt_hf_filename
            local_path = self.llm_config.alt_model_path
        else:
            repo_id = self.llm_config.hf_repo
            filename = self.llm_config.hf_filename
            local_path = self.llm_config.model_path
        
        logger.info(f"Downloading model from {repo_id}/{filename}...")
        
        # Download to models directory
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=self.llm_config.models_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Model downloaded to: {downloaded_path}")
        return downloaded_path
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the LLM model into memory.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successful, False otherwise
        """
        if self._is_loaded and not force_reload:
            logger.info("Model already loaded")
            return True
        
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. Install with:\n"
                "pip install llama-cpp-python\n"
                "For GPU support: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python"
            )
        
        # Try primary model first, then alternative
        model_path = self.llm_config.model_path
        if not os.path.exists(model_path):
            model_path = self.llm_config.alt_model_path
            self._model_type = 'phi'
        else:
            self._model_type = 'qwen'
        
        if not os.path.exists(model_path):
            logger.warning("No model found. Attempting to download...")
            try:
                model_path = self.download_model()
                self._model_type = 'qwen'
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                return False
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Model type: {self._model_type}")
        
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.llm_config.n_ctx,
                n_threads=self.llm_config.n_threads,
                n_gpu_layers=self.llm_config.n_gpu_layers,
                verbose=False
            )
            self._is_loaded = True
            logger.info("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._is_loaded = False
            return False
    
    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format prompt according to model's chat template.
        
        Args:
            system_prompt: System instructions
            user_prompt: User's request
            
        Returns:
            Formatted prompt string
        """
        if self._model_type == 'qwen':
            # Qwen2.5 chat format
            return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
        else:
            # Phi-3 chat format
            return f"""<|system|>
{system_prompt}<|end|>
<|user|>
{user_prompt}<|end|>
<|assistant|>
"""
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using the loaded LLM.
        
        Args:
            system_prompt: System instructions for the model
            user_prompt: The actual prompt/request
            max_tokens: Override max tokens (uses config default if None)
            temperature: Override temperature (uses config default if None)
            stop_sequences: Custom stop sequences
            
        Returns:
            Dict with 'text', 'tokens_used', 'finish_reason'
        """
        if not self._is_loaded:
            if not self.load_model():
                return {
                    'text': '',
                    'tokens_used': 0,
                    'finish_reason': 'error',
                    'error': 'Model not loaded'
                }
        
        # Format the prompt
        formatted_prompt = self._format_prompt(system_prompt, user_prompt)
        
        # Set parameters
        max_tokens = max_tokens or self.llm_config.max_tokens
        temperature = temperature or self.llm_config.temperature
        
        # Default stop sequences based on model type
        if stop_sequences is None:
            if self._model_type == 'qwen':
                stop_sequences = ["<|im_end|>", "<|im_start|>"]
            else:
                stop_sequences = ["<|end|>", "<|user|>"]
        
        logger.debug(f"Generating with max_tokens={max_tokens}, temp={temperature}")
        
        try:
            output = self.model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.llm_config.top_p,
                repeat_penalty=self.llm_config.repeat_penalty,
                stop=stop_sequences,
                echo=False
            )
            
            generated_text = output['choices'][0]['text'].strip()
            finish_reason = output['choices'][0].get('finish_reason', 'stop')
            tokens_used = output.get('usage', {}).get('total_tokens', 0)
            
            return {
                'text': generated_text,
                'tokens_used': tokens_used,
                'finish_reason': finish_reason,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                'text': '',
                'tokens_used': 0,
                'finish_reason': 'error',
                'error': str(e)
            }
    
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_format: str = 'markdown'
    ) -> Dict[str, Any]:
        """
        Generate structured output (Markdown or JSON).
        
        Args:
            system_prompt: System instructions
            user_prompt: User request
            output_format: 'markdown' or 'json'
            
        Returns:
            Dict with generated content
        """
        # Add format instructions to system prompt
        if output_format == 'json':
            format_instruction = "\nYou MUST respond with valid JSON only. No explanations outside the JSON."
        else:
            format_instruction = "\nYou MUST respond with properly formatted Markdown only."
        
        enhanced_system = system_prompt + format_instruction
        
        result = self.generate(enhanced_system, user_prompt)
        
        # Parse JSON if requested
        if output_format == 'json' and result['text']:
            import json
            try:
                # Try to extract JSON from the response
                text = result['text']
                # Find JSON boundaries
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = text[start:end]
                    result['parsed'] = json.loads(json_str)
                else:
                    result['parsed'] = None
                    result['parse_error'] = 'No JSON object found in response'
            except json.JSONDecodeError as e:
                result['parsed'] = None
                result['parse_error'] = str(e)
        
        return result
    
    def unload_model(self):
        """Unload model from memory."""
        if self.model:
            del self.model
            self.model = None
        self._is_loaded = False
        self._model_type = None
        logger.info("Model unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'is_loaded': self._is_loaded,
            'model_type': self._model_type,
            'model_path': self.llm_config.model_path if self._model_type == 'qwen' else self.llm_config.alt_model_path,
            'context_size': self.llm_config.n_ctx,
            'gpu_layers': self.llm_config.n_gpu_layers,
            'threads': self.llm_config.n_threads
        }


# Singleton instance for reuse
_engine_instance: Optional[LLMEngine] = None


def get_llm_engine(config: Optional[ReportConfig] = None) -> LLMEngine:
    """
    Get or create the LLM engine singleton.
    
    Args:
        config: Optional configuration override
        
    Returns:
        LLMEngine instance
    """
    global _engine_instance
    
    if _engine_instance is None or config is not None:
        _engine_instance = LLMEngine(config)
    
    return _engine_instance
