"""
Speech-to-Text Module for Agentic RAG Customer Support System
===========================================================

This module handles voice input processing, speech recognition,
audio preprocessing, and transcription for customer support interactions.

Supports multiple STT engines:
- OpenAI Whisper API
- Google Speech Recognition
- Local Whisper model (optional)

Author: Research Team
Version: 1.0.0
Date: March 2024
"""

import os
import io
import wave
import tempfile
import logging
import datetime 
from typing import Optional, Dict, Any, List, Union, BinaryIO
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class STTProvider(Enum):
    """Supported Speech-to-Text providers."""
    OPENAI_WHISPER = "openai_whisper"
    GOOGLE_SPEECH = "google_speech"
    LOCAL_WHISPER = "local_whisper"
    AZURE_SPEECH = "azure_speech"


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""
    text: str
    confidence: float
    language: str
    duration_seconds: float
    word_count: int
    segments: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'language': self.language,
            'duration_seconds': self.duration_seconds,
            'word_count': self.word_count,
            'segments': self.segments,
            'metadata': self.metadata
        }


@dataclass
class AudioConfig:
    """Audio configuration parameters."""
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit
    chunk_size: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'sample_width': self.sample_width,
            'chunk_size': self.chunk_size
        }


class AudioPreprocessor:
    """Audio preprocessing utilities."""
    
    def __init__(self, config: AudioConfig = None):
        """
        Initialize audio preprocessor.
        
        Args:
            config: Audio configuration
        """
        self.config = config or AudioConfig()
    
    def validate_audio_format(self, audio_data: bytes) -> bool:
        """
        Validate audio data format.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if it's a valid WAV file
            with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                
                logger.debug(f"Audio format: {channels}ch, {sample_width}bytes, {sample_rate}Hz")
                return True
        except Exception as e:
            logger.warning(f"Invalid audio format: {e}")
            return False
    
    def convert_to_wav(self, audio_data: bytes, source_format: str = None) -> bytes:
        """
        Convert audio to standard WAV format.
        
        Args:
            audio_data: Raw audio bytes
            source_format: Source format (mp3, m4a, ogg, etc.)
            
        Returns:
            WAV format audio bytes
        """
        try:
            # Try using pydub if available
            try:
                from pydub import AudioSegment
                
                # Detect format if not specified
                if source_format is None:
                    source_format = self._detect_format(audio_data)
                
                # Load audio
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format=source_format)
                
                # Convert to target format
                audio = audio.set_frame_rate(self.config.sample_rate)
                audio = audio.set_channels(self.config.channels)
                audio = audio.set_sample_width(self.config.sample_width)
                
                # Export as WAV
                output = io.BytesIO()
                audio.export(output, format='wav')
                return output.getvalue()
                
            except ImportError:
                logger.warning("pydub not available, returning original audio")
                return audio_data
                
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return audio_data
    
    def _detect_format(self, audio_data: bytes) -> str:
        """Detect audio format from magic bytes."""
        if audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb'):
            return 'mp3'
        elif audio_data.startswith(b'fLaC'):
            return 'flac'
        elif audio_data.startswith(b'OggS'):
            return 'ogg'
        elif audio_data.startswith(b'RIFF'):
            return 'wav'
        elif audio_data.startswith(b'\x00\x00\x00\x20ftyp'):
            return 'm4a'
        else:
            return 'wav'  # Default
    
    def normalize_audio(self, audio_data: bytes) -> bytes:
        """
        Normalize audio volume.
        
        Args:
            audio_data: WAV audio bytes
            
        Returns:
            Normalized audio bytes
        """
        try:
            import numpy as np
            
            # Read WAV data
            with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                
                raw_data = wf.readframes(n_frames)
                
                # Convert to numpy array
                if sample_width == 2:
                    audio_array = np.frombuffer(raw_data, dtype=np.int16)
                elif sample_width == 4:
                    audio_array = np.frombuffer(raw_data, dtype=np.int32)
                else:
                    return audio_data
                
                # Normalize
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    audio_array = (audio_array / max_val * 32767).astype(np.int16)
                
                # Write back to WAV
                output = io.BytesIO()
                with wave.open(output, 'wb') as wo:
                    wo.setnchannels(n_channels)
                    wo.setsampwidth(sample_width)
                    wo.setframerate(sample_rate)
                    wo.writeframes(audio_array.tobytes())
                
                return output.getvalue()
                
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio_data
    
    def remove_silence(self, audio_data: bytes, 
                       silence_threshold: float = -40,
                       min_silence_len: int = 500) -> bytes:
        """
        Remove silence from audio.
        
        Args:
            audio_data: WAV audio bytes
            silence_threshold: Silence threshold in dB
            min_silence_len: Minimum silence length in ms
            
        Returns:
            Audio with silence removed
        """
        try:
            from pydub import AudioSegment
            from pydub.silence import detect_nonsilent
            
            # Load audio
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format='wav')
            
            # Detect non-silent ranges
            nonsilent_ranges = detect_nonsilent(
                audio, 
                min_silence_len=min_silence_len,
                silence_thresh=silence_threshold
            )
            
            if not nonsilent_ranges:
                return audio_data
            
            # Extract non-silent parts
            output_audio = AudioSegment.empty()
            for start, end in nonsilent_ranges:
                output_audio += audio[start:end]
            
            # Export
            output = io.BytesIO()
            output_audio.export(output, format='wav')
            return output.getvalue()
            
        except ImportError:
            logger.warning("pydub not available, skipping silence removal")
            return audio_data
        except Exception as e:
            logger.error(f"Error removing silence: {e}")
            return audio_data
    
    def get_audio_info(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Get audio file information.
        
        Args:
            audio_data: Audio bytes
            
        Returns:
            Audio information dictionary
        """
        try:
            with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / sample_rate
                
                return {
                    'channels': n_channels,
                    'sample_width': sample_width,
                    'sample_rate': sample_rate,
                    'frames': n_frames,
                    'duration_seconds': round(duration, 2),
                    'bitrate': sample_rate * sample_width * n_channels * 8
                }
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {}


class OpenAIWhisperSTT:
    """OpenAI Whisper API speech-to-text implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI Whisper STT.
        
        Args:
            api_key: OpenAI API key (or from environment)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
        
        self.api_url = "https://api.openai.com/v1/audio/transcriptions"
        self.supported_formats = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 
                                  'oga', 'ogg', 'wav', 'webm']
    
    def transcribe(self, audio_data: bytes, 
                   language: Optional[str] = None,
                   prompt: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe audio using OpenAI Whisper API.
        
        Args:
            audio_data: Audio bytes
            language: Optional language code (e.g., 'en', 'vi')
            prompt: Optional prompt to guide transcription
            
        Returns:
            Transcription result
        """
        try:
            import requests
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            try:
                with open(tmp_path, 'rb') as audio_file:
                    files = {
                        'file': ('audio.wav', audio_file, 'audio/wav'),
                        'model': (None, 'whisper-1')
                    }
                    
                    if language:
                        files['language'] = (None, language)
                    if prompt:
                        files['prompt'] = (None, prompt)
                    
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        text = result.get('text', '')
                        
                        # Get audio info for metadata
                        preprocessor = AudioPreprocessor()
                        audio_info = preprocessor.get_audio_info(audio_data)
                        
                        return TranscriptionResult(
                            text=text,
                            confidence=0.95,  # Whisper doesn't provide confidence
                            language=result.get('language', language or 'unknown'),
                            duration_seconds=audio_info.get('duration_seconds', 0),
                            word_count=len(text.split()),
                            segments=result.get('segments', []),
                            metadata={
                                'provider': 'openai_whisper',
                                'model': 'whisper-1',
                                'audio_info': audio_info
                            }
                        )
                    else:
                        logger.error(f"Whisper API error: {response.status_code} - {response.text}")
                        raise Exception(f"API error: {response.status_code}")
                        
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {e}")
            raise


class GoogleSpeechSTT:
    """Google Speech Recognition implementation."""
    
    def __init__(self):
        """Initialize Google Speech STT."""
        self.recognizer = None
        self._init_recognizer()
    
    def _init_recognizer(self):
        """Initialize speech recognizer."""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            logger.info("Google Speech recognizer initialized")
        except ImportError:
            logger.warning("speech_recognition not installed. Run: pip install SpeechRecognition")
    
    def transcribe(self, audio_data: bytes,
                   language: str = "en-US") -> TranscriptionResult:
        """
        Transcribe audio using Google Speech Recognition.
        
        Args:
            audio_data: Audio bytes (WAV format)
            language: Language code
            
        Returns:
            Transcription result
        """
        if not self.recognizer:
            raise RuntimeError("Speech recognizer not initialized")
        
        try:
            import speech_recognition as sr
            
            # Load audio
            with io.BytesIO(audio_data) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)
            
            # Get audio info
            preprocessor = AudioPreprocessor()
            audio_info = preprocessor.get_audio_info(audio_data)
            
            # Transcribe
            text = self.recognizer.recognize_google(
                audio, 
                language=language,
                show_all=False
            )
            
            return TranscriptionResult(
                text=text,
                confidence=0.85,  # Google doesn't provide exact confidence
                language=language,
                duration_seconds=audio_info.get('duration_seconds', 0),
                word_count=len(text.split()),
                segments=[],
                metadata={
                    'provider': 'google_speech',
                    'audio_info': audio_info
                }
            )
            
        except sr.UnknownValueError:
            logger.warning("Google Speech could not understand audio")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=language,
                duration_seconds=0,
                word_count=0,
                segments=[],
                metadata={'error': 'Could not understand audio'}
            )
        except sr.RequestError as e:
            logger.error(f"Google Speech API error: {e}")
            raise


class LocalWhisperSTT:
    """Local Whisper model implementation."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize local Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            import whisper
            self.model = whisper.load_model(self.model_size)
            logger.info(f"Loaded Whisper model: {self.model_size}")
        except ImportError:
            logger.warning("openai-whisper not installed. Run: pip install openai-whisper")
    
    def transcribe(self, audio_data: bytes,
                   language: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe audio using local Whisper model.
        
        Args:
            audio_data: Audio bytes
            language: Optional language code
            
        Returns:
            Transcription result
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            try:
                # Transcribe
                result = self.model.transcribe(
                    tmp_path,
                    language=language,
                    fp16=False
                )
                
                text = result.get('text', '').strip()
                
                # Get audio info
                preprocessor = AudioPreprocessor()
                audio_info = preprocessor.get_audio_info(audio_data)
                
                return TranscriptionResult(
                    text=text,
                    confidence=result.get('confidence', 0.9),
                    language=result.get('language', language or 'unknown'),
                    duration_seconds=audio_info.get('duration_seconds', 0),
                    word_count=len(text.split()),
                    segments=result.get('segments', []),
                    metadata={
                        'provider': 'local_whisper',
                        'model_size': self.model_size,
                        'audio_info': audio_info
                    }
                )
                
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Error in local Whisper transcription: {e}")
            raise


class SpeechToTextManager:
    """Manager for speech-to-text operations."""
    
    def __init__(self, provider: STTProvider = STTProvider.OPENAI_WHISPER,
                 api_key: Optional[str] = None):
        """
        Initialize STT manager.
        
        Args:
            provider: STT provider to use
            api_key: API key for cloud providers
        """
        self.provider = provider
        self.preprocessor = AudioPreprocessor()
        self.stt_engine = None
        
        # Initialize engine
        self._init_engine(api_key)
    
    def _init_engine(self, api_key: Optional[str]):
        """Initialize STT engine."""
        try:
            if self.provider == STTProvider.OPENAI_WHISPER:
                self.stt_engine = OpenAIWhisperSTT(api_key)
            elif self.provider == STTProvider.GOOGLE_SPEECH:
                self.stt_engine = GoogleSpeechSTT()
            elif self.provider == STTProvider.LOCAL_WHISPER:
                self.stt_engine = LocalWhisperSTT()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
            logger.info(f"Initialized STT engine: {self.provider.value}")
            
        except Exception as e:
            logger.error(f"Error initializing STT engine: {e}")
            raise
    
    def transcribe(self, audio_input: Union[str, bytes],
                   language: Optional[str] = None,
                   preprocess: bool = True) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio_input: Audio file path or bytes
            language: Optional language code
            preprocess: Whether to preprocess audio
            
        Returns:
            Transcription result
        """
        # Load audio if path provided
        if isinstance(audio_input, str):
            with open(audio_input, 'rb') as f:
                audio_data = f.read()
        else:
            audio_data = audio_input
        
        # Preprocess if requested
        if preprocess:
            audio_data = self._preprocess_audio(audio_data)
        
        # Transcribe
        result = self.stt_engine.transcribe(audio_data, language=language)
        
        logger.info(f"Transcribed {result.duration_seconds}s audio to {result.word_count} words")
        return result
    
    def _preprocess_audio(self, audio_data: bytes) -> bytes:
        """
        Preprocess audio for transcription.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Preprocessed audio bytes
        """
        # Convert to WAV if needed
        audio_data = self.preprocessor.convert_to_wav(audio_data)
        
        # Normalize volume
        audio_data = self.preprocessor.normalize_audio(audio_data)
        
        # Remove silence
        audio_data = self.preprocessor.remove_silence(audio_data)
        
        return audio_data
    
    def transcribe_batch(self, audio_files: List[str],
                         language: Optional[str] = None) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            language: Optional language code
            
        Returns:
            List of transcription results
        """
        results = []
        for file_path in audio_files:
            try:
                result = self.transcribe(file_path, language=language)
                results.append(result)
            except Exception as e:
                logger.error(f"Error transcribing {file_path}: {e}")
                results.append(TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language=language or 'unknown',
                    duration_seconds=0,
                    word_count=0,
                    segments=[],
                    metadata={'error': str(e), 'file': file_path}
                ))
        
        return results
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of language codes
        """
        # Common languages supported by most STT engines
        return [
            'en',  # English
            'es',  # Spanish
            'fr',  # French
            'de',  # German
            'it',  # Italian
            'pt',  # Portuguese
            'nl',  # Dutch
            'ja',  # Japanese
            'ko',  # Korean
            'zh',  # Chinese
            'vi',  # Vietnamese
            'ru',  # Russian
            'ar',  # Arabic
            'hi',  # Hindi
        ]


class VoiceCommandProcessor:
    """Process voice commands for customer support."""
    
    def __init__(self, stt_manager: SpeechToTextManager):
        """
        Initialize voice command processor.
        
        Args:
            stt_manager: STT manager instance
        """
        self.stt_manager = stt_manager
        
        # Command patterns
        self.command_patterns = {
            'check_order': [
                'check my order', 'where is my order', 'order status',
                'track my order', 'order number', 'when will my order arrive'
            ],
            'billing_issue': [
                'billing problem', 'charge issue', 'wrong charge',
                'refund', 'payment problem', 'billing question'
            ],
            'technical_support': [
                'technical issue', 'not working', 'error message',
                'bug', 'crash', 'login problem', 'can\'t access'
            ],
            'account_help': [
                'account issue', 'password reset', 'can\'t login',
                'update account', 'close account', 'delete account'
            ],
            'product_info': [
                'product information', 'tell me about', 'features',
                'how does it work', 'product details', 'specifications'
            ],
            'speak_agent': [
                'speak to agent', 'talk to human', 'customer service',
                'representative', 'transfer to agent', 'need help from person'
            ]
        }
    
    def process_voice_query(self, audio_data: bytes,
                           customer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a voice query from customer.
        
        Args:
            audio_data: Audio bytes
            customer_id: Optional customer identifier
            
        Returns:
            Processed query with intent classification
        """
        # Transcribe audio
        transcription = self.stt_manager.transcribe(audio_data)
        
        # Classify intent
        intent = self._classify_intent(transcription.text)
        
        # Extract entities
        entities = self._extract_entities(transcription.text)
        
        # Determine urgency
        urgency = self._determine_urgency(transcription.text)
        
        return {
            'transcription': transcription.to_dict(),
            'intent': intent,
            'entities': entities,
            'urgency': urgency,
            'customer_id': customer_id,
            'timestamp': datetime.now().isoformat(),
            'recommended_action': self._get_recommended_action(intent, urgency)
        }
    
    def _classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify intent from transcribed text.
        
        Args:
            text: Transcribed text
            
        Returns:
            Intent classification
        """
        text_lower = text.lower()
        
        scores = {}
        for intent, patterns in self.command_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            scores[intent] = score
        
        # Get highest scoring intent
        if max(scores.values()) > 0:
            detected_intent = max(scores, key=scores.get)
            confidence = min(scores[detected_intent] / 2, 1.0)
        else:
            detected_intent = 'general_inquiry'
            confidence = 0.5
        
        return {
            'primary_intent': detected_intent,
            'confidence': confidence,
            'all_scores': scores
        }
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Extracted entities
        """
        import re
        
        entities = {
            'order_numbers': re.findall(r'order\s*#?\s*(\d+)', text, re.IGNORECASE),
            'email_addresses': re.findall(r'\S+@\S+', text),
            'phone_numbers': re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text),
            'amounts': re.findall(r'\$?\d+\.?\d*', text),
            'dates': re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        }
        
        return entities
    
    def _determine_urgency(self, text: str) -> str:
        """
        Determine urgency level from text.
        
        Args:
            text: Input text
            
        Returns:
            Urgency level (low, medium, high, critical)
        """
        text_lower = text.lower()
        
        critical_keywords = ['emergency', 'critical', 'urgent', 'immediately', 'asap', 'right now']
        high_keywords = ['important', 'serious', 'frustrated', 'angry', 'disappointed', 'terrible']
        
        if any(word in text_lower for word in critical_keywords):
            return 'critical'
        elif any(word in text_lower for word in high_keywords):
            return 'high'
        elif any(word in text_lower for word in ['problem', 'issue', 'help']):
            return 'medium'
        else:
            return 'low'
    
    def _get_recommended_action(self, intent: Dict, urgency: str) -> str:
        """
        Get recommended action based on intent and urgency.
        
        Args:
            intent: Intent classification
            urgency: Urgency level
            
        Returns:
            Recommended action
        """
        primary_intent = intent.get('primary_intent', '')
        
        if urgency in ['critical', 'high']:
            return 'escalate_to_human'
        elif primary_intent == 'speak_agent':
            return 'transfer_to_agent'
        elif primary_intent in ['check_order', 'billing_issue']:
            return 'query_crm_and_respond'
        elif primary_intent == 'technical_support':
            return 'search_knowledge_base'
        elif primary_intent == 'product_info':
            return 'provide_product_info'
        else:
            return 'general_response'


# Demo usage
if __name__ == "__main__":
    print("Speech-to-Text Module Demo")
    print("=" * 50)
    
    # Check available providers
    print("\nAvailable STT Providers:")
    for provider in STTProvider:
        print(f"  - {provider.value}")
    
    # Initialize manager (will use mock if no API key)
    try:
        manager = SpeechToTextManager(
            provider=STTProvider.GOOGLE_SPEECH
        )
        
        print("\nSupported Languages:")
        languages = manager.get_supported_languages()
        print(f"  {', '.join(languages[:10])}... ({len(languages)} total)")
        
    except Exception as e:
        print(f"\nNote: {e}")
        print("Install dependencies: pip install SpeechRecognition pydub")
