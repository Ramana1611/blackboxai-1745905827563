import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BertTokenizer, BertModel, pipeline
from gtts import gTTS
import os
import platform
import tempfile

class RealTimeSpeechProcessor:
    def __init__(self):
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Speech Recognition (Wav2Vec 2.0)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        
        # Text Correction (BERT + LLM)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.llm = pipeline("text-generation", 
                            model="gpt2",
                            device=0 if self.device == "cuda" else -1)
        
        # Audio Config
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

    def list_input_devices(self):
        """List available audio input devices."""
        p = pyaudio.PyAudio()
        print("\nAvailable audio input devices:")
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                print(f"{i}: {dev['name']}")
        p.terminate()

    def capture_audio(self, duration=5, input_device_index=None):
        """Capture audio with PyAudio with error handling."""
        p = pyaudio.PyAudio()
        stream = None
        frames = []
        
        try:
            # Open stream
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                input_device_index=input_device_index
            )
            
            print(f"\nRecording for {duration} seconds...")
            for _ in range(0, int(self.RATE / self.CHUNK * duration)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            
            audio = np.concatenate(frames)
            return audio, self.RATE
            
        except Exception as e:
            print(f"Audio error: {str(e)}")
            return None, self.RATE
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

    def speech_to_text(self, audio):
        """Convert speech to text with Wav2Vec2."""
        if audio is None or len(audio) == 0:
            return ""
        
        inputs = self.processor(audio, sampling_rate=self.RATE, return_tensors="pt", padding=True).input_values.to(self.device)
        with torch.no_grad():
            logits = self.asr_model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            return transcription.lower().strip()

    def correct_text(self, text):
        """Correct text using LLM with refined prompt."""
        if not text.strip():
            return ""
        
        prompt = f"Please correct the following transcription for grammar and clarity:\n\"{text}\"\nCorrected version:"
        try:
            corrected = self.llm(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            # Extract the corrected part after the prompt
            corrected_text = corrected.split("Corrected version:")[-1].strip()
            # Sometimes the model may continue with extra text, so split by newline or period
            corrected_text = corrected_text.split('\n')[0].strip()
            return corrected_text
        except Exception as e:
            print(f"LLM error: {str(e)}")
            return text

    def text_to_speech(self, text):
        """Convert text to speech and play it cross-platform."""
        if not text.strip():
            return
        
        try:
            tts = gTTS(text=text, lang='en')
            # Use a temporary file to save the audio
            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
                tts.save(fp.name)
                if platform.system() == "Windows":
                    os.system(f'start {fp.name}')
                elif platform.system() == "Darwin":  # macOS
                    os.system(f'afplay {fp.name}')
                else:  # Linux and others
                    # Try to use mpg123 or play command
                    if os.system(f"mpg123 {fp.name}") != 0:
                        os.system(f"play {fp.name}")
        except Exception as e:
            print(f"TTS error: {str(e)}")

    def process_realtime(self):
        """Main processing loop for real-time speech processing."""
        print("\nStarting real-time processing. Press Ctrl+C to stop...")
        self.list_input_devices()
        try:
            while True:
                audio, sr = self.capture_audio(duration=3)
                if audio is None:
                    continue
                
                raw_text = self.speech_to_text(audio)
                print(f"\nRaw transcription: {raw_text}")
                
                corrected_text = self.correct_text(raw_text)
                print(f"Corrected transcription: {corrected_text}")
                
                self.text_to_speech(corrected_text)
                
        except KeyboardInterrupt:
            print("\nProcessing stopped by user.")
        except Exception as e:
            print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    processor = RealTimeSpeechProcessor()
    processor.process_realtime()
