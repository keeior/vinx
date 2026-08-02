# stt_service.py
import modal

app = modal.App("shona-stt-service")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "torchaudio",
    "transformers",
    "huggingface_hub",
    "fastapi[standard]",
)


@app.cls(
    image=image,
    gpu="T4",
    timeout=300,
    scaledown_window=120,
)
class ShonaSTT:
    @modal.enter()
    def load_model(self):
        from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC
        import torch

        self.processor = Wav2Vec2BertProcessor.from_pretrained(
            "badrex/w2v-bert-2.0-shona-asr"
        )
        self.model = Wav2Vec2BertForCTC.from_pretrained(
            "badrex/w2v-bert-2.0-shona-asr"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        import torch
        import torchaudio
        import io

        audio_input, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        inputs = self.processor(
            audio_input.squeeze(), sampling_rate=sample_rate, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def transcribe_endpoint(audio_bytes: bytes):
    stt = ShonaSTT()
    result = stt.transcribe.remote(audio_bytes)
    return {"transcription": result}        import torch
        import torchaudio
        import io

        audio_input, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        inputs = self.processor(
            audio_input.squeeze(), sampling_rate=sample_rate, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def transcribe_endpoint(audio_bytes: bytes):
    stt = ShonaSTT()
    result = stt.transcribe.remote(audio_bytes)
    return {"transcription": result}
