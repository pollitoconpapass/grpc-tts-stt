import io
import grpc
import torch
import torchaudio
import numpy as np
import soundfile as sf
from concurrent import futures
from transformers import VitsModel, AutoTokenizer
from transformers import Wav2Vec2ForCTC, AutoProcessor
from qhali_pb2 import TTSResponse, STTResponse
from qhali_pb2_grpc import add_SpeechRecognitionServicer_to_server, SpeechRecognitionServicer


class SpeechRecognitionService(SpeechRecognitionServicer):
    # === TEXT TO SPEECH PROCESS ===
    def tts(self, request, context):
        text = request.text
        region = request.region

        default="facebook/mms-tts-quz"
        hg_models = {
            "san-martin": "facebook/mms-tts-qvs",
            "cuzco": "facebook/mms-tts-quz",
            "huallaga": "facebook/mms-tts-qub",
            "lambayeque": "facebook/mms-tts-quf",
            "south-bolivia": "facebook/mms-tts-quh",
            "north-bolivia": "facebook/mms-tts-qul",
            "tena-lowland": "facebook/mms-tts-quw",
            "ayacucho": "facebook/mms-tts-quy",
            "cajamarca": "facebook/mms-tts-qvc",
            "eastern-apurimac": "facebook/mms-tts-qve",
            "huamelies": "facebook/mms-tts-qvh",
            "margos-lauricocha": "facebook/mms-tts-qvm",
            "north-junin": "facebook/mms-tts-qvn",
            "huaylas": "facebook/mms-tts-qwh",
            "panao": "facebook/mms-tts-qxh",
            "northern-conchucos": "facebook/mms-tts-qxn",
            "southern-conchucos": "facebook/mms-tts-qxo",
        }

        selected_quechua = hg_models.get(region, default)  
        model = VitsModel.from_pretrained(selected_quechua)
        tokenizer = AutoTokenizer.from_pretrained(selected_quechua)
        
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs).waveform.cpu().numpy()
        output = output / np.max(np.abs(output))

        rate = int(model.config.sampling_rate)
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, output.T, rate, subtype='PCM_16', format='WAV')
        
        return TTSResponse(audio=audio_bytes.getvalue())
    

    # === SPEECH TO TEXT PROCESS ===
    def stt(self, request, context):
        model_id = "facebook/mms-1b-all"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)

        audio_data, sampling_rate = torchaudio.load(io.BytesIO(request.audio))
        #sampling_rate = torchaudio.load(io.BytesIO(request.audio))[1]

        inputs = processor(audio_data.numpy(), sampling_rate=sampling_rate, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)

        return STTResponse(transcription=transcription)
    

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_SpeechRecognitionServicer_to_server(SpeechRecognitionService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
