from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torchaudio
import soundfile as sf
from xcodec2.modeling_xcodec2 import XCodec2Model
from google.cloud import storage
import os
from dotenv import load_dotenv
from huggingface_hub import HfFolder;

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "trainer/text-to-speech-451918-8309307a12ed.json"
HfFolder.save_token(os.environ["HF_TOKEN"])

bucket_uri = os.getenv("BUCKET_URI") 
bucket_name = bucket_uri.replace("gs://", "")
model_dir = "llasa_finetuned_model"

os.makedirs(model_dir, exist_ok=True)

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

blobs = list(storage_client.list_blobs(bucket_name))
for blob in blobs:
    if "/" not in blob.name or blob.name.startswith("/checkpoint-"):
        continue
    blob_name = blob.name.replace("/", "")
    file_path = os.path.join(model_dir, blob_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Downloading {blob_name} to {file_path}...")
    blob.download_to_filename(file_path)

llasa_1b ='llasa_finetuned_model'

tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
model = AutoModelForCausalLM.from_pretrained(llasa_1b)
model.eval() 
model.to('cuda')
 
model_path = "HKUSTAudio/xcodec2"  
 
Codec_model = XCodec2Model.from_pretrained(model_path)
Codec_model.eval().cuda()

sample_audio_path = "krzysiek.wav"
waveform, sample_rate = torchaudio.load(sample_audio_path)
if len(waveform[0])/sample_rate > 15:
  print("Trimming audio to first 15secs.")
  waveform = waveform[:, :sample_rate*15]
# Check if the audio is stereo (i.e., has more than one channel)
if waveform.size(0) > 1:
  # Convert stereo to mono by averaging the channels
  waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
else:
  # If already mono, just use the original waveform
  waveform_mono = waveform
prompt_wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_mono)

# prompt_text = "Hi! I'm Amelia, a super high quality English voice. I love to read. Seriously, I'm a total bookworm. So what are you waiting for? Get me reading!"
# prompt_text ="Uczenie maszynowe to dynamicznie rozwijająca się dziedzina informatyki. Modele sztucznej inteligencji potrafią analizować dane, rozpoznawać wzorce i podejmować decyzje."
# prompt_text ="The box sat on the desk next to the computer. It had arrived earlier in the day and business had interrupted her opening it earlier."
prompt_text = "Nie wiem, co powiedzieć. Będzie słonecznie? Czy deszczowo? Pogoda jest zupełnie nieprzewidywalna. To jest całkiem zabawne."
# target_text = 'Współczesne algorytmy uczą się na podstawie ogromnych zbiorów danych'
# target_text = "She didn't who had sent it and briefly wondered who it might have been. As she began to unwrap it, she had no idea that opening it would completely change her life."
target_text = "A jednak zawsze znajdzie się ktoś, kto z pełnym przekonaniem przewidzi ulewny deszcz – i zapomni zabrać parasol."
# target_text = 'Współczesne algorytmy uczą się na podstawie ogromnych zbiorów danych, co pozwala im na coraz bardziej precyzyjne działanie'
input_text = prompt_text + ' ' + target_text

def ids_to_speech_tokens(speech_ids):
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
 
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

#TTS start!
with torch.no_grad():
    # Encode the prompt wav
    vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
    print("Prompt Vq Code Shape:", vq_code_prompt.shape )   

    vq_code_prompt = vq_code_prompt[0,0,:]
    # Convert int 12345 to token <|s_12345|>
    speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

    # Tokenize the text and the speech prefix
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
    ]

    input_ids = tokenizer.apply_chat_template(
        chat, 
        tokenize=True, 
        return_tensors='pt', 
        continue_final_message=True
    )
    input_ids = input_ids.to('cuda')
    speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

    # Generate the speech autoregressively
    outputs = model.generate(
        input_ids,
        max_length=2048,  # We trained our model with a max length of 2048
        eos_token_id= speech_end_id ,
        do_sample=True,
        top_p=1,           
        temperature=0.8,
    )
    # Extract the speech tokens
    generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]

    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   

    # Convert  token <|s_23456|> to int 23456 
    speech_tokens = extract_speech_ids(speech_tokens)

    speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

    # Decode the speech tokens to speech waveform
    gen_wav = Codec_model.decode_code(speech_tokens) 

    # if only need the generated part
    # gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]

sf.write("speech_multilingual.wav", gen_wav[0, 0, :].cpu().numpy(), 16000)
