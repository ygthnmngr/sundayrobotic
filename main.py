import librosa #ses dosyalarının nitelikleri için kullanılan kütüphanedir.
import torch #machineleraning kütüphanesi, doğal dil işleme, bilgisayarla görme gibi fonksiyonları var.
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#import speech_recognition as sr

#r = sr.Recognizer()



#with sr.Microphone() as source:
   # auido = r.listen(source)
  #  with open('speech.wav','wb') as f:
       # f.write(auido.get_wav_data())

#Modeli ve tokenizer i yükledik. Tokenizer metin gövdesini küçük satırlara hatta kelimelere bölmek için kullanılır.Hatta ingilizce olmayan diller için bile geçerli.
tokenizer = Wav2Vec2Processor.from_pretrained("cahya/wav2vec2-base-turkish")
model = Wav2Vec2ForCTC.from_pretrained("cahya/wav2vec2-base-turkish")


#ses dosyasını librosa kütüphanesi ile yüklüyoruz ve sample rate'i 16000 ayarlıyoruz.16000 ayarlama sebebimiz wav2vec 2.0 kütüphanesinin örnekleme frekansının 16 kHz olmasıdır.
speech, rate = librosa.load("sunday.wav", sr=16000)

#  return_tensors = 'pt' --- tamsayılar yerine tensörleri döndürüyoruz.pytorch'a dönebilmesi için 'pt' kullanıyoruz.
# tensör dediğimiz şey ise vektörlerin ve matrislerin isteğe bağlı sayıda boyuta genelleştirilmesidir.
# en sondada döndürülen diziyi doldurmak için gruptaki en uzun diziye dolgu yapılır.
input_values = tokenizer(speech, return_tensors = 'pt', sampling_rate = 16000, padding="longest"  ).input_values

#sinir ağının son katmanından çıkan ham tahminler.
logits = model(input_values).logits
# giriş tensöründeki tüm öğelerin maksimum değerinin endekslerini döndürür.
predicted_ids = torch.argmax(logits, dim =-1)
# metin oluşturmak için sesin kodunu çözeriz.
transcriptions = tokenizer.decode(predicted_ids[0])
print(transcriptions)
