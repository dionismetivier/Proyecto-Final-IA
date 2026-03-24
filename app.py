import torch
from torch import nn
import gradio as gr
from torchvision import models, transforms
from gtts import gTTS
import io
import requests

# Mis Datos:
# Dionis Metivier Santana
# 24-EISN-2-022
# Proyecto Final: Calculador de Calorias de Comidas

LOGMEAL_TOKEN = "c0d39bd02a471568df2035cd44156c6494459265"
EDAMAM_ID = "c1b7ea6f"
EDAMAM_KEY = "436e2d11559c0e38c012687272750201"

#1. Cerebro de la aplicacion para recibir datos de las comidas y saber como procesarlas para darnos el resultado final
class ModeloNutricion (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
    def forward(self, x):
        return self.fc(x)
    
#2. Configuracion y guardado del progreso que se realice de la IA 
Modelo = ModeloNutricion()
Modelo.eval()

def detectar_comida(img_pil):
     url = 'https://api.logmeal.es/v2/image/recognition/complete'
     headers = {'Authorization': f'Bearer {LOGMEAL_TOKEN}'}

     buf = io.BytesIO()
     img_pil.save(buf, format= 'JPEG')
     files = {'image': buf.getvalue()}

     try:
          response = requests.post(url, files=files, headers=headers)
          data = response.json()
          nombre = data['recognition_results'][0]['name']
          return nombre
     except:
          return "food"
     
def obtener_calorias(nombre):
     url = "https://api.edamam.com/api/food-database/v2/parser"
     params = {"app_id": EDAMAM_ID, "app_key": EDAMAM_KEY, "ingr": nombre}
     try:
          res =requests.get(url, params=params).json()
          return float(res['hints'][0]['food']['nutrients']['ENERC_KCAL'])
     except:
          return 150.0

def analisis_plato(img_pil, gramos):
    if img_pil is None: 
            return "Captura Imagen", None
        
    nombre_alimento = detectar_comida(img_pil)
    calorias = obtener_calorias(nombre_alimento)
    
    with torch.no_grad():
        base = torch.tensor([[float(calorias)]])
        prediccion = Modelo(base)
        resultado = (abs(prediccion.item()) / 100.0) * float(gramos)
        calorias = round(resultado, 2)

    Voz_resultado = f"He identificado {nombre_alimento}. Son aproximadamente {calorias} calorias en {gramos} gramos."
    tts = gTTS(text=Voz_resultado, lang= 'es')
    tts.save("resultado_final.mp3")

    return Voz_resultado, "resultado_final.mp3"
    
with gr.Blocks(theme= 'Soft') as interfaz:
    gr.Markdown("# Calculador de Calorias")
    gr.Markdown("Sube un archivo desde tu pc o utiliza la Webcam")

    with gr.Row():
            with gr.Column():
                img_input = gr.Image(sources=["webcam", "upload"], type="pil", label="Camara en Vivo")
                peso_input = gr.Number(label="Gramos de la porcion", value=100)
                btn = gr.Button("Analisis de Plato", variant="primary")

            with gr.Column():
                txt_output = gr.Textbox(label="Analisis")
                aud_output = gr.Audio(label="Audio", autoplay=True)
    btn.click(analisis_plato, [img_input, peso_input], [txt_output, aud_output])

if __name__ == "__main__":
    interfaz.launch()