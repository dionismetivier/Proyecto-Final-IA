import torch
from torch import nn
import gradio as gr
import requests
import io
from PIL import Image
from gtts import gTTS
import os
from safetensors.torch import save_model, load_model

# Mis Datos:
# Dionis Metivier Santana
# 24-EISN-2-022
# Proyecto Final: Calculador de Calorias de Comidas

#1. Cerebro de la aplicacion para recibir datos de las comidas y saber como procesarlas para darnos el resultado final
class ModeloNutricion (nn.Module):
    def __init__(self):
        super(ModeloNutricion).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)
    
#2. Configuracion y guardado del progreso que se realice de la IA 
Modelo = ModeloNutricion()
MODEL_PATH = "Modelo_Nutricion-safetensors"

#3. Logica de seguridad ya que nos asegura que el trabajo no se pierda por algun fallo.
if not os.path.exists(MODEL_PATH):
    save_model(Modelo, MODEL_PATH)

Token_HuggingFace = "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"
EDAMAM_ID = "c1b7ea6f"
EDAMAM_KEY = "436e2d11559c0e38c012687272750201"

def identificar_comidas(img_pil):
    url= "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
    headers = {"Autorizacion": f"Bearer {Token_HuggingFace}"}
    buf = io.BytesIO()
    img_pil.save(buf, format= 'JPEG')

    response = requests.post(url, headers=headers, data=buf.getvalue())
    if response.status_code == 200:
       return response.json() [0] ['label']
    return "Comida"

def obtener_Calorias(nombre):
    url = f"https://api.edamam.com/api/food-database/v2/parser"
    params = {"ID_Aplicacion": EDAMAM_ID, "Key_Aplicacion": EDAMAM_KEY, "ingredientes": nombre}
    res = requests.get(url, params=params).json()
    try:
        return res['hints'] [0] ['Comida'] ['Nutrientes'] ['Energias_Kilocalorias']
    except:
        return 120.0