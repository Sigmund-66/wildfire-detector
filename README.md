# Modelo de detecção de fogo e fumaça (Wildfire Detect)
Modelo de detecção de incêndios em florestas 



# <img width="48" height="48" alt="20733653371681985287-48" src="https://github.com/user-attachments/assets/766a5cfe-16f5-481e-b07c-a1f8e16145b5" />  Sobre


# Objetivo
Monitorar florestas para detectar incêndios ou focos para adotar estratégias de combate e prevenção.

# Como usar

# Resultados
Os resultados foram obtidos fazendo dois treinameinos distintos, sendo o segundo realizado com 
os pesos do primeiro utilizando a técnica do fine-tuning (treino contínuo). O segundo se saiu melhor do primeiro visto que além de utilizar os pesos do anterior também contou com um dataset mais robusto.

## Treinamento 1
**Configuração da máquina:**  
**Dataset utilizado:**  
🔥 **fire-smoke data Dataset** — por *me*, publicado no [Roboflow Universe](https://universe.roboflow.com/me-p4nto/fire-smoke-data), junho de 2025. Visitado em 18 de julho de 2025.  
**Modelo YOLOv11:** `yolo11s.pt`  
**Código do treino:**  

```
model = YOLO("yolo11s.pt")

resultados = model.train(
    data="/content/drive/MyDrive/fire-smoke_data.v3i.yolov11/data.yaml",\
    epochs=120,
    patience=40,
    imgsz=640,
    pretrained=True,
    batch=8,
    hsv_v=0.5,
    hsv_h=0.3,
    degrees=0.3,
    mosaic=0.5,
    mixup=0.0,
)
```
### Validação - Métricas

### Gráfico F1 confidence curve
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/26528de3-fdc8-491e-bd4c-d58fb0865de1" />

### Gráfico curve Precision x Recall
<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/9ce5c943-a25c-4b74-8374-71f89960c56e" />

### Matriz confusão normalizada
<img width="2250" height="1500" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/e113552a-4ba9-4c1f-9fee-6ab9ed38fadc" />

## Treinamento 2



# Conclusão


