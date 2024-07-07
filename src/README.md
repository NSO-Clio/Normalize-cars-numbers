# Инструкция по запуску

- скачайте веса модели https://drive.google.com/drive/folders/1NFzOVovRzFjN6iBO0q58cqdTAWhHdaTt?usp=drive_link
- поместите веса в данную директорию
- ВАЖНО! должен быть установлен docker
- ```sudo docker build -t app .```
- <your_path> - путь к вашей папке с изображениями <br>
  ```sudo docker run -v <your_path>:. app```
- результат работы модуля будет в <your_path>/result

