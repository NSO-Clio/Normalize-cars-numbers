# Инструкция по запуску

- скачайте веса модели https://drive.google.com/drive/folders/1NFzOVovRzFjN6iBO0q58cqdTAWhHdaTt?usp=drive_link
- поместите веса в данную директорию
- ВАЖНО! должен быть установлен docker

```
sudo docker build -t app .
```


- ```<your_path>``` - путь к вашей папке с изображениями

```
sudo docker run -v <your_path>:. app
```
  
- результат работы модуля будет в ```<your_path>/result```

# Если вы планируете запускать через python (не используя докер)

- сначала скачиваем все бибиблиотеки
```
pip install -r requirements.txt
```

- после этого можно запустить проект

your_path - папка где лежат ваши изображения 

output_path - в какую папку надо сохранить

```
python main.py <your_path> <output_path>
```
