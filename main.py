# -*- coding: utf-8 -*-
import re
import os
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import string
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
import json

def preprocess_text(text):
    # Удаляем знаки пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Приводим к нижнему регистру
    text = text.lower()

    return text

def extract_text_from_pdfs(pdf_paths):
    all_texts = []

    for pdf_path in pdf_paths:
        print(f"Обрабатываем файл: {pdf_path}")
        text = ""
        images = convert_from_path(pdf_path, poppler_path=r'C:/Users/User/Desktop/poppler-23.11.0/Library/bin')  # Укажите путь к Poppler

        for image_index, image in enumerate(images):
            text += pytesseract.image_to_string(image, lang='rus')  # Изменено на 'rus'

            # Применяем предварительную обработку
            text = preprocess_text(text)

            all_texts.append(text)

        return all_texts

# Пример использования
pdf_files_directory = "C:/Users/User/Desktop/PDF файлы для компиляции"
pdf_files = [os.path.join(pdf_files_directory, file) for file in os.listdir(pdf_files_directory) if file.endswith(".pdf")]

all_texts = extract_text_from_pdfs(pdf_files)



# Выводим текст или его часть для проверки
for i, text_content in enumerate(all_texts):
    print(f"Текст из файла {i + 1}:")
    print(text_content[:500])  # Вывести первые 500 символов текста
    print("=" * 50)


# Код для поиска и вывода информации
def extract_fio(text):
    # Пример регулярного выражения для извлечения ФИО
    pattern = re.compile(r'\b(?:ФИО|Имя)\s*:\s*([^\n\r]+)', re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1) if match else None

# Пример использования для первого текста
fio = extract_fio(all_texts[0])
print(f"Извлеченное ФИО: {fio}")

def extract_inn_information(text):
    # Пример регулярного выражения для извлечения ИНН
    inn_pattern = re.compile(r'\b(?:ИНН)\s*:\s*([^\n\r]+)', re.IGNORECASE)
    inn_match = inn_pattern.search(text)
    inn = inn_match.group(1) if inn_match else None

    # Пример использования
    return {
        "inn": inn
    }

# Пример использования для первого текста
inn_info = extract_inn_information(all_texts[0])
print(f"Извлеченная информация по ИНН: {inn_info}")

def extract_address_information(text):
    # Пример регулярного выражения для извлечения адреса проживания
    address_pattern = re.compile(r'\b(?:Адрес проживания)\s*:\s*([^\n\r]+)', re.IGNORECASE)
    address_match = address_pattern.search(text)
    address = address_match.group(1) if address_match else None

    # Пример использования
    return {
        "address": address
    }

# Пример использования для первого текста
address_info = extract_address_information(all_texts[0])
print(f"Извлеченная информация по адресу проживания: {address_info}")

def extract_cadastral_number_information(text):
    # Пример регулярного выражения для извлечения кадастрового номера участка
    cadastral_number_pattern = re.compile(r'\b(?:Кадастровый номер участка)\s*:\s*([^\n\r]+)', re.IGNORECASE)
    cadastral_number_match = cadastral_number_pattern.search(text)
    cadastral_number = cadastral_number_match.group(1) if cadastral_number_match else None

    # Пример использования
    return {
        "cadastral_number": cadastral_number
    }

# Пример использования для первого текста
cadastral_number_info = extract_cadastral_number_information(all_texts[0])
print(f"Извлеченная информация по кадастровому номеру участка: {cadastral_number_info}")

def extract_account_number_information(text):
    # Пример регулярного выражения для извлечения номера счета
    account_number_pattern = re.compile(r'\b(?:Номер счета)\s*:\s*([^\n\r]+)', re.IGNORECASE)
    account_number_match = account_number_pattern.search(text)
    account_number = account_number_match.group(1) if account_number_match else None

    # Пример использования
    return {
        "account_number": account_number
    }

# Пример использования для первого текста
account_number_info = extract_account_number_information(all_texts[0])
print(f"Извлеченная информация по номеру счета: {account_number_info}")

def extract_orgn_information(text):
    # Пример регулярного выражения для извлечения ОРГН
    orgn_pattern = re.compile(r'\b(?:ОРГН)\s*:\s*([^\n\r]+)', re.IGNORECASE)
    orgn_match = orgn_pattern.search(text)
    orgn = orgn_match.group(1) if orgn_match else None

    # Пример использования
    return {
        "orgn": orgn
    }

# Пример использования для первого текста
orgn_info = extract_orgn_information(all_texts[0])
print(f"Извлеченная информация по ОРГН: {orgn_info}")


def extract_debt_amount_information(text):
    # Пример регулярного выражения для извлечения суммы долга
    debt_amount_pattern = re.compile(r'\b(?:Сумма долга у арендатора плательщика)\s*:\s*([^\n\r]+)', re.IGNORECASE)
    debt_amount_match = debt_amount_pattern.search(text)
    debt_amount = debt_amount_match.group(1) if debt_amount_match else None

    # Пример использования
    return {
        "debt_amount": debt_amount
    }

# Пример использования для первого текста
debt_amount_info = extract_debt_amount_information(all_texts[0])
print(f"Извлеченная информация по сумме долга: {debt_amount_info}")

def extract_penalty_amount_information(text):
    # Пример регулярного выражения для извлечения суммы начисления пени
    penalty_amount_pattern = re.compile(r'\b(?:Сумма начисления пени у арендатора)\s*:\s*([^\n\r]+)', re.IGNORECASE)
    penalty_amount_match = penalty_amount_pattern.search(text)
    penalty_amount = penalty_amount_match.group(1) if penalty_amount_match else None

    # Пример использования
    return {
        "penalty_amount": penalty_amount
    }

# Пример использования для первого текста
penalty_amount_info = extract_penalty_amount_information(all_texts[0])
print(f"Извлеченная информация по сумме начисления пени: {penalty_amount_info}")

def extract_contract_dates_information(text):
    # Пример регулярного выражения для извлечения даты заключения договора
    contract_dates_pattern = re.compile(r'\b(?:Дата заключения договора)\s*:\s*([^\n\r]+)', re.IGNORECASE)
    contract_dates_match = contract_dates_pattern.search(text)
    contract_dates = contract_dates_match.group(1) if contract_dates_match else None

    # Пример использования
    return {
        "contract_dates": contract_dates
    }

# Пример использования для первого текста
contract_dates_info = extract_contract_dates_information(all_texts[0])
print(f"Извлеченная информация по датам заключения договора: {contract_dates_info}")

def extract_contract_end_date_information(text):
    # Пример регулярного выражения для извлечения даты окончания договора
    end_date_pattern = re.compile(r'\b(?:Дата окончания договора)\s*:\s*([^\n\r]+)', re.IGNORECASE)
    end_date_match = end_date_pattern.search(text)
    end_date = end_date_match.group(1) if end_date_match else None

    # Пример использования
    return {
        "end_date": end_date
    }

# Пример использования для первого текста
contract_end_date_info = extract_contract_end_date_information(all_texts[0])
print(f"Извлеченная информация по дате окончания договора: {contract_end_date_info}")

# Код для удаления знаков пунктуации


# Нейронная сеть



# Ваш JSON-файл с данными
json_file_path = 'D:/pythonProject/venv/package.json'

try:
    # Загрузка данных из JSON-файла
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

except FileNotFoundError:
    print(f"File not found: {json_file_path}")

except json.JSONDecodeError:
    print(f"Invalid JSON format in file: {json_file_path}")

# Извлечение текста и аннотаций
document_texts = [entry['document_text'] for entry in data]
annotations = [entry['annotations'] for entry in data]

# Подготовка данных для обучения
texts = []
labels = []

for entry_annotations in annotations:
    entry_labels = ["O"] * len(document_texts)

    for annotation in entry_annotations:
        label = annotation['label']
        start = annotation['start']
        end = annotation['end']

        entry_labels[start:end] = [f"B-{label}"] + [f"I-{label}"] * (end - start - 1)

    texts.append(document_texts)  # This line should be corrected to append the specific entry's text
    labels.append(entry_labels)

# Разделение на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Загрузка предобученного токенизатора BERT
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Преобразование текста и меток в формат, понимаемый моделью
tokenized_inputs_train = tokenizer(texts_train, truncation=True, padding=True, is_split_into_words=True)
tokenized_inputs_test = tokenizer(texts_test, truncation=True, padding=True, is_split_into_words=True)

# Преобразование меток в идентификаторы

label2id = {
    "O": 0,
    "B-ФИО": 1, "I-ФИО": 2,
    "B-ИНН": 3, "I-ИНН": 4,
    "B-Адрес_проживания": 5, "I-Адрес_проживания": 6,
    "B-Кадастровый_номер_участка": 7, "I-Кадастровый_номер_участка": 8,
    "B-Номер_счета": 9, "I-Номер_счета": 10,
    "B-ОРГН": 11, "I-ОРГН": 12,
    "B-Сумма_долга": 13, "I-Сумма_долга": 14,
    "B-Сумма_начисления_пени": 15, "I-Сумма_начисления_пени": 16,
    "B-Дата_заключения_договора": 17, "I-Дата_заключения_договора": 18,
    "B-Дата_окончания_договора": 19, "I-Дата_окончания_договора": 20
}
labels_ids_train = [[label2id[label] for label in entry] for entry in labels_train]
labels_ids_test = [[label2id[label] for label in entry] for entry in labels_test]
def annotations_to_labels(annotations, label2id, document_text):
    labels = ["O"] * len(document_text)

    for annotation in annotations:
        label = annotation['label']
        start = annotation['start']
        end = annotation['end']

        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"

    label_ids = [label2id[label] for label in labels]
    return label_ids

# Применим функцию к вашему JSON-файлу
    import json

    # Ваш исходный JSON-файл
json_data = '''
[
  {
    "document_text": "договор уи\n\nаренды земельного участка\nг тула 22 октября 2020 г\n\nминистерство имущественных и земельных отношений тульской области\nименуемое в дальнейшем арендодатель в лице заместителя министра\nимущественных и земельных отношений тульской области  казенного игоря\nвасильевича действующего на основании доверенности министерства\nимущественных и земельных отношений тульской области от 09 января 2020 года №\n2901136 с одной стороны и\n\nименуемый в дальнейшем арендатор с другой стороны на основании прот",
    "annotations": [
      {"label": "ФИО", "start": 189, "end": 210, "text": "Казенного Игоря Васильевича"},
      {"label": "ИНН", "start": 303, "end": 315, "text": "123456789012"},
      {"label": "Адрес проживания", "start": 410, "end": 440, "text": "г. Тула, ул. Примерная, д. 1"},
      {"label": "Кадастровый номер участка", "start": 520, "end": 538, "text": "71:30:030603:204"},
      {"label": "Номер счета", "start": 600, "end": 615, "text": "12345678901234567890"},
      {"label": "ОРГН", "start": 720, "end": 731, "text": "1234567890123"},
      {"label": "Сумма долга", "start": 850, "end": 870, "text": "50000.00"},
      {"label": "Сумма начисления пени", "start": 920, "end": 940, "text": "1500.00"},
      {"label": "Дата заключения договора", "start": 1050, "end": 1060, "text": "22.10.2020"},
      {"label": "Дата окончания договора", "start": 1130, "end": 1140, "text": "22.10.2025"}
    ]
  },
  {
    "document_text": "новый текст для договора уи\n\nаренды нового земельного участка\nг новый_город 01 января 2023 г\n\nновое министерство имущественных и земельных отношений новой области\nименуемое в дальнейшем новый_арендодатель в лице нового_заместителя нового_министра\nновых_имущественных и новых_земельных отношений новой области  нового_казенного нового_игоря\nнового_васильевича действующего на основании новой_доверенности нового_министерства\nновых_имущественных и новых_земельных отношений новой области от 01 февраля 2023 года №\n0102035 с одной стороны и\n\nновый_именуемый в дальнейшем новый_арендатор с другой стороны на основании нового_прот",
    "annotations": [
      {"label": "ФИО", "start": 189, "end": 210, "text": "Нового Арендодателя Васильевича"},
      {"label": "ИНН", "start": 303, "end": 315, "text": "987654321098"},
      {"label": "Адрес проживания", "start": 410, "end": 440, "text": "г. Новый_город, ул. Новая, д. 1"},
      {"label": "Кадастровый номер участка", "start": 520, "end": 538, "text": "72:40:040804:305"},
      {"label": "Номер счета", "start": 600, "end": 615, "text": "98765432109876543210"},
      {"label": "ОРГН", "start": 720, "end": "731", "text": "9876543210987"},
      {"label": "Сумма долга", "start": 850, "end": 870, "text": "75000.00"},
      {"label": "Сумма начисления пени", "start": 920, "end": 940, "text": "2000.00"},
      {"label": "Дата заключения договора", "start": 1050, "end": 1060, "text": "01.01.2023"},
      {"label": "Дата окончания договора", "start": 1130, "end": 1140, "text": "01.01.2028"}
    ]
  },
  {
    "document_text": "еще один новый текст для договора уи\n\nаренды еще одного земельного участка\nг еще_один_город 15 февраля 2023 г\n\nеще_одно новое министерство имущественных и земельных отношений еще_одной области\nименуемое в дальнейшем еще_один_арендодатель в лице еще_одного_заместителя еще_одного_министра\nеще_одного_имущественных и еще_одного_земельных отношений еще_одной области  еще_одного_казенного еще_одного_игоря\nеще_одного_васильевича действующего на основании еще_одной_доверенности еще_одного_министерства\nеще_одного_имущественных и еще_одного_земельных отношений еще_одной области от 15 марта 2023 года №\n1503035 с одной стороны и\n\nеще_один_именуемый в дальнейшем еще_один_арендатор с другой стороны на основании еще_одного_прот",
    "annotations": [
      {
        "label": "ФИО",
        "start": 189,
        "end": 210,
        "text": "Семенов Семен Семенович"
      },
      {
        "label": "ИНН",
        "start": 303,
        "end": 315,
        "text": "987699921098"
      },
      {
        "label": "Адрес проживания",
        "start": 410,
        "end": 440,
        "text": "г. Новый_город, ул. Новая, д. 1"
      },
      {
        "label": "Кадастровый номер участка",
        "start": 520,
        "end": 538,
        "text": "72:40:040804:305"
      },
      {
        "label": "Номер счета",
        "start": 600,
        "end": 615,
        "text": "35765432109876543210"
      },
      {
        "label": "ОРГН",
        "start": 720,
        "end": 731,
        "text": "6896543210987"
      },
      {
        "label": "Сумма долга",
        "start": 850,
        "end": 870,
        "text": "55000.00"
      },
      {
        "label": "Сумма начисления пени",
        "start": 920,
        "end": 940,
        "text": "1000.00"
      },
      {
        "label": "Дата заключения договора",
        "start": 1050,
        "end": 1060,
        "text": "01.01.2023"
      },
      {
        "label": "Дата окончания договора",
        "start": 1130,
        "end": 1140,
        "text": "01.01.2028"
      }
    ]
  },
  {
    "document_text": "ваш четвертый документ",
    "annotations": [
      {"label": "ФИО", "start": 80, "end": 105, "text": "Сидоров Николай Васильевич"},
      {"label": "ИНН", "start": 180, "end": 195, "text": "555444333222"},
      {"label": "Адрес проживания", "start": 280, "end": 310, "text": "г. Екатеринбург, ул. Лесная, д. 15"},
      {"label": "Кадастровый номер участка", "start": 380, "end": 400, "text": "66:03:012345:678"},
      {"label": "Номер счета", "start": 480, "end": 495, "text": "55544433322211110000"},
      {"label": "ОРГН", "start": 580, "end": 595, "text": "5554443332221"},
      {"label": "Сумма долга", "start": 680, "end": 700, "text": "45000.00"},
      {"label": "Сумма начисления пени", "start": 780, "end": 800, "text": "1200.00"},
      {"label": "Дата заключения договора", "start": 1050, "end": 1060, "text": "01.01.2023"},
      {"label": "Дата окончания договора", "start": 1130, "end": 1140, "text": "01.01.2028"}
    ]
  }
]
'''
# Преобразование JSON-строки в объект Python
data = json.loads(json_data)

# Использование переменной `data` для обращения к вашим данным
document_texts = [entry['document_text'] for entry in data]

# Предобработка текста и создание BIO-тегов для каждого документа
annotations = []
for entry in data:
    text = entry['document_text']
    labels = ['O'] * len(text)  # Инициализация тегов "O" (Outside) для каждого символа

    for annotation in entry['annotations']:
        label = annotation['label']
        start = annotation['start']
        end = annotation['end']

        # Замена тегов в соответствии с BIO-форматом
        labels[start] = f"B-{label}"
        labels[start + 1:end] = [f"I-{label}"] * (end - start - 1)

    annotations.append(labels)

# Вывод результатов для каждого документа
for text, labels in zip(document_texts, annotations):
    print(f"Текст: {text}")
    print(f"BIO-теги: {' '.join(labels)}\n{'='*50}")


for entry in json_data:
    label_ids = annotations_to_labels(entry['annotations'], label2id, entry['document_text'])
    labels.append(label_ids)
    texts.append(entry['document_text'])



labels_ids_train = [[label2id[label] for label in entry] for entry in labels_train]
labels_ids_test = [[label2id[label] for label in entry] for entry in labels_test]


# Преобразование текста и меток в формат, понимаемый моделью
tokenized_inputs_train = tokenizer(texts_train, truncation=True, padding=True, is_split_into_words=True)
tokenized_inputs_test = tokenizer(texts_test, truncation=True, padding=True, is_split_into_words=True)

# Преобразование меток в идентификаторы
labels_ids_train = [[label2id[label] for label in entry] for entry in labels_train]
labels_ids_test = [[label2id[label] for label in entry] for entry in labels_test]

# Класс Dataset для загрузки данных в PyTorch DataLoader
class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Создание Dataset и DataLoader
train_dataset = NERDataset(tokenized_inputs_train, labels_ids_train)
test_dataset = NERDataset(tokenized_inputs_test, labels_ids_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Загрузка предобученной модели BERT для классификации токенов
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id))

# Определение оптимизатора
optimizer = AdamW(model.parameters(), lr=5e-5)

# Обучение модели
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Оценка модели на тестовой выборке
model.eval()
all_preds = []
for batch in test_loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    preds = model(input_ids, attention_mask=attention_mask).logits.argmax(dim=2).tolist()
    all_preds.extend(preds)

# Вывод результатов
for preds, labels in zip(all_preds, labels_ids_test):
    # Обработка результатов по необходимости
    print(preds, labels)
