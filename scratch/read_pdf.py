from app.document_loader import load_pdf
import json

pages = load_pdf('data/documents/MACHINE LEARNING(R17A0534).pdf')

res = []
for i in [5, 12, 18, 25, 32, 45, 55, 65, 75, 85, 95, 105, 110, 115, 119]:
    if i < len(pages):
        text = pages[i]['text'][:300].replace('\n', ' ')
        res.append({"page": pages[i]['page_number'], "text": text})

with open('scratch/pdf_texts.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4)
