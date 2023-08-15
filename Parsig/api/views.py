from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

#pos
import re
import torch
# import evaluate


import tokenizers

# import numpy as np
import transformers
# import matplotlib.pyplot as plt
# from huggingface_hub import notebook_login


map_char = {'seen': 'É',
 'zah': 'Ë',
 'backslash': 'Ì',
 'sheen': 'Í',
 'ornateleftparenthesis': 'Î',
 'gaf': 'Ï',
 'jeem': 'Ð',
 'w': 'Ñ',
 'jeh': 'Ò',
 'florin': 'Ó',\
 'z': 'Ô',
 'a': 'Õ',
 's': 'Ö',
 'tatweel': '×',
 'fathatan': 'Ø',
 'dad': 'Ù',
 'percent': 'Ú',
 'heh': 'Û',
 'cedilla': 'Ü',
 'sad': 'Ý',
 'comma': 'Þ',
 'x': 'ß',
 't': 'à',
 'i': 'á',
 'dotlessi': 'â',
 'qaf': 'ã',
 'kasratan': 'ä',
 'farsiyeh': 'å',
 'e': 'æ',
 'colon': 'ç',
 'alefwithmaddaabove': 'è',
 'y': 'é',
 'hah': 'ê',
 'plus': 'ë',
 'shadda': 'ì',
 'ampersand': 'í',
 'ydieresis': 'î',
 'tehmarbuta': 'ï',
 'underscore': 'ð',
 'g': 'ñ',
 'zero': 'ò',
 'yehwithhamzaabove': 'ó',
 'p': 'ô',
 'circumflex': 'õ',
 'd': 'ö',
 'k': '÷',
 'divide': 'ø',
 'khah': 'ù',
 'h': 'ú',
 'arabiccomma': 'û',
 'lefttoright': 'ü',
 'lam': 'ý',
 'asciicircum': 'þ',
 'b': 'ÿ',
 'beh': 'Ā',
 'kafisolated': 'ā',
 'peh': 'Ă',
 'at': 'ă',
 'ain': 'Ą',
 'feh': 'ą',
 'logicalnot': 'Ć',
 'zain': 'ć',
 'OE': 'Ĉ',
 'two': 'ĉ',
 'r': 'Ċ',
 'tah': 'ċ',
 'n': 'Č',
 'dollar': 'č',
 'tcheh': 'Ď',
 'numbersign': 'ď',
 'noon': 'Đ',
 'l': 'đ',
 'ellipsis': 'Ē',
 'three': 'ē',
 'ghain': 'Ĕ',
 'exclam': 'ĕ',
 'question': 'Ė',
 'bar': 'ė',
 'slash': 'Ę',
 'wawwithhamzaabove': 'ę',
 'm': 'Ě',
 'scaron': 'ě',
 'meem': 'Ĝ',
 'reh': 'ĝ',
 'theh': 'Ğ',
 'multiply': 'ğ',
 'u': 'Ġ',
 'space': ' '}


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer_enc = transformers.AutoTokenizer.from_pretrained("omidgh/parsig_tokenizer")
tokenizer_dec = transformers.AutoTokenizer.from_pretrained("shirzady1934/parsig_tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained("omidgh/parsig_font2phon")
def run(sent):
    new_sent = []

    for i in sent.split():
        new_sent.append(map_char[i])
    
    sent = ''.join(new_sent)

    inputs = tokenizer_enc(sent, padding="max_length", truncation=True, max_length=100, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    output_str = tokenizer_dec.batch_decode(outputs, skip_special_tokens=True)

    return output_str[0]




tags = {'AAX',
        'B-ADJ',
        'I-ADJ',
        'B-ADV',
        'I-ADV',
        'B-N',
        'I-N',
        'B-V',
        'I-V',
        'B-PRONOUN',
        'I-PRONOUN',
        'B-NUM',
        'I-NUM',
        'B-DET',
        'I-DET',
        'B-PRE',
        'I-PRE',
        'B-POST',
        'I-POST',
        'B-CONJ',
        'I-CONJ',
        'B-JUNK',
        'I-JUNK',
        'B-MARKER',
        'I-MARKER',
        }

tag2id = {tag: id for id, tag in enumerate(sorted(list(tags)))}
#tag2id['AAX'] = -100
id2tag = {id: tag for tag, id in tag2id.items()}

modell = transformers.AutoModelForTokenClassification.from_pretrained('shirzady1934/parsig_pos')
tokenizerr = transformers.AutoTokenizer.from_pretrained('shirzady1934/parsig_tokenizer')

def test_input(inp):
    prd = torch.argmax(modell(input_ids=torch.unsqueeze(torch.tensor(tokenizerr(inp)['input_ids']), dim=0)).logits, dim=2)[0].detach().cpu().numpy()
    prd_lb = [id2tag[n] for n in prd ]
    fin = [tokenizerr.decode(x) for x in tokenizerr(inp)['input_ids'][1:-1]]
    print("\n\n\nInput: ", inp)
    print("\n\n\nInput: ", fin)
    print("Predicted:", prd_lb[1:-1])
    return str(prd_lb[1:-1])


# Create your views here.
# @api_view([ 'POST'])
def parsig_pos(request):
    parsed_text=None
    if request.method == 'POST':
        try:
            text =  request.POST.get('text', '')
            if text == '':
                pass
                # return Response({"status": 0, "error":"can not get text"})
            parsed_text=test_input(text)[1:-1]
            print(parsed_text)
            # return Response({"status": 1, "result":test_input(text)})
        except Exception as e:
            print(str(e))
            # return Response({"status": 0, "error":str(e)})
    return render(request, 'hello.html', {'parsed_text': parsed_text})



def parsig_Seq2Seq(request):
    parsed_text=None
    if request.method == 'POST':
        try:
            text =  request.POST.get('text', '')
            if text == '':
                pass
                # return Response({"status": 0, "error":"can not get text"})
            parsed_text=run(text)
            print(parsed_text)
            # return Response({"status": 1, "result":test_input(text)})
        except Exception as e:
            print(str(e))
            # return Response({"status": 0, "error":str(e)})
    return render(request, 'seq2seq.html', {'parsed_text': parsed_text})

