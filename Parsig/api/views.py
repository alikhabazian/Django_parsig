from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

#pos
import re
import torch
# import evaluate

# import numpy as np
import transformers
# import matplotlib.pyplot as plt
# from huggingface_hub import notebook_login


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
@api_view([ 'POST'])
def parsig_pos(request):
    if request.method == 'POST':
        try:
            text = request.data.get('text', '')
            if text == '':
                return Response({"status": 0, "error":"can not get text"})
            return Response({"status": 1, "result":test_input(text)})
        except Exception as e:
            return Response({"status": 0, "error":str(e)})


