# checking with Albert

from transformers import AlbertTokenizer
import torch

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

def question_answering(model, question, text, correct_label, device):
    
    encoding = tokenizer.encode_plus(question, text)
    input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
    
    target_start = torch.tensor([correct_label[0]], device = device)
    target_end = torch.tensor([correct_label[1]], device = device)

    loss, start_scores, end_scores = model(torch.tensor([input_ids]).to(device), token_type_ids=torch.tensor([token_type_ids]).to(device), start_positions= target_start , end_positions= target_end)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

    
    return loss, answer, torch.argmax(start_scores), torch.argmax(end_scores)

def multiple_choice(model, question, text, candidate ,correct_label, device):
    
    encoding = tokenizer.encode_plus(question, text)

    max_len = max([len(tokenizing(opt)) for opt in candidate])
    
    input_ids, token_type_ids = [], []
    for opt in candidate:
        tokenized_opt = tokenizing(opt)
        num_tok = len(tokenized_opt)
        encoded_options = tokenizer.encode(tokenized_opt + ['<pad>']*(max_len - num_tok))#[1:]
        input_ids += [encoded_options + encoding["input_ids"][1:]]
        token_type_ids += [[0]*(max_len+1) + encoding["token_type_ids"]]
 
    input_ids = torch.tensor(input_ids, device = device).unsqueeze(0) 
    token_type_ids = torch.tensor(token_type_ids, device = device).unsqueeze(0) 
 
    
    labels = torch.tensor(correct_label[0], device = device).unsqueeze(0)  # Batch size 1
 
    outputs = model(input_ids, labels=labels)

    loss, classification_scores = outputs[:2]
 
    return loss, torch.argmax(classification_scores)


def boolean_classification(model, question, text, q_type, candidate ,correct_label,other, device):

    encoding = tokenizer.encode_plus(question, text)

    if candidate: max_len = max([len(tokenizing(opt)) for opt in candidate])
    
    input_ids, token_type_ids = [], []
    if q_type == 'CO':
        labels = torch.tensor([[0]]*2, device = device).long()
        for opt in candidate[:2]:
            tokenized_opt = tokenizing(opt)
            num_tok = len(tokenized_opt)
            encoded_options = tokenizer.encode(tokenized_opt + ['<pad>']*(max_len - num_tok))#[1:]
            input_ids += [encoded_options + encoding["input_ids"][1:]]
        if correct_label == [0] or correct_label == [2]: labels[0][0] = 1
        if correct_label == [1] or correct_label == [2]: labels[1][0] = 1

            
    elif q_type == 'FR':
        labels = torch.tensor([0]*7, device = device).long()
        for ind, opt in enumerate(candidate[:7]): #[:7]):
            input_ids += [encoding["input_ids"]]
            if ind in correct_label:labels[ind] = 1

    elif q_type == 'FB':
        
        labels = torch.tensor([[0]]*len(candidate), device = device).long()
        for opt in candidate:
            tokenized_opt = tokenizing(opt)
#             num_tok = len(tokenized_opt)
            encoded_options = tokenizer.encode(tokenized_opt)#[1:]
            input_ids += [encoded_options + encoding["input_ids"][1:]]
            
        if 'A' in correct_label: labels[0][0] = 1
        if 'B' in correct_label: labels[1][0] = 1
        if 'C' in correct_label: labels[2][0] = 1
        
#         labels = torch.tensor([0]*3, device = device).long()
#         blocks = ['A', 'B', 'C']
#         for ind, opt in enumerate(blocks):
#             input_ids += [encoding["input_ids"]]
#             if blocks[ind] in correct_label: labels[ind] = 1
    
    elif q_type == 'YN':# and candidate != ['babi']:
        if correct_label == ['Yes']: labels = torch.tensor([1,0,0], device = device).long()
        elif correct_label == ['No']: labels = torch.tensor([0,1,0], device = device).long()
        else: labels = torch.tensor([0,0,1], device = device).long()
        input_ids = [encoding["input_ids"]]
        
#     elif q_type == 'YN' and candidate == ['babi']:
#         labels = torch.tensor([1,0], device = device).long() if correct_label == ['Yes'] else torch.tensor([0,1], device = device).long()
#         input_ids = [encoding["input_ids"]]    



    input_ids = torch.tensor(input_ids, device = device)

    outputs = model(input_ids, labels=labels)

    loss, logits = outputs[:2]
    out_logit = [torch.argmax(log) for log in logits]


    out = [0]
    if q_type == 'FR':
        
        out = [ind for ind,o in enumerate(out_logit) if o.item() == 1]
        if 2 in out and 3 in out:
            if logits[2][1] >= logits[3][1]:
                out.remove(3)
            else:
                out.remove(2)
        if 0 in out and 1 in out:
            if logits[0][1] >= logits[1][1]:
                out.remove(1)
            else:
                out.remove(0)
        if 4 in out and 5 in out:
            if logits[4][1] >= logits[5][1]:
                out.remove(5)
            else:
                out.remove(4)
        if out == []: out = [7]
            
    elif q_type == 'FB':
        blocks = ['A', 'B', 'C']
        out = [blocks[ind] for ind,o in enumerate(out_logit) if o.item() == 1]
#         out = [blocks[ind] for ind,o in enumerate(out_logit) if o.item() == 1]
#         if 'C' in out and 'C' not in candidate: out.remove('C')
            
    elif q_type == 'YN' and candidate != ['babi']:
        
        max_arg = torch.argmax(logits[:, 1])

        if max_arg.item() == 0: out = ['Yes']
        elif max_arg.item() == 1: out = ['No']
        else: out = ['DK']
            
#         if out_logit[0] == out_logit[1]:
#             if out_logit[0].item() == 0: out = ['DK'] 
#             else: 
#                 max_arg = torch.argmax(logits[:, 1])
#                 out = ['Yes'] if max_arg.item() == 0 else ['No']
#         else: out = ['Yes'] if out_logit[0].item() == 1 else ['No']
    
    elif q_type == 'YN' and candidate == ['babi']:
        max_arg = torch.argmax(logits[:, 1])
        out = ['Yes'] if max_arg.item() == 0 else ['No']
        
    elif q_type == 'CO':
        out = [ind for ind,o in enumerate(out_logit) if o.item() == 1]
        if 0 in out and 1 in out:
            out = [2]
        elif out == []: out = [3]
    return loss, out
    

def tokenizing(text):
    
    encoding = tokenizer.tokenize(text)
    
    return encoding