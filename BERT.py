
# checking with BERT

from transformers import BertTokenizer, BertTokenizerFast
import torch
import random
import torch.nn as nn

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')

def question_answering(model, question, text, correct_label, device):
    
    encoding = tokenizer.encode_plus(question, text)
    input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
    
    target_start = torch.tensor([correct_label[0]], device = device)
    target_end = torch.tensor([correct_label[1]], device = device)

    loss, start_scores, end_scores = model(torch.tensor([input_ids]).to(device), token_type_ids=torch.tensor([token_type_ids]).to(device), start_positions= target_start , end_positions= target_end)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

    
    return loss, answer, torch.argmax(start_scores), torch.argmax(end_scores)

# def multiple_choice(model, question, text, candidate ,correct_label, device):
    
#     encoding = tokenizer.encode_plus(question, text)

#     max_len = max([len(tokenizing(opt)) for opt in candidate])
    
#     input_ids, token_type_ids = [], []
#     for opt in candidate:
#         tokenized_opt = tokenizing(opt)
#         num_tok = len(tokenized_opt)
#         encoded_options = tokenizer.encode(tokenized_opt + ['[PAD]']*(max_len - num_tok))#[1:]
#         input_ids += [encoded_options + encoding["input_ids"][1:]]
#         token_type_ids += [[0]*(max_len+1) + encoding["token_type_ids"]]
 
#     input_ids = torch.tensor(input_ids, device = device).unsqueeze(0) 
#     token_type_ids = torch.tensor(token_type_ids, device = device).unsqueeze(0) 
 
    
#     labels = torch.tensor(correct_label[0], device = device).unsqueeze(0)  # Batch size 1
 
#     outputs = model(input_ids, labels=labels)

#     loss, classification_scores = outputs[:2]
 
#     return loss, torch.argmax(classification_scores)


def boolean_classification(model, question, text, q_type, candidate ,correct_label, other, device):

    encoding = tokenizer.encode_plus(question, text)
#     print(text, question, candidate, correct_label)
    
    if candidate: max_len = max([len(tokenizing(opt)) for opt in candidate])
    
    input_ids, token_type_ids = [], []
    if q_type == 'CO':
        labels = torch.tensor([[0]]*2, device = device).long()
        for opt in candidate[:2]:
            tokenized_opt = tokenizing(opt)
            num_tok = len(tokenized_opt)
            encoded_options = tokenizer.encode(tokenized_opt + ['[PAD]']*(max_len - num_tok))#[1:]
            input_ids += [encoded_options + encoding["input_ids"][1:]]
            
        if correct_label == [0] or correct_label == [2]: labels[0][0] = 1
        if correct_label == [1] or correct_label == [2]: labels[1][0] = 1

            
    elif q_type == 'FR':
        labels = torch.tensor([0]*7, device = device).long()
        for ind, opt in enumerate(candidate[:7]): 
            input_ids += [encoding["input_ids"]]
            if ind in correct_label:labels[ind] = 1
#         print('labels', labels)
    
    if q_type == 'FB':
        
        labels = torch.tensor([[0]]*len(candidate), device = device).long()
        for opt in candidate:
            tokenized_opt = tokenizing(opt)
#             num_tok = len(tokenized_opt)
            encoded_options = tokenizer.encode(tokenized_opt)#[1:]
            input_ids += [encoded_options + encoding["input_ids"][1:]]
            
        if 'A' in correct_label: labels[0][0] = 1
        if 'B' in correct_label: labels[1][0] = 1
        if 'C' in correct_label: labels[2][0] = 1
            
#         print('labels', labels)
#     elif q_type == 'FB':
#         labels = torch.tensor([0]*3, device = device).long()
#         blocks = ['A', 'B', 'C']
#         for ind, opt in enumerate(blocks):
#             input_ids += [encoding["input_ids"]]
#             if blocks[ind] in correct_label: labels[ind] = 1
#     elif q_type == 'YN' and other == "multiple_class":
#         if correct_label == ['Yes']: labels = torch.tensor([0], device = device).long()
#         elif correct_label == ['No']: labels = torch.tensor([1], device = device).long()
#         else: labels = torch.tensor([2], device = device).long()
#         input_ids = [encoding["input_ids"]]*3
        
    elif q_type == 'YN' and other == "DK": #and candidate != ['babi']:
        if correct_label == ['Yes']: labels = torch.tensor([1,0,0], device = device).long()
        elif correct_label == ['No']: labels = torch.tensor([0,1,0], device = device).long()
        else: labels = torch.tensor([0,0,1], device = device).long()
        input_ids = [encoding["input_ids"]]
        
    elif q_type == 'YN' and other == "noDK":
        if correct_label == ['Yes']: labels = torch.tensor([1,0], device = device).long()
        elif correct_label == ['No']: labels = torch.tensor([0,1], device = device).long()
        input_ids = [encoding["input_ids"]]
    
    elif q_type == 'YN' and candidate == ['boolq']:
        if correct_label == ['Yes']: labels = torch.tensor([1,0], device = device).long()
        elif correct_label == ['No']: labels = torch.tensor([0,1], device = device).long()
#         else: labels = torch.tensor([0,0,1], device = device).long()
        input_ids = [encoding["input_ids"]]
    
    elif q_type == 'YN': #and candidate != ['babi']:
        if correct_label == ['Yes']: labels = torch.tensor([1,0,0], device = device).long()
        elif correct_label == ['No']: labels = torch.tensor([0,1,0], device = device).long()
        else: labels = torch.tensor([0,0,1], device = device).long()
        input_ids = [encoding["input_ids"]]
#         else : labels = torch.tensor([0,0], device = device).long()
    
        
#     elif q_type == 'YN' and candidate == ['babi']:
#         labels = torch.tensor([1,0], device = device).long() if correct_label == ['Yes'] else torch.tensor([0,1], device = device).long()
#         input_ids = [encoding["input_ids"]]    

    input_ids = torch.tensor(input_ids, device = device)
#     print(input_ids.shape, labels.shape)
    
    outputs = model(input_ids, labels=labels)
    
    loss, logits = outputs[:2]
#     print("loss, logits ", loss, logits)
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
    
    elif q_type == 'CO':
        out = [ind for ind,o in enumerate(out_logit) if o.item() == 1]
        if 0 in out and 1 in out:
            out = [2]
        elif out == []: out = [3]
    
    elif q_type == 'YN' and other == 'multiple_class':
        
        max_arg = torch.argmax(logits)
#         print(correct_label, logits, max_arg)
        if max_arg.item() == 0: out = ['Yes']
        elif max_arg.item() == 1: out = ['No']
        else: out = ['DK']
        
    elif q_type == 'YN' and other == 'DK' and (candidate == ['babi'] or candidate == ['boolq']):
#         print('logits: ', logits)
        max_arg = torch.argmax(logits[:2, 1])
#         print("2", max_arg )
        if max_arg.item() == 0: out = ['Yes']
        elif max_arg.item() == 1: out = ['No']
    
    elif q_type == 'YN' and other == 'DK':
        
        max_arg = torch.argmax(logits[:, 1])
#         print("2", max_arg , logits)
        if max_arg.item() == 0: out = ['Yes']
        elif max_arg.item() == 1: out = ['No']
        else: out = ['DK']
    
    elif q_type == 'YN' and other == 'noDK':
        
        max_arg = torch.argmax(logits[:, 1])
#         print("2", max_arg)
        if max_arg.item() == 0: out = ['Yes']
        elif max_arg.item() == 1: out = ['No']
#         else: out = ['DK']
    
    elif q_type == 'YN' and other == 'change_model':
        
#         max_arg = torch.argmax(logits[:, 1])
#         if max_arg.item() == 0: out = ['Yes']
#         elif max_arg.item() == 1: out = ['No']
        max_arg = torch.argmax(logits[:, 1])

        if max_arg.item() == 0: out = ['Yes']
        elif max_arg.item() == 1: out = ['No']
        else: out = ['DK']
            
        
    
    elif q_type == 'YN' and candidate != ['babi']:
        
        max_arg = torch.argmax(logits[:, 1])

        if max_arg.item() == 0: out = ['Yes']
        elif max_arg.item() == 1: out = ['No']
        else: out = ['DK']
            
#         if out_logit[0] == out_logit[1]:
            
#             if out_logit[0].item() == 0: out = ['DK'] 
#             else: 
#                 max_arg = torch.argmax(logits[: , 1])
#                 out = ['Yes'] if max_arg.item() == 0 else ['No']
                
#         else: out = ['Yes'] if out_logit[0].item() == 1 else ['No']
    
    
    elif q_type == 'YN' and (candidate == ['babi'] or candidate == ['boolq']):
        
        max_arg = torch.argmax(logits[:, 1])
        
        out = ['Yes'] if max_arg.item() == 0 else ['No']
        
    
            
#     print('out logit: ', out)
    
    return loss, out #, out_logit

    
def multiple_classification(model, question, text, q_type, candidate ,correct_label, other, device):

    encoding = tokenizer.encode_plus(question, text)
#     print(text, question, candidate, correct_label)
    
#     if candidate: max_len = max([len(tokenizing(opt)) for opt in candidate])
    
    input_ids, token_type_ids = [], []
    
    if q_type == 'YN' and other == "DK": #and candidate != ['babi']:
        if correct_label == ['Yes']: labels = torch.tensor([0], device = device).long()
        elif correct_label == ['No']: labels = torch.tensor([1], device = device).long()
        else: labels = torch.tensor([2], device = device).long()
        input_ids = [encoding["input_ids"]]
        
    elif q_type == 'YN' and other == "noDK":
        if correct_label == ['Yes']: labels = torch.tensor([0], device = device).long()
        elif correct_label == ['No']: labels = torch.tensor([1], device = device).long()
        input_ids = [encoding["input_ids"]]
        
    elif q_type == 'YN': #and candidate != ['babi']:
        if correct_label == ['Yes']: labels = torch.tensor([0], device = device).long()
        elif correct_label == ['No']: labels = torch.tensor([1], device = device).long()
        else: labels = torch.tensor([2], device = device).long()
        input_ids = [encoding["input_ids"]]
        
#     elif q_type == 'YN' and candidate == ['babi']:
#         labels = torch.tensor([1,0], device = device).long() if correct_label == ['Yes'] else torch.tensor([0,1], device = device).long()
#         input_ids = [encoding["input_ids"]]    

    input_ids = torch.tensor(input_ids, device = device)
#     print(input_ids.shape, labels.shape, labels)
    
    outputs = model(input_ids, labels=labels)
    
    loss, logits = outputs[:2]
    
#     print("loss, logits ", loss, logits)
    
#     out_logit = [torch.argmax(log) for log in logits]
    
    out = [0]
    
    if q_type == 'YN':
        
        max_arg = torch.argmax(logits[0])
# #         print(correct_label, max_arg)
        if max_arg.item() == 0: out = ['Yes']
        elif max_arg.item() == 1: out = ['No']
        else: out = ['DK']
    
            
#     print('out: ', out)
    
    return loss, out #, out_logit

def Masked_LM(model, text, question, answer, other, device, file):
    
#     print(question, answer)
#     input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    input_ids = tokenizer.encode_plus(question, text, return_tensors="pt")["input_ids"].to(device)

    masked = []
    for ind, i in enumerate(input_ids[0]):
        if i == 103: masked.append(ind)

    token_answer = [tokenizer.convert_tokens_to_ids(i) for i in tokenizing(answer)]
    
    x = [-100] * len(input_ids[0])
    for ind,i in enumerate(masked):
        x[i] = token_answer[ind]
    
    label_ids = torch.tensor([x], device = device)
    
#     print("input_ids2",input_ids)
#     print("label_ids",label_ids)
    
#     segments_ids = torch.tensor([[0]* (len(input_ids))], device = device)
    
    outputs = model(input_ids, labels  = label_ids) #, token_type_ids=segments_ids)
    
#     print(outputs)
    loss, predictions = outputs[:2]
    

    ground_truth = [label_ids[0][i].item() for i in masked]
    truth_token = tokenizer.convert_ids_to_tokens(ground_truth)
    print("truth token: ", truth_token, file = file)
    
    predicted_index = [torch.argmax(predictions[0, predict]).item() for predict in masked ]
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
#     print(predicted_index,predicted_token, file = file)


    print("pred_token: ", predicted_token, file = file)
    
    return loss, predicted_index, ground_truth
    
def Masked_LM_random(model, text, seed_num, other, device, file):
    
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    
#     print("input_ids1",input_ids)
    
    # masked tokens
    masked = []
#     forbiden = [".", ",", "!", "'","?", 'the', 'of']
    random.seed(seed_num)
    masked = random.sample(range(1, len(input_ids[0])-1), int(len(input_ids[0])* 0.12))
#     print("masks: ", masked[0], story_tokenized[masked[0]], file = file)
    
    
    #unchanged tokens, just predict
    not_changed = []
    temp , not_changed_num= 0, int(len(input_ids[0])* 0.015)
    while True:
        x = random.choice(range(1, len(input_ids[0])-1))
        if x not in masked: not_changed.append(x); temp+=1 
        if temp >= not_changed_num: break
    
    #changed tokens
    changed = []
    temp , changed_num= 0, int(len(input_ids[0])* 0.015)
    while True:
        x = random.choice(range(1, len(input_ids[0])-1))
        if x not in masked and x not in not_changed: changed.append(x); temp+=1 
        if temp >= changed_num: break
    
#     print("nums",masked, not_changed, changed)
    
    x = [-100] * len(input_ids[0])
    for i in masked:
        x[i] = input_ids[0][i].item()
        input_ids[0][i] = tokenizer.convert_tokens_to_ids('[MASK]') #103 #[masked]

    for i in not_changed:
        x[i] = input_ids[0][i].item()
    
    for i in changed:
        changed_word =random.choice(range(0,30522))
        x[i] = input_ids[0][i].item()
        input_ids[0][i]= changed_word
    
    label_ids = torch.tensor([x], device = device)
    
#     print("input_ids2",input_ids)
#     print("label_ids",label_ids)
    
    segments_ids = torch.tensor([[0]* (len(input_ids))], device = device)
    
    outputs = model(input_ids, labels  = label_ids, token_type_ids=segments_ids) #, token_type_ids=segments_ids)
    
#     print(outputs)
    loss, predictions = outputs[:2]
    
    predicted = []
    ground_truth = [label_ids[0][i].item() for i in masked] + [label_ids[0][i].item() for i in not_changed] + [label_ids[0][i].item() for i in changed]
    truth_token = tokenizer.convert_ids_to_tokens(ground_truth)
    print("truth token: ", truth_token, file = file)
    
    predicted_index = [torch.argmax(predictions[0, predict]).item() for predict in masked ]
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
#     print(predicted_index,predicted_token, file = file)
    predicted += predicted_index
    
    predicted_index = [torch.argmax(predictions[0, predict]).item() for predict in not_changed ]
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
#     print(predicted_index,predicted_t, file = fileoken)
    predicted += predicted_index
    
    predicted_index = [torch.argmax(predictions[0, predict]).item() for predict in changed ]
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
#     print(predicted_index,predicte, file = filed_token)
    predicted += predicted_index
    
    pred_token = tokenizer.convert_ids_to_tokens(predicted)
    print("pred_token: ", pred_token, file = file)
    
    return loss, predicted, ground_truth

    
# with just triplet classifier
def spatial_classification(model, sentence, triplet, label, device):
#     print('triplet',triplet, label)
    encoding = tokenizer.encode(sentence, add_special_tokens=True)
    token_type_ids = [0]*len(encoding)
    triplets_tokens = []
#     print('encoded sentence', encoding)
    if triplet['trajector'] != [-1,-1] :
        triplets_tokens += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1]+[102] 
        for x in range(triplet['trajector'][0],triplet['trajector'][1]+1):
            token_type_ids[x] = 1 
    else: triplets_tokens += [102]
    
    if triplet['landmark'] != [-1,-1]:
        triplets_tokens += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1]+[102]  
        for x in range(triplet['landmark'][0],triplet['landmark'][1]+1):
            token_type_ids[x] = 1 
    
    else: triplets_tokens += [102]
        
    
    if triplet['spatial_indicator'] != [-1,-1]:
        triplets_tokens += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1]+[102]
        for x in range(triplet['spatial_indicator'][0],triplet['spatial_indicator'][1]+1):
            token_type_ids[x] = 1 
    
#     print('triple tokens: ', triplets_tokens)
    
    token_type_ids = [0]*len(triplets_tokens) + token_type_ids
    encoding =[ encoding[0]]+ triplets_tokens + encoding[1:]
    
    token_type_ids = torch.tensor([token_type_ids], device = device)
    inputs = torch.tensor(encoding, device = device).unsqueeze(0)
    labels = torch.tensor([label], device = device).float()
#     print(inputs.shape, labels.shape)
    
    loss, logits = model(inputs, token_type_ids = token_type_ids, labels=labels)
#     print(outputs)
#     loss = outputs.loss
#     logits = outputs.logits
#     print('logits', logits)
    predicted_index = 1 if logits[0] > 0.5 else 0
#     predicted_index = torch.argmax(logits).item() 
    
#     print('predict', predicted_index)
    return loss, predicted_index#, predicted_index


#triplet classifier and relation tyoe extraction
def spatial_type_classification(model, sentence, triplet, label, type_label, device):

#     print(sentence,'\n',triplet)
    #tokenized sentence
    tokenized_text = tokenizing(sentence)
    
    encoding = tokenizer.encode(sentence, add_special_tokens=True)
    token_type_ids = [0]*len(encoding)
    
    triplets_tokens = []
#     print('encoded sentence', encoding)
    if triplet['trajector'] != ['','']:
        triplets_tokens += encoding[triplet['trajector'][0]:triplet['trajector'][1]+1]+[102] 
        for x in range(triplet['trajector'][0],triplet['trajector'][1]+1):
            token_type_ids[x] = 1 
        
    if triplet['landmark'] != ['','']:
        triplets_tokens += encoding[triplet['landmark'][0]:triplet['landmark'][1]+1]+[102]  
        for x in range(triplet['landmark'][0],triplet['landmark'][1]+1):
            token_type_ids[x] = 1 
        
    if triplet['spatial_indicator'] != ['','']:
        triplets_tokens += encoding[triplet['spatial_indicator'][0]:triplet['spatial_indicator'][1]+1]+[102]
        
        for x in range(triplet['spatial_indicator'][0],triplet['spatial_indicator'][1]+1):
            token_type_ids[x] = 1 
#         print('&&', tokenized_text[triplet['spatial_indicator'][0]-1:triplet['spatial_indicator'][1]]) # shouldn't consider [cls] so subtract 1
        
        spatial_indicator = ''
        for z in tokenized_text[triplet['spatial_indicator'][0]-1:triplet['spatial_indicator'][1]]:
            spatial_indicator += z +' '
#         spatial_indicator = [z+' ' for z in tokenized_text[triplet['spatial_indicator'][0]-1:triplet['spatial_indicator'][1]]][0]
#         print(all_rels_type[spatial_indicator[:-1]])
        
#         # for incorrect triplets
#         if label != None and all_rels_type != None:
#             if label == 0: type_label = 0
#             else:
#                 if spatial_indicator[:-1] in all_rels_type:
#                     type_label = rel_types.index(all_rels_type[spatial_indicator[:-1]]) 

#                 else:
#                     if spatial_indicator[:-1] == 'near': type_label = 5
#                     elif spatial_indicator[:-1] == 'far': type_label = 6
#                     else: type_label = 0
        
#     print(spatial_indicator, 'type rel',  type_label, all_rels_type)
#     print()
    token_type_ids = [0]*len(triplets_tokens) + token_type_ids
    encoding =[ encoding[0]]+ triplets_tokens + encoding[1:]
    
    token_type_ids = torch.tensor([token_type_ids], device = device)
    inputs = torch.tensor(encoding, device = device).unsqueeze(0)
    if label != None:
        labels = [torch.tensor([label], device = device).float(), torch.tensor([type_label], device = device).long()]
    #     print(inputs.shape, labels.shape)

        loss, logits1, logits2 = model(inputs, token_type_ids = token_type_ids, labels=labels)
    #     print(logits1)
    else:
        type_label = None
        loss = None
        _,logits1, logits2 = model(inputs, token_type_ids = token_type_ids)
        
    predicted_index1 = 1 if logits1[0] > 0.5 else 0
#     print('logits2', logits2)
    predicted_index2 = torch.argmax(logits2[0]).item()
#     predicted_index = torch.argmax(logits).item() 


    return loss, predicted_index1, predicted_index2#, predicted_index
    
    
        


def token_classification(model, text, traj, land, indicator, other, device, file = ''):    
    
    loss, truth = '', ''
    
    inputs = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    
    if traj or land or indicator:
        
        labels = make_token_label(text, traj, land, indicator, other).to(device)

        outputs = model(inputs, labels=labels)
        
        loss = outputs.loss
        truth = [element.item() for element in labels[0].flatten()]
    
    else: outputs = model(inputs)
    
    
    logits = outputs.logits
    
    predicted_index = [torch.argmax(predict).item() for predict in logits[0] ]
    
    return loss, predicted_index, truth
    


def make_token_label(text, traj, land, indicator,other):

#     print(text, traj, land, indicator)
    
    encoding = tokenizerFast(text, return_offsets_mapping= True)
    token_starts = [item[0] for item in encoding['offset_mapping']]
    token_ends = [item[1] for item in encoding['offset_mapping']]
#     print(token_starts, token_ends)
#     print()
#     labels_id = ['O', 'B_traj', 'I_traj', 'B_land', 'I_land', 'B_spatial', 'I_spatial']
    labels_id = ['O', 'B_entity', 'I_entity', 'B_indicator', 'I_indicator']
#     text_token = tokenized_text #tokenizing(text)
    labels = torch.tensor([0] * len(encoding['input_ids']))
    
    
    
    #trajector
    for ind, t in enumerate(traj):
        if t['start']!= '' and t['end']!= '':
#             print(t)
            B_token = token_starts[1:-1].index(t['start'])+1
            E_token = token_ends[1:-1].index(t['end'])+1
    #         print(B_token, E_token)
            labels[B_token] = labels_id.index('B_entity')
            for i in range(B_token+1, E_token+1):
                labels[i] = labels_id.index('I_entity')
        
        
    #landmark
    for l in land:
        if l['start']!= '' and l['end']!= '':
    #         print(l)
            B_token = token_starts[1:-1].index(l['start'])+1
            E_token = token_ends[1:-1].index(l['end'])+1
    #         print(B_token, E_token)
            labels[B_token] = labels_id.index('B_entity')
            for i in range(B_token+1, E_token+1):
                labels[i] = labels_id.index('I_entity')
        
    
    #spatial
    for ind in indicator:
        if ind['start']!= '' and ind['end']!= '':
    #         print(ind)
            B_token = token_starts[1:-1].index(ind['start'])+1
            E_token = token_ends[1:-1].index(ind['end'])+1
    #         print(B_token, E_token)
            labels[B_token] = labels_id.index('B_indicator')
            for i in range(B_token+1, E_token+1):
                labels[i] = labels_id.index('I_indicator')
    
#     print('labels:', labels)

    labels = labels.unsqueeze(0)
    return labels

def extract_entity_token(text, traj, land, indicator):

#     print(text, traj, land, indicator)
    
    encoding = tokenizerFast(text, return_offsets_mapping= True)
    token_starts = [item[0] for item in encoding['offset_mapping']]
    token_ends = [item[1] for item in encoding['offset_mapping']]   
#     print(token_starts, token_ends)

    token_index = {'trajector': [-1,-1], 'landmark': [-1,-1], 'spatial_indicator':[-1,-1]}
    #trajector

    if (traj['start']!= '' and traj['start']!= -1) and (traj['end']!= '' and traj['end']!= -1):
#             print(t)
        #start
        token_index['trajector'][0]= token_starts[1:-1].index(traj['start'])+1
        #end
        token_index['trajector'][1]= token_ends[1:-1].index(traj['end'])+1
        
    #landmark
    if (land['start']!= '' and land['start']!= -1) and (land['end']!= '' and land['end']!= -1):
#         print(l)
        #start
        token_index['landmark'][0]= token_starts[1:-1].index(land['start'])+1
        #end
        token_index['landmark'][1]= token_ends[1:-1].index(land['end'])+1
    
    #spatial
    if (indicator['start']!= '' and indicator['start']!= -1) and (indicator['end']!= '' and indicator['end']!= -1):
#         print(ind)
        token_index['spatial_indicator'][0] = token_starts[1:-1].index(indicator['start'])+1
        token_index['spatial_indicator'][1] = token_ends[1:-1].index(indicator['end'])+1

#     print('token_index:', token_index)

    return token_index




# def boolean_classification_end2end(model, question, text, q_type, candidate ,correct_label, other, device):
def boolean_classification_end2end(model, questions, text, q_type, candidates ,correct_labels, other, device):

#     encoding = tokenizer.encode_plus(question, text)
#     print(text, questions, candidates, correct_labels)
    
    #seperate each sentence
    sentences = [h+'.' for h in text.split('. ')]
    sentences[-1] = sentences[-1][:-1]
    #sentences = [question] + sentences #the first sentence always is the question
#     print('sentences',sentences)
    text_tokenized = tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"].to(device)
    
#     print('&&', questions)
    
    qs_tokenized = tokenizer(questions, return_tensors="pt", padding=True)["input_ids"].to(device)

#     print('## text+qs_tokenized: ',text_tokenized)
#     qs_tokenized = tokenizer(question, return_tensors="pt")["input_ids"].to(device)
#     print('### qs_tokenized: ',qs_tokenized)
    
#     inputs = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        
#     if candidate: max_len = max([len(tokenizing(opt)) for opt in candidate])
    
    input_ids, token_type_ids = [], []
    labels =[]
    options = None
    for _ind,correct_label in enumerate(correct_labels):
        
        if q_type == 'CO':

            label = torch.tensor([[0]]*2, device = device).long()
    #         candid_tokenized = tokenizer(candidate[:2], return_tensors="pt", padding=True)["input_ids"].to(device)
            for opt in candidates[_ind][:2]:
    #             tokenized_opt = tokenizing(opt)
    #             num_tok = len(tokenized_opt)
    #             encoded_options = tokenizer.encode(tokenized_opt + ['[PAD]']*(max_len - num_tok))#[1:]
                input_ids += [encoded_options + encoding["input_ids"][1:]]

            if correct_label == [0] or correct_label == [2]: label[0][0] = 1
            if correct_label == [1] or correct_label == [2]: label[1][0] = 1


        elif q_type == 'FB':

            label = torch.tensor([[0]]*len(candidates[_ind]), device = device).long()
#             candid_tokenized = tokenizer(candidates[_ind], return_tensors="pt", padding=True)["input_ids"].to(device)
#             for opt in candidates[_ind]:
    #             tokenized_opt = tokenizing(opt)
    # #             num_tok = len(tokenized_opt)
    #             encoded_options = tokenizer.encode(tokenized_opt)#[1:]
#                 input_ids += [encoded_options + encoding["input_ids"][1:]]
            options = len(candidates[_ind])
            if 'A' in correct_label: label[0][0] = 1
            if 'B' in correct_label: label[1][0] = 1
            if 'C' in correct_label: label[2][0] = 1

        elif q_type == 'FR':
#             _input = []
            
            label = torch.tensor([0]*7, device = device).long()
            for ind, opt in enumerate(candidates[_ind][:7]): 
#                 _input += [text_tokenized]
                if ind in correct_label:label[ind] = 1
#             text_tokenized = torch.stack(_input)
    #         print('label', label)

    #     elif q_type == 'YN' and other == "DK": #and candidate != ['babi']:
    #         if correct_label == ['Yes']: label = torch.tensor([1,0,0], device = device).long()
    #         elif correct_label == ['No']: label = torch.tensor([0,1,0], device = device).long()
    #         else: label = torch.tensor([0,0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]

    #     elif q_type == 'YN' and other == "noDK":
    #         if correct_label == ['Yes']: label = torch.tensor([1,0], device = device).long()
    #         elif correct_label == ['No']: label = torch.tensor([0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]

    #     elif q_type == 'YN' and candidate == ['boolq']:
    #         if correct_label == ['Yes']: label = torch.tensor([1,0], device = device).long()
    #         elif correct_label == ['No']: label = torch.tensor([0,1], device = device).long()
    # #         else: label = torch.tensor([0,0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]

        elif q_type == 'YN': #and candidate != ['babi']:
            if correct_label == ['Yes']: label = torch.tensor([1,0,0], device = device).long()
            elif correct_label == ['No']: label = torch.tensor([0,1,0], device = device).long()
            else: label = torch.tensor([0,0,1], device = device).long()
#             input_ids = text_tokenized
    #         else : label = torch.tensor([0,0], device = device).long()


    #     elif q_type == 'YN' and candidate == ['babi']:
    #         label = torch.tensor([1,0], device = device).long() if correct_label == ['Yes'] else torch.tensor([0,1], device = device).long()
    #         input_ids = [encoding["input_ids"]]    
    
        labels += [label]
    labels = torch.stack(labels).to(device)
#     print('$', correct_labels)
#     print('$$', labels, type(labels))
    
#     input_ids = torch.tensor(input_ids, device = device)
#     print('input shape, label shape',text_tokenized.shape, qs_tokenized.shape, labels.shape)
    
    _outputs = model(text_tokenized, qs_tokenized, labels=labels, options = options)
#     print('&&&&&&&&&& outputs', outputs)
    
    losses, outs = [], []
    for outputs in _outputs:
        
        loss, logits = outputs[:2]
        
#         print('$$', loss)
        losses += [loss]
#         print("loss, logits ", loss, logits)
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

        elif q_type == 'CO':
            out = [ind for ind,o in enumerate(out_logit) if o.item() == 1]
            if 0 in out and 1 in out:
                out = [2]
            elif out == []: out = [3]

        elif q_type == 'YN' and other == 'multiple_class':

            max_arg = torch.argmax(logits)
    #         print(correct_label, logits, max_arg)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']

        elif q_type == 'YN' and other == 'DK' and (candidates == ['babi'] or candidates == ['boolq']):
    #         print('logits: ', logits)
            max_arg = torch.argmax(logits[:2, 1])
    #         print("2", max_arg )
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']

        elif q_type == 'YN' and other == 'DK':

            max_arg = torch.argmax(logits[:, 1])
    #         print("2", max_arg , logits)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']

        elif q_type == 'YN' and other == 'noDK':

            max_arg = torch.argmax(logits[:, 1])
    #         print("2", max_arg)
            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
    #         else: out = ['DK']

        elif q_type == 'YN' and other == 'change_model':

    #         max_arg = torch.argmax(logits[:, 1])
    #         if max_arg.item() == 0: out = ['Yes']
    #         elif max_arg.item() == 1: out = ['No']
            max_arg = torch.argmax(logits[:, 1])

            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']



        elif q_type == 'YN' and candidates != ['babi']:

            max_arg = torch.argmax(logits[:, 1])

            if max_arg.item() == 0: out = ['Yes']
            elif max_arg.item() == 1: out = ['No']
            else: out = ['DK']

    #         if out_logit[0] == out_logit[1]:

    #             if out_logit[0].item() == 0: out = ['DK'] 
    #             else: 
    #                 max_arg = torch.argmax(logits[: , 1])
    #                 out = ['Yes'] if max_arg.item() == 0 else ['No']

    #         else: out = ['Yes'] if out_logit[0].item() == 1 else ['No']


        elif q_type == 'YN' and (candidates == ['babi'] or candidates == ['boolq']):

            max_arg = torch.argmax(logits[:, 1])

            out = ['Yes'] if max_arg.item() == 0 else ['No']
        
        outs += [out]
    losses = torch.stack(losses)        
#     print('out logit: ', outs, losses)
    
    return losses, outs #, out_logit


def tokenizing(text):
    
    encoding = tokenizer.tokenize(text)
    
    return encoding




