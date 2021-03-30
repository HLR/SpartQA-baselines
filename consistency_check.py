import json
import torch

def consistency(model, pretrain, baseline, num_sample, qtype, other,device, file):
    
    
    #import baseline
    if baseline == 'bert':
        from BERT import question_answering, tokenizing, boolean_classification
    elif baseline == 'xlnet':
        from XLNet import question_answering, tokenizing, boolean_classification
    elif baseline == 'albert':
        from ALBERT import question_answering, tokenizing, boolean_classification
    
    
    with open('dataset/test.json') as json_file:
        data = json.load(json_file)
    
    s_ind = 0
    correct_consistency, consistency_total, is_correct =0, 0, 0
    
    model.eval()
    
    # with no auto gradient calculation, torch runs a bit faster
    with torch.no_grad():
        for story in data['data'][:num_sample]:
            s_ind+= 1
            print('sample ',s_ind)
            print('sample ',s_ind, file = file)
            story_txt = story['story'][0]
            print(story_txt, file = file)
            #embed text
            
            # each question (span)
            for question in story['questions']:
                
                if question['consistency_check'] == []: continue
                
                q_text, q_emb= '', []
                
                if question['q_type'] in [qtype]: #and len(question['answer']) == 1: #and x == 0:
                    
                    if question['q_type'] in ['FA'] and question['start_end_char'] == []: continue
                    
                    print('Main q:', question['question'], question['answer'] , file = file)
                    
                    q_text = question['question']  
                    
                    if pretrain == 'bertqa':
                            
                        correct_start_end_word = correct_token_id(story_txt, q_text, question['start_end_char'],tokenizing,  file)

                        _, output, start, end = question_answering(model, q_text, story_txt,correct_start_end_word, device)

                        print("Correct start end: ", correct_start_end_word, "\npredict: ", output, start, end, "\nstart end:", question['start_end_char'], file = file)

                        if question['answer'][0] == output and (start == correct_start_end_word[0] and end == correct_start_end_word[1]): 
                            consistency_total += 1
                        else: continue
                        
                        is_correct = 1
                        
                        for cons in question['consistency_check']:
                            
                            print('consistency q: ', cons['question'],cons['answer'], file = file)
                            
                            correct_start_end_word = correct_token_id(story_txt, cons['question'], cons['start_end_char'],tokenizing,  file)

                            _, output, start, end = question_answering(model, cons['question'], story_txt, correct_start_end_word, device)
                            
                            if cons['answer'][0] != output or (start != correct_start_end_word[0] or end != correct_start_end_word[1]): is_correct = 0; print('wrong', file=file) 


                    elif pretrain == 'bertbc':
                        
                        _, output = boolean_classification(model, q_text, story_txt, question['q_type'], question['candidate_answers'], question['answer'], other,device)
                        
                        print('predict: ', output, file=file)
                        
                        if question['answer'] == output : consistency_total += 1
                        else: continue
                        
                        is_correct = 1
                        
                        for cons in question['consistency_check']:
                            
                            print('consistency q: ', cons['question'],cons['answer'], file = file)
                            
                            _, output = boolean_classification(model, cons['question'], story_txt, question['q_type'], cons['candidate_answers'], cons['answer'], other,device)
                            print('predict: ', output, file=file)
                            if cons['answer'] != output : is_correct = 0; print('wrong', file=file) 
 
                        
                    correct_consistency += is_correct
                    
                    
    print('Consistency accuracy: ', correct_consistency, consistency_total, correct_consistency/ consistency_total)
    print('Consistency accuracy: ', correct_consistency, consistency_total, correct_consistency/ consistency_total, file = file)
    
    return -1#correct_consistency/ consistency_total
    


def contrast(model, pretrain, baseline, num_sample, qtype, other, device, file):
    

    #import baseline
    if baseline == 'bert':
        from BERT import question_answering, tokenizing, boolean_classification
    elif baseline == 'xlnet':
        from XLNet import question_answering, tokenizing, boolean_classification
    elif baseline == 'albert':
        from ALBERT import question_answering, tokenizing, boolean_classification
    
    

    with open('dataset/test.json') as json_file:
        data = json.load(json_file)
    
    s_ind = 0
    correct_contrast, contrast_total, is_correct =0, 0, 0
    
    model.eval()
    
    # with no auto gradient calculation, torch runs a bit faster
    with torch.no_grad():
        for story in data['data'][:num_sample]:
            s_ind+= 1
            print('sample ',s_ind)
            print('sample ',s_ind, file = file)
            story_txt = story['story'][0]
            print(story_txt, file = file)
            #embed text
            
            # each question (span)
            for question in story['questions']:
                
                if question['contrast_set'] == []: continue
                
                q_text, q_emb= '', []
                
                if question['q_type'] in [qtype]: #and len(question['answer']) == 1: #and x == 0:
                    
                    if question['q_type'] in ['FA'] and question['start_end_char'] == []: continue
                    
                    print('Main q:', question['question'], question['answer'] , file = file)
                    
                    q_text = question['question']  
                    
                    
                    if pretrain == 'bertqa':
                            
                        correct_start_end_word = correct_token_id(story_txt, q_text, question['start_end_char'],tokenizing,  file)

                        _, output, start, end = question_answering(model, q_text, story_txt,correct_start_end_word, device)

                        print("Correct start end: ", correct_start_end_word, "\npredict: ", output, start, end, "\nstart end:", question['start_end_char'], file = file)

                        if question['answer'][0] == output and (start == correct_start_end_word[0] and end == correct_start_end_word[1]): 
                            contrast_total += 1
                        else: continue
                        
                        is_correct = 1
                        
                        for cons in question['contrast_set']:
                            
                            print('contrast q: ', cons['question'],cons['answer'], file = file)
                            correct_start_end_word = correct_token_id(story_txt, cons['question'], cons['start_end_char'],tokenizing,  file)

                            _, output, start, end = question_answering(model, cons['question'], story_txt, correct_start_end_word, device)
                            
                            if cons['answer'][0] != output or (start != correct_start_end_word[0] or end != correct_start_end_word[1]): is_correct = 0; print('wrong', file=file) 
                            
                            
                    elif pretrain == 'bertbc':
                        
                        _, output = boolean_classification(model, q_text, story_txt, question['q_type'], question['candidate_answers'], question['answer'], other, device)
                        
                        print('predict: ', output, file=file)
                        
                        if question['answer'] == output : contrast_total += 1
                        else: continue
                        
                        is_correct = 1
                        
                        for cons in question['contrast_set']:
                            
                            print('contrast q: ', cons['question'],cons['answer'], file = file)
                            
                            _, output = boolean_classification(model, cons['question'], story_txt, question['q_type'], cons['candidate_answers'], cons['answer'], other, device)
                            print('predict: ', output, file=file)
                            if cons['answer'] != output : is_correct = 0; print('wrong', file=file) 
                                      
                                
                    correct_contrast += is_correct
                    
                    
    print('Contrast accuracy: ', correct_contrast, contrast_total, correct_contrast/ contrast_total)
    print('Contrast accuracy: ', correct_contrast, contrast_total, correct_contrast/ contrast_total, file = file)
    
    return -1




def correct_token_id(story, question, start_end, tokenizing, file):

    story_tokenized = tokenizing(story)
    q_tokenized = tokenizing(question)

    #finding the start and end token based on the characters
    sum_char = 0
    start_end_token = []
    for s_e in start_end[:1]:
        temp = s_e[0]
        sum_char = 0
        is_start,start, end = True, None, None
        for ind,word in enumerate(story_tokenized):
            len_word = len(word)
            if temp > sum_char + len(word) : sum_char += len_word; 
            else: 
                if is_start: 
                    start, is_start = ind , False
                    if  s_e[1]-1 <= sum_char + len(word): start_end_token+=[[start, ind]];break 
                    else: temp = s_e[1]-1
                else: start_end_token+=[[start, ind]]; break
            if ind != len(story_tokenized)-1 and story_tokenized[ind+1] != '.' and story_tokenized[ind+1] != ',' and story_tokenized[ind+1] != "'" and  story_tokenized[ind] != "'": sum_char += 1 # plus one for space

        
        start_end_token[-1][0] += len(q_tokenized)+2 # 2 for [cls] and [SEP]
        start_end_token[-1][1] += len(q_tokenized)+2



    return start_end_token[0] 