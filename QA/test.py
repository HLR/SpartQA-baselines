import json
import re
import random
import torch
import numpy as np
import torch.nn as nn
from QA.train import question_to_sentence, F1_measure, precision, recall, confusion_matrix

from BERT import tokenizing
# from ALBERT import tokenizing
# from XLNet import tokenizing


def test(model, pretrain, baseline, test_or_dev,num_sample, unseen, qtype, other, human, device, file):
    
    
    #import baseline
    if baseline == 'bert':
        from BERT import question_answering, tokenizing, boolean_classification, Masked_LM, Masked_LM_random, token_classification, multiple_classification
    elif baseline == 'xlnet':
        from XLNet import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'albert':
        from ALBERT import question_answering, tokenizing, multiple_choice, boolean_classification
    
    
    all_q = 0
    correct = 0
    correct_no_distance = 0
    s_ind = 0
    correct_consistency, consistency_total =0, 0
    
    model.eval()
    
    
    
    # with no auto gradient calculation, torch runs a bit faster
    with torch.no_grad():
        
        if unseen:
            with open('./dataset/unseen_test.json') as json_file:
                data = json.load(json_file)

        elif human:
            with open('./dataset/human_test.json') as json_file:
                data = json.load(json_file)
        else:
#             with open('./dataset/new_test.json') as json_file:
            with open('dataset/'+test_or_dev+'.json') as json_file:
                data = json.load(json_file)

        if qtype == 'YN': TPFN, TP, TPFP = np.array([0]*3), np.array([0]*3), np.array([0]*3)

        for story in data['data'][:num_sample]:
            s_ind+= 1
            print('sample ',s_ind)
            print('sample ',s_ind, file = file)
            story_txt = story['story'][0]

            #MLM tasks
            if pretrain == 'mlm':

#                 story_txt = "the square is above the cicle. the circle is above the rectangle. the square is above rectangle."
                print('Story:\n',story_txt, file = file)
                tasks_list = ['FB', 'FR', 'CO'] if qtype == 'all' else [qtype]

                for question in story['questions']:
                    q_text, q_emb= '', []
                    q_type = question['q_type']
                    if q_type in tasks_list :

                        q_text = question_to_sentence(question['question'], question['q_type'], question['answer'], question['candidate_answers'])

                        if len(question['answer'])> 1 or (q_type == 'FB' and question['answer'] == []) or (q_type == 'FR' and 7 in question['answer']) or (q_type == 'CO' and (question['answer'] == [2] or question['answer'] == [3])): continue

#                             print(question['q_id'],q_type, question['candidate_answers'], question['answer'][0])
                        answer = question['answer'][0] if q_type == 'FB' else question['candidate_answers'][question['answer'][0]]

                        if q_type == 'CO' and ('which' in answer  or 'in' in answer or 'that' in answer): continue

                        all_q += 1

                        print('Question: ',q_text,'\nAnswer: ', answer, file = file)

                        _, output, truth = Masked_LM(model, story_txt, q_text, answer, other, device, file)

            #             print("predict: ", output)
                        print("truth: ", truth, "\npredict: ", output, file = file)
            #             print("truth: ", truth, "\npredict: ", output)

                        correct_temp = 0
                        for i in range(len(output)):
                            if output[i] == truth[i]: correct_temp+=1

                        correct += correct_temp / len(output)

#                             if correct_temp / len(output) == 1:
#                                 correct += 1

                        print('total: ', all_q, ' correct: ', correct, file = file)
                        print('total: ', all_q, ' correct: ', correct)


            elif pretrain == 'mlmr':

                print('Story:\n',story_txt, file = file)

                all_q += 1

                _, output, truth = Masked_LM_random(model, story_txt, s_ind, other, device, file)

    #             print("predict: ", output)
                print("truth: ", truth, "\npredict: ", output, file = file)
    #             print("truth: ", truth, "\npredict: ", output)

                correct_temp = 0
                for i in range(len(output)):
                    if output[i] == truth[i]: correct_temp+=1

                correct += correct_temp / len(output)

#                             if correct_temp / len(output) == 1:
#                                 correct += 1

                print('total: ', all_q, ' correct: ', correct, file = file)
                print('total: ', all_q, ' correct: ', correct)


            #QA tasks
            else:
                # each question (span)
                for question in story['questions']:
                    q_text, q_emb= '', []
                    if question['q_type'] in [qtype]: #and len(question['answer']) == 1: #and x == 0:
                        if question['q_type'] in ['FA'] and question['start_end_char'] == []: continue
                        if other == 'noDK' and question['answer'] == ['DK']: continue

                        q_text = question['question']

                        all_q += 1
                        print('Story:\n',story_txt, file = file)
                        print('question: ', q_text, '\nanswer: ',question['answer'], file = file)  

                        if pretrain == 'bertqa':
                            correct_start_end_word = correct_token_id(story_txt, q_text, question['start_end_char'],tokenizing,  file)

                            _, output, start, end = question_answering(model, q_text, story_txt,correct_start_end_word, device)

                            print("Correct start end: ", correct_start_end_word, "\npredict: ", output, start, end, "\nstart end:", question['start_end_char'], file = file)

                            if question['answer'][0] == output and (start == correct_start_end_word[0] and end == correct_start_end_word[1]): 
                                correct+=1
                                print('total: ', all_q, ' correct: ', correct, file = file)
                                print('total: ', all_q, ' correct: ', correct)

                        elif pretrain == 'bertmc':

                            _, output = multiple_classification(model, q_text, story_txt, question['q_type'], question['candidate_answers'], question['answer'], other, device)

                            print("predict: ", output, file = file)

                            correct_answer = question['answer']
                            correct_answer.sort()
                            if correct_answer == output : 
                                correct+=1
                                print('total: ', all_q, ' correct: ', correct, file = file)
                                print('total: ', all_q, ' correct: ', correct)

                            if qtype == 'YN':
                                if correct_answer == ['Yes']: TPFN[0] += 1
                                elif correct_answer == ['No']: TPFN[1] += 1
                                elif correct_answer == ['DK']: TPFN[2] += 1

                                if output == ['Yes']: TPFP[0] += 1
                                elif output == ['No']: TPFP[1] += 1
                                elif output == ['DK']: TPFP[2] += 1

                                if output == correct_answer == ['Yes']: TP[0] += 1
                                elif output == correct_answer == ['No']: TP[1] += 1
                                elif output == correct_answer == ['DK']: TP[2] += 1

                            if qtype == 'FR' and human:
#                                     print(correct_answer, output)
                                if 4 in correct_answer: correct_answer.remove(4)
                                if 5 in correct_answer: correct_answer.remove(5)
                                if 4 in output: output.remove(4)
                                if 5 in output: output.remove(5)
                                if correct_answer == output : 
                                    correct_no_distance += 1
                                    print('total: ', all_q, ' correct_no_dist: ', correct_no_distance, file = file)
                                    print('total: ', all_q, ' correct_no_dist: ', correct_no_distance)

                        elif pretrain == 'bertbc':

                            _, output = boolean_classification(model, q_text, story_txt, question['q_type'], question['candidate_answers'], question['answer'], other, device)

                            print("predict: ", output, file = file)

                            correct_answer = question['answer']
                            correct_answer.sort()
                            output.sort()
                            
                            if correct_answer == output : 
                                correct+=1
                                print('total: ', all_q, ' correct: ', correct, file = file)
#                                     print('total: ', all_q, ' correct: ', correct)
                            else: print(s_ind, 'wrong')

                            if qtype == 'YN':
                                if correct_answer == ['Yes']: TPFN[0] += 1
                                elif correct_answer == ['No']: TPFN[1] += 1
                                elif correct_answer == ['DK']: TPFN[2] += 1

                                if output == ['Yes']: TPFP[0] += 1
                                elif output == ['No']: TPFP[1] += 1
                                elif output == ['DK']: TPFP[2] += 1

                                if output == correct_answer == ['Yes']: TP[0] += 1
                                elif output == correct_answer == ['No']: TP[1] += 1
                                elif output == correct_answer == ['DK']: TP[2] += 1

                            if qtype == 'FR' and human:
#                                     print(correct_answer, output)
                                if 4 in correct_answer: correct_answer.remove(4)
                                if 5 in correct_answer: correct_answer.remove(5)
                                if 4 in output: output.remove(4)
                                if 5 in output: output.remove(5)
                                if correct_answer == output : 
                                    correct_no_distance += 1
                                    print('total: ', all_q, ' correct_no_dist: ', correct_no_distance, file = file)
                                    print('total: ', all_q, ' correct_no_dist: ', correct_no_distance)




        if unseen:
            print(test_or_dev+ ' unseen Final accuracy: ', correct/ all_q)
            print(test_or_dev+' unseen Final accuracy: ', correct/ all_q, file = file)
        else: 
            print(test_or_dev, ' Final accuracy: ', correct/ all_q)
            print(test_or_dev, ' Final accuracy: ', correct/ all_q, file = file)

        if human and qtype == 'FR':
            print(test_or_dev, ' accuracy with no distance: ', correct_no_distance/ all_q, file = file)

        if qtype == 'YN':

            print('TP:',TP, '  TPFP: ', TPFP,'   TPFN: ', TPFN ,file = file)
            Precision = np.nan_to_num(TP / TPFP)
            Recall = np.nan_to_num(TP / TPFN)
            F1 = np.nan_to_num((2 * (Precision * Recall)) / (Precision + Recall))
            Macro_F1 = np.average(F1)

            print('Test Final Precision: ', Precision, file = file)

            print('Test Final Recall: ', Recall, file = file)

            print('Test Final F1: ', F1, file = file)

            print('Test Final Macro_F1: ', Macro_F1)
            print('Test Final Macro_F1: ', Macro_F1, file = file)

    #         return Macro_F1

        return correct/ all_q
    
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
            if temp > sum_char + len(word) : sum_char += len_word
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


def test_babi(model, pretrain, baseline, test_or_dev,num_sample,unseen, qtype, other, device, file):
    
    #import baseline
    if baseline == 'bert':
        from BERT import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'xlnet':
        from XLNet import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'albert':
        from ALBERT import question_answering, tokenizing, multiple_choice, boolean_classification
        
    
    with open('dataset/babi/'+test_or_dev+'.json') as json_file:
        data = json.load(json_file)
    
    all_q = 0
    correct = 0
    s_ind = 0
    
    #random sampling or not
    random.seed(1)
    stories = data['data'] #if other != 'random' else random.sample(data['data'], num_sample)
    
    number_samples = int((num_sample/8)+1) if num_sample else num_sample
    model.eval()
    
    # with no auto gradient calculation, torch runs a bit faster
    with torch.no_grad():
        for story in stories[:number_samples]:
            
#             if is_DK_babi(story['story'][0]): continue
            
            s_ind+= 1
            print('sample ',s_ind)
            print('sample ',s_ind, file = file)
            story_txt = story['story'][0]
            
            
            
            # each question (span)
            for question in story['questions']:
                q_text, q_emb= '', []
                
                if question['q_type'] in [qtype]: 
                    q_text = question['question']
                    
                    
                    
                    all_q += 1
                    print('Story:\n',story_txt, file = file)
                    print('question: ', q_text, '\nanswer: ',question['answer'], file = file)
                    
                    
                    _, output = boolean_classification(model, q_text, story_txt, question['q_type'], ['babi'], question['answer'], other, device)
                    
                    #print("logit: ", logit, file = file)
                    
                    print("predict: ", output, file = file)

                    correct_answer = question['answer']
                    correct_answer.sort()
                    if correct_answer == output : 
                        correct+=1
                        print('total: ', all_q, ' correct: ', correct, file = file)
                        print('total: ', all_q, ' correct: ', correct)
        
    print(test_or_dev,' Final accuracy: ', correct/ all_q)
    print(test_or_dev,' Final accuracy: ', correct/ all_q, file = file)
    
    return correct/ all_q


def test_boolq(model, pretrain, baseline, test_or_dev,num_sample,unseen, qtype, other, device, file):
    
    #import baseline
    if baseline == 'bert':
        from BERT import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'xlnet':
        from XLNet import question_answering, tokenizing, multiple_choice, boolean_classification
    elif baseline == 'albert':
        from ALBERT import question_answering, tokenizing, multiple_choice, boolean_classification
        
    start_number = 0
    
    if test_or_dev == 'dev':
        with open('dataset/boolQ/dev.json') as json_file:
            data = json.load(json_file)
    else:
        with open('dataset/boolQ/test_1.json') as json_file:
            data = json.load(json_file)
        
        start_number =  len(data['data']) - num_sample
        num_sample = None
    
    all_q = 0
    correct = 0
    s_ind = 0
    
    model.eval()
    
    # with no auto gradient calculation, torch runs a bit faster
    with torch.no_grad():
        for story in data['data'][start_number:num_sample]:
            s_ind+= 1
            print('sample ',s_ind)
            print('sample ',s_ind, file = file)
            story_txt = story['passage'][:1000]


            # each question (span)
#             for question in story['questions']:
#                 q_text, q_emb= '', []
#                 if question['q_type'] in [qtype]: 
            q_text = story['question']+'?'
            answer = ['Yes'] if story['answer'] == True else ['No']
            
            all_q += 1
            print('Story:\n',story_txt, file = file)
            print('question: ', q_text, '\nanswer: ', answer, file = file)  


            _, output = boolean_classification(model, q_text, story_txt, 'YN', ['boolq'], answer, other, device)

            print("predict: ", output, file = file)
            
            correct_answer = answer
            correct_answer.sort()
            if correct_answer == output : 
                correct+=1
                print('total: ', all_q, ' correct: ', correct, file = file)
                print('total: ', all_q, ' correct: ', correct)
        
    print('Test Final accuracy: ', correct/ all_q)
    print('Test Final accuracy: ', correct/ all_q, file = file)
    
    return correct/ all_q



def is_DK_babi(story):
    
    has_left = True if 'left' in story else False
    has_right = True if 'right' in story else False
    has_below = True if 'below' in story else False
    has_above = True if 'above' in story else False
    
    if has_left and (has_above or has_below): return True
    elif has_right and (has_above or has_below): return True
    elif has_above and (has_left or has_right): return True
    elif has_below and (has_left or has_right): return True
    
    return False
    