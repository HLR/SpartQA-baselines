import torch
import torch.nn as nn
import argparse
import os
# from transformers import AdamW
from BertModels import BertForQuestionAnswering, BertForBooleanQuestionFR, BertForBooleanQuestionFR1, BertForBooleanQuestionFB,BertForBooleanQuestionFB1, BertForBooleanQuestionYN , BertForBooleanQuestionYNboolq ,BertForBooleanQuestionYN1 , BertForBooleanQuestionCO, BertForBooleanQuestionCO1, BertForMaskedLM, BertForTokenClassification, BertForBooleanQuestion3ClassYN, BertForMultipleClass, BertForSequenceClassification, BertForSequenceClassification1, BertForSequenceClassification2
from XLNETModels import XLNETForQuestionAnswering, XLNETForBooleanQuestionFR, XLNETForBooleanQuestionFB, XLNETForBooleanQuestionYN , XLNETForBooleanQuestionCO
from ALBertModels import ALBertForQuestionAnswering, ALBertForBooleanQuestionFR, ALBertForBooleanQuestionFB, ALBertForBooleanQuestionYN , ALBertForBooleanQuestionCO
from BertSpatialQA import SpatialQA
from consistency_check import consistency, contrast
import matplotlib.pyplot as plt

#adding arguments
parser = argparse.ArgumentParser()

parser.add_argument("--result",help="Name of the result's saving file", type= str, default='test')
parser.add_argument("--result_folder",help="Name of the folder of the results file", type= str, default='SpaRT/Results')
parser.add_argument("--model",help="Name of the model's saving file", type= str, default='')
parser.add_argument("--model_folder",help="Name of the folder of the models file", type=str, default = "SpaRT//Models")

parser.add_argument("--dataset",help="name of the dataset like mSpRL or spaceeval", type = str, default = 'spartqa')

parser.add_argument("--no_save",help="If save the model or not", action='store_true', default = False)
parser.add_argument("--load",help="For loading model", type=str)
parser.add_argument("--cuda",help="The index of cuda", type=int, default=None)
parser.add_argument("--qtype",help="Name of Question type. (FB, FR, CO, YN)", type=str, default = 'all')
parser.add_argument("--train24k",help="Train on 24k data", action='store_true', default = True)
parser.add_argument("--train100k",help="Train on 100k data", action='store_true', default = False)
parser.add_argument("--train500",help="Train on 500 data", action='store_true', default = False)
parser.add_argument("--unseentest",help="Test on unseen data", action='store_true', default = False)
parser.add_argument("--human",help="Train and Test on human data", action='store_true', default = False)
parser.add_argument("--humantest",help="Test on human data", action='store_true', default = False)
parser.add_argument("--dev_exists", help="If development set is used", action='store_true', default = False)
parser.add_argument("--no_train",help="Number of train samples", action='store_true', default = False)

parser.add_argument("--baseline",help="Name of the baselines. Options are 'bert', 'xlnet', 'albert'", type=str, default = 'bert')
parser.add_argument("--pretrain",help="Name of the pretrained model. Options are 'bertqa', 'bertbc' (for bert boolean clasification), 'mlm', 'mlmr', 'tokencls'", type=str, default = 'bertbc')
parser.add_argument("--con",help="Testing consistency or contrast", type=str, default = 'not')
parser.add_argument("--optim",help="Type of optimizer. options 'sgd', 'adamw'.", type=str, default = 'adamw')
parser.add_argument("--loss",help="Type of loss function. options 'cross'.", type=str, default = 'cross')

parser.add_argument("--train",help="Number of train samples", type = int)
parser.add_argument("--train_log", help="save the log of train if true", default = False, action='store_true')
parser.add_argument("--start",help="The start number of train samples", type = int, default = 0)
parser.add_argument("--dev",help="Number of dev samples", type = int)
parser.add_argument("--test",help="Number of test samples", type = int)
parser.add_argument("--unseen",help="Number of unseen test samples", type = int)

parser.add_argument("--epochs",help="Number of epochs for training", type = int, default=0)
parser.add_argument("--lr",help="learning rate", type = float, default=2e-6)

parser.add_argument("--dropout", help="If you want to set dropout=0", action='store_true', default = False)
parser.add_argument("--unfreeze", help="unfreeze the first layeres of the model except this numbers", type=int, default = 0)

parser.add_argument("--other_var",  dest='other_var', action='store', help="Other variable: classification (DK, noDK), random, fine-tune on unseen. for changing model load MLM from pre-trained model and replace other parts with new on", type=str)
parser.add_argument("--detail",help="a description about the model", type = str)

args = parser.parse_args()

result_adress = os.path.join('/tank/space/rshnk/'+args.result_folder+('/' if args.dataset == 'spartqa' else '/'+args.dataset+'/')+args.baseline+'/',args.result)
if not os.path.exists(result_adress):
    os.makedirs(result_adress)
#saved_file = open('results/train'+args.result+'.txt','w')

#choosing device
if torch.cuda.is_available():
    print('Using ', torch.cuda.device_count() ,' GPU(s)')
    mode = 'cuda:'+str(args.cuda) if args.cuda else 'cuda'    
else:
    print("WARNING: No GPU found. Using CPUs...")
    mode = 'cpu'
device = torch.device(mode)

def config():

    f = open(result_adress+'/config.txt','w')
    print('Configurations:\n', args , file=f)
    f.close()

config()

epochs = args.epochs
if args.human: args.humantest = True


if args.train24k: train_num = 'train24k'
elif args.train100k: train_num = 'train100k'
elif args.train500: train_num = 'train500'
else: train_num = None

if args.model == '': args.model = args.result
    
#calling test and train based on the task
if args.pretrain == 'tokencls':
    if args.dataset == 'msprl':
        from msprl.train_tokencls_msprl import train
        from msprl.test_tokencls_msprl import test
    elif args.dataset == 'spaceEval':
        from spaceeval.train_tokencls_spaceEval import train
        from spaceeval.test_tokencls_spaceEval import test
    else:
        from spInfo.train_tokencls import train
        from spInfo.test_tokencls import test

elif args.pretrain == 'spcls' or args.pretrain == 'sptypecls':
    if args.dataset == 'msprl':
        from msprl.train_spcls_msprl import train
        from msprl.test_spcls_msprl import test
    else:    
        from spInfo.train_spcls import train
        from spInfo.test_spcls import test

elif args.pretrain == 'end2end':
    
    from end2end.train import train
    from end2end.test import test
    
else:
    if args.dataset == 'boolq':
        
        from boolq.train_boolQ import train
        from boolq.test_boolQ import test
        
    elif args.dataset == 'babi':
        from QA.train import train_babi
        from QA.test import test_babi
        
    else:
        from QA.train import train
        from QA.test import test
    
#model
# model = None
if args.load:
    
#     print('/tank/space/rshnk/'+args.model_folder+'/'+args.load+'.th')
    model = torch.load('/tank/space/rshnk/'+args.model_folder+'/'+args.load+'.th', map_location={'cuda:0': 'cuda:'+str(args.cuda),'cuda:1': 'cuda:'+str(args.cuda),'cuda:2': 'cuda:'+str(args.cuda),'cuda:3': 'cuda:'+str(args.cuda), 'cuda:5': 'cuda:'+str(args.cuda), 'cuda:4': 'cuda:'+str(args.cuda), 'cuda:6': 'cuda:'+str(args.cuda),'cuda:7': 'cuda:'+str(args.cuda)})
#     model.to(device)
    
    if args.unfreeze:
        if args.baseline == 'bert':
            for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)]: 
        #         print('I will be frozen: {}'.format(name)) 
                param.requires_grad = False
    
    if args.other_var == 'change_model':
        
        pretrained_dict = model.state_dict()
        if args.pretrain == 'bertbc':
            if args.qtype == 'YN':
                if args.baseline == 'bert':
                    model2 = BertForBooleanQuestionYN1.from_pretrained('bert-base-uncased',  device = device, drp= args.dropout)
            elif args.qtype == 'FB':
                if args.baseline == 'bert':
                    model2 = BertForBooleanQuestionFB1.from_pretrained('bert-base-uncased',  device = device, drp= args.dropout)
            elif args.qtype == 'FR':
                if args.baseline == 'bert':
                    model2 = BertForBooleanQuestionFR1.from_pretrained('bert-base-uncased',  device = device, drp= args.dropout)
            elif args.qtype == 'CO':
                if args.baseline == 'bert':
                    model2 = BertForBooleanQuestionCO1.from_pretrained('bert-base-uncased',  device = device, drp= args.dropout)
        elif args.pretrain == 'bertmc':
            if args.qtype == 'YN':
                if args.baseline == 'bert':
                    model2 =  BertForMultipleClass.from_pretrained('bert-base-uncased',  device = device,  drp= args.dropout)
        elif args.pretrain == 'sptypecls':
            if args.baseline == 'bert' and args.dataset == 'msprl':
                model2 = BertForSequenceClassification2.from_pretrained('bert-base-uncased', num_labels = 1, type_class = 23 , device = device,  drp= args.dropout)
        
#         if args.baseline == 'bert':
#             if args.unfreeze:
#                     for name, param in list(model2.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
#                         #print('I will be frozen: {}'.format(name)) 
#                         param.requires_grad = False
        
        model_dict = model2.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print(pretrained_dict.keys())
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # # 3. load the new state dict
        model2.load_state_dict(model_dict)
        
        model = model2
        
    model.to(device)   
else:
    if args.pretrain == 'bertqa': # for FA
        if args.baseline == 'bert':
            model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        elif args.baseline == 'albert':
            model = ALBertForQuestionAnswering.from_pretrained('albert-base-v2',  device = device)
        elif args.baseline == 'xlnet':
            model = XLNETForQuestionAnswering.from_pretrained('xlnet-base-cased',  device = device)
        model.to(device)
    
    elif args.pretrain == 'mlm' or args.pretrain =='mlmr':
        if args.baseline == 'bert':
            drop = 0 if args.dropout else 0.1
            #bert-large-uncased-whole-word-masking-finetuned-squad
#             bert-base-uncased
            model = BertForMaskedLM.from_pretrained('bert-base-uncased', hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True)
    
            if args.unfreeze:
                for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                    #print('I will be frozen: {}'.format(name)) 
                    param.requires_grad = False
        model.to(device)
    
    elif args.pretrain == 'end2end':
        if args.qtype == 'YN': qa_num_labels = 2
        elif args.qtype == 'FR': qa_num_labels = 7
        elif args.qtype == 'CO': qa_num_labels = 2
        elif args.qtype == 'FB': qa_num_labels = 3
        else: qa_num_labels = None
        
        if args.baseline == 'bert':
            drop = 0 if args.dropout else 0.1
            
            model = SpatialQA(drp=drop, qa_num_labels = qa_num_labels, rel_type_num = 11, qtype = args.qtype, device = device, unfreeze = args.unfreeze)
        
        model.to(device)
        
    elif args.pretrain == 'tokencls':
        if args.baseline == 'bert':
            drop = 0 if args.dropout else 0.1
            model = BertForTokenClassification.from_pretrained('bert-base-uncased', hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True, num_labels = 5)
            
            if args.unfreeze:
                for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                    #print('I will be frozen: {}'.format(name)) 
                    param.requires_grad = False    
        
        model.to(device)
    
    elif args.pretrain == 'spcls':
        if args.baseline == 'bert':
            drop = 0 if args.dropout else 0.1
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 1, device = device,  drp= args.dropout)
#             model = BertForSequenceClassification.from_pretrained('bert-base-uncased', hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True, num_labels = 1)

            if args.unfreeze:
                for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                    #print('I will be frozen: {}'.format(name)) 
                    param.requires_grad = False
        model.to(device)
        
    elif args.pretrain == 'sptypecls':
        if args.baseline == 'bert':
#             drop = 0 if args.dropout else 0.1
            if args.dataset == 'msprl':
                model = BertForSequenceClassification1.from_pretrained('bert-base-uncased', num_labels = 1, type_class = 23 , device = device,  drp= args.dropout)
                
            elif args.dataset == 'spaceEval':
                model = BertForSequenceClassification1.from_pretrained('bert-base-uncased', num_labels = 1, type_class = 22 , device = device,  drp= args.dropout)
            else:
                model = BertForSequenceClassification1.from_pretrained('bert-base-uncased', num_labels = 1, type_class = 11 , device = device,  drp= args.dropout)
#             model = BertForSequenceClassification.from_pretrained('bert-base-uncased', hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True, num_labels = 1)

            #unfreeze the layers
            if args.unfreeze:
                for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                    #print('I will be frozen: {}'.format(name)) 
                    param.requires_grad = False
        model.to(device)
    
    elif args.pretrain == 'bertmc':
        
        if args.qtype == 'YN':
#             drop = 0 if args.dropout else 0.1
            if args.baseline == 'bert':
                model =  BertForMultipleClass.from_pretrained('bert-base-uncased',  device = device,  drp= args.dropout)
#                 model =  BertForMultipleChoice.from_pretrained('bert-base-uncased',  hidden_dropout_prob = drop, attention_probs_dropout_prob = drop, return_dict=True)
                if args.unfreeze:
                    for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                        #print('I will be frozen: {}'.format(name)) 
                        param.requires_grad = False    
            model.to(device)
    
    elif args.pretrain == 'bertbc':
        
        if args.qtype == 'FR':
            if args.baseline == 'bert':
                model = BertForBooleanQuestionFR.from_pretrained('bert-base-uncased',  device = device,  drp= args.dropout)
                if args.unfreeze:
                    for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                        #print('I will be frozen: {}'.format(name)) 
                        param.requires_grad = False
            elif args.baseline == 'albert':
                model = ALBertForBooleanQuestionFR.from_pretrained('albert-base-v2',  device = device,  drp= args.dropout)
            elif args.baseline == 'xlnet':
                model = XLNETForBooleanQuestionFR.from_pretrained('xlnet-base-cased',  device = device,  drp= args.dropout)
            model.to(device)
            
        elif args.qtype == 'FB':
            if args.baseline == 'bert':
                model = BertForBooleanQuestionFB.from_pretrained('bert-base-uncased',  device = device,  drp= args.dropout)
                if args.unfreeze:
                    for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                        #print('I will be frozen: {}'.format(name)) 
                        param.requires_grad = False
            elif args.baseline == 'albert':
                model = ALBertForBooleanQuestionFB.from_pretrained('albert-base-v2',  device = device,  drp= args.dropout)
            elif args.baseline == 'xlnet':
                model = XLNETForBooleanQuestionFB.from_pretrained('xlnet-base-cased',  device = device,  drp= args.dropout)
            model.to(device)
            
        elif args.qtype == 'YN' and args.other_var == 'DK':
            
            if args.baseline == 'bert':
                model =  BertForBooleanQuestion3ClassYN.from_pretrained('bert-base-uncased',  device = device, drp= args.dropout) 
                if args.unfreeze:
                    for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                        #print('I will be frozen: {}'.format(name)) 
                        param.requires_grad = False
            model.to(device)
        
#         elif args.type =='YN' and args.other_var == 'YN1':
#             if args.baseline == 'bert':
#                 model =  BertForBooleanQuestionYN1.from_pretrained('bert-base-uncased',  device = device, drp= args.dropout)    
#             model.to(device)
        elif args.qtype == 'YN' and args.dataset == 'boolq':
            if args.baseline == 'bert':
                model = BertForBooleanQuestionYNboolq.from_pretrained('bert-base-uncased',  device = device, drp= args.dropout)
                if args.unfreeze:
                    for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                        #print('I will be frozen: {}'.format(name)) 
                        param.requires_grad = False
            model.to(device)
            
        elif args.qtype == 'YN':
            if args.baseline == 'bert':
                model = BertForBooleanQuestionYN.from_pretrained('bert-base-uncased',  device = device, drp= args.dropout)
                if args.unfreeze:
                    for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                        #print('I will be frozen: {}'.format(name)) 
                        param.requires_grad = False
            elif args.baseline == 'albert':
                model = ALBertForBooleanQuestionYN.from_pretrained('albert-base-v2',  device = device, drp= args.dropout)
            elif args.baseline == 'xlnet':
                model = XLNETForBooleanQuestionYN.from_pretrained('xlnet-base-cased',  device = device, drp= args.dropout)
            model.to(device)
        
        
            
        elif args.qtype == 'CO':
            if args.baseline == 'bert':
                model = BertForBooleanQuestionCO.from_pretrained('bert-base-uncased',  device = device, drp= args.dropout)
                if args.unfreeze:
                    for name, param in list(model.bert.named_parameters())[:(-12 * args.unfreeze)-2]: 
                        #print('I will be frozen: {}'.format(name)) 
                        param.requires_grad = False
            elif args.baseline == 'albert':
                model = ALBertForBooleanQuestionCO.from_pretrained('albert-base-v2',  device = device, drp= args.dropout)
            elif args.baseline == 'xlnet':
                model = XLNETForBooleanQuestionCO.from_pretrained('xlnet-base-cased',  device = device, drp= args.dropout)
            model.to(device)
            
            
# optimizer = None
if args.optim == 'sgd': 
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
elif args.optim == 'adamw':
     optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

# criterion = None
if args.loss == 'cross':
    criterion = nn.CrossEntropyLoss()


#training starts
all_loss, inter_test_all_accuracy, dev_all_accuracy, inter_test_unseen_all_accuracy, human_all_accuracy = [], [], [],[], []
all_accuracy = []
best_val, best_val_unseen = 0, 0


if not args.no_train: 
    print('~~~~~~~~~~~~ Train ~~~~~~~~~~~~ ')
    train_file = open(result_adress+'/train.txt','w')
    dev_file = open(result_adress+'/dev.txt','w')
    inter_test_file = open(result_adress+'/intermediate_test.txt','w')
    
for ep in range(epochs):
    
    
    
    print('******** Epoch '+str(ep)+' ******** ', file = train_file)
    print('******** Epoch '+str(ep)+' ******** ', file = dev_file)
    
    #train
    if args.no_train != True:

        losses, accuracy = train(model, criterion, optimizer, args.pretrain, args.baseline, args.start, args.train, train_num, args.qtype, args.human, args.other_var, device, args.train_log, train_file)

        all_loss.append(losses)
        all_accuracy.append(accuracy)
    
    #save model
    if not args.no_save:
#         print('/tank/space/rshnk/'+args.model_folder+'/model_'+args.baseline+('' if args.dataset == 'spartqa' else '_'+args.dataset)+'_final_'+args.model+'.th')
        torch.save(model, '/tank/space/rshnk/'+args.model_folder+'/model_'+args.baseline+('' if args.dataset == 'spartqa' else '_'+args.dataset)+'_final_'+args.model+'.th')
    
    #valid (actucally test)
    if args.dev_exists:
        
        dev_accuracy = test(model, args.pretrain, args.baseline, 'dev', args.dev, False, args.qtype, args.other_var, args.human, device, dev_file)
        dev_all_accuracy.append(dev_accuracy)
        accu = dev_accuracy
        
    else:
#         if args.human: 
        inter_test_accuracy = test(model, args.pretrain, args.baseline, 'test', args.test, False, args.qtype, args.other_var, args.human, device, inter_test_file)
#         else: 
#             inter_test_accuracy = test(model, args.pretrain, args.baseline, 'test', args.test, False, args.qtype, args.other_var, args.humantest, device, inter_test_file)
        
        inter_test_all_accuracy.append(inter_test_accuracy)
        accu = inter_test_accuracy
        
        
    if not args.no_save and best_val <= accu: 
        torch.save(model, '/tank/space/rshnk/'+args.model_folder+'/model_'+args.baseline+('' if args.dataset == 'spartqa' else '_'+args.dataset)+'_best_'+args.model+'.th')
        best_val = accu
        
    
    # show image of accuracy  
    if args.dev_exists:
        plt.figure()
        plt.plot(dev_all_accuracy, label="accuracy")
        plt.legend()
        plt.savefig(result_adress+'/dev_plot_acc.png')
    #     plt.show()
        plt.close()
    else:
        plt.figure()
        plt.plot(inter_test_all_accuracy, label="accuracy")
        plt.legend()
        plt.savefig(result_adress+'/inter_test_plot_acc.png')
    #     plt.show()
        plt.close()
    
    # show image of accuracy    
    if args.no_train != True:
        plt.figure()
        plt.plot(all_accuracy, label="accuracy")
        plt.legend()
        plt.savefig(result_adress+'/train_plot_acc.png')
    #     plt.show()
        plt.close()

        #show image of losses
        plt.figure()
        plt.plot(all_loss, label="loss")
        plt.legend()
        plt.savefig(result_adress+'/train_plot_loss.png')
    #     plt.show()
        plt.close()

if not args.no_train:
    dev_file.close()    
    train_file.close()
    inter_test_file.close()

if args.load and args.no_train:
    best_model = model
    best_model.to(device) 
    
elif args.no_train:
    best_model = model
    best_model.to(device) 
    
else:
    best_model = torch.load('/tank/space/rshnk/'+args.model_folder+'/model_'+args.baseline+('' if args.dataset == 'spartqa' else '_'+args.dataset)+'_best_'+args.model+'.th', map_location={'cuda:0': 'cuda:'+str(args.cuda),'cuda:1': 'cuda:'+str(args.cuda),'cuda:2': 'cuda:'+str(args.cuda),'cuda:3': 'cuda:'+str(args.cuda), 'cuda:5': 'cuda:'+str(args.cuda), 'cuda:4': 'cuda:'+str(args.cuda), 'cuda:6': 'cuda:'+str(args.cuda),'cuda:7': 'cuda:'+str(args.cuda)})
    best_model.to(device)        

print('~~~~~~~~~~~~ Test ~~~~~~~~~~~~ ')

if not args.human: 
    test_file = open(result_adress+'/test.txt','w')
    
    test_accuracy = test(best_model, args.pretrain, args.baseline, 'test', args.test, False, args.qtype, args.other_var, args.human, device, test_file)
#     test_all_accuracy.append(test_accuracy) 
    
if args.unseentest:
    
    inter_test_unseen_file = open(result_adress+'/unseen_test.txt','w')
    inter_test_unseen_accuracy = test(best_model, args.pretrain, args.baseline, 'test', args.unseen, True ,args.qtype, args.other_var, args.human, device, inter_test_unseen_file)
#     inter_test_unseen_all_accuracy.append(inter_test_unseen_accuracy)
    
    inter_test_unseen_file.close()
    
if args.humantest:

    human_file = open(result_adress+'/human_test.txt','w')
    human_accuracy = test(best_model, args.pretrain, args.baseline, 'test', args.test, False ,args.qtype, args.other_var, args.humantest, device, human_file)
#     human_all_accuracy.append(human_accuracy)
    
    
    human_file.close()
    
        

#test starts
  
if args.con != 'not' :
    print('~~~~~~~~~~~~ Consistency and Contrast ~~~~~~~~~~~~ ')
    
    
    if args.con == 'consistency':
        con_file = open(result_adress+'/consistency.txt','w') 
        test_accuracy = consistency(model, args.pretrain, args.baseline, args.test, args.qtype, args.other_var, args.human, device, con_file)
        con_file.close()
        
    elif args.con == 'contrast':
        con_file = open(result_adress+'/contrast.txt','w') 
        test_accuracy = contrast(model, args.pretrain, args.baseline, args.test, args.qtype, args.other_var, args.human, device, con_file)
        con_file.close()
    
    elif args.con == 'both':
        cons_file = open(result_adress+'/consistency.txt','w') 
        test_accuracy = consistency(model, args.pretrain, args.baseline, args.test, args.qtype, args.other_var, args.human, device, cons_file)
        cons_file.close()
        
        cont_file = open(result_adress+'/contrast.txt','w') 
        test_accuracy = contrast(model, args.pretrain, args.baseline, args.test, args.qtype, args.other_var, args.human, device, cont_file)
        cont_file.close()
    