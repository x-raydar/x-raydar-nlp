import numpy as np
import torch
from transformers import RobertaTokenizer
from simpletransformers.custom_models.models import RobertaForMultiLabelSequenceClassification

def doc_to_torch(docs, tokenizer):
    # takes list of reports(strings) and turns into input tensors to model

    docs = ["<s> " + d + " </s>" for d in docs]
    tokenized_texts = [tokenizer.tokenize(d) for d in docs]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # pad or truncate input
    # max_len hardcoded as 512
    for i in range(len(input_ids)):
        if len(input_ids[i]) > 512:
            input_ids[i] = input_ids[i][:512]
        else:
            input_ids[i] = input_ids[i] + list(np.zeros(512 - len(input_ids[i]), dtype = int))

    #create attention masks (1s where tokens, 0s where no tokens)
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # put things into np arrays into torch tensors
    input_ids = torch.tensor(np.array(input_ids))
    attention_masks = torch.tensor(np.array(attention_masks))

    return input_ids, attention_masks

def build_model():
    state_dict = "./model/robertax1.0.pt"
    pretrained_model = "./model/robertax_pretrained/"

    model = RobertaForMultiLabelSequenceClassification.from_pretrained(pretrained_model, num_labels = 45)
    model.load_state_dict(torch.load(state_dict, map_location="cpu"))
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, do_lower_case = False)
    device = torch.device("cpu")
    model.eval()
    
    print('INFO: Model RoBERTaX ready')

    return model, tokenizer

def build_predictions(probs):
    # Assigns predicted probabilities to the 45 labels
    probs = probs.squeeze().tolist()
    predictions = dict(zip(load_list_nlp_labels(), probs))

    return predictions

def main(input_ids, attention_masks, model):
    # Given a report, returns the predictions
    
    with torch.no_grad():
        logits = model(input_ids, token_type_ids = None, attention_mask = attention_masks)[0]
    
    probs = logits.sigmoid().numpy()
    preds = build_predictions(probs)
    
    return preds

############################################################################################

def load_list_nlp_labels():
    l = [
        'abnormal_non_clinically_important',
        'aortic_calcification',
        'apical_fibrosis',
        'atelectasis',
        'axillary_abnormality',
        'bronchial_wall_thickening',
        'bulla',
        'cardiomegaly',
        'cavitating_lung_lesion',
        'clavicle_fracture',
        'comparison',
        'consolidation',
        'coronary_calcification',
        'dextrocardia',
        'dilated_bowel',
        'emphysema',
        'ground_glass_opacification',
        'hemidiaphragm_elevated',
        'hernia',
        'hyperexpanded_lungs',
        'interstitial_shadowing',
        'mediastinum_displaced',
        'mediastinum_widened',
        'normal',
        'object',
        'other',
        'paraspinal_mass',
        'paratracheal_hilar_enlargement',
        'parenchymal_lesion',
        'pleural_abnormality',
        'pleural_effusion',
        'pneumomediastinum',
        'pneumoperitoneum',
        'pneumothorax',
        'possible_diagnosis',
        'recommendation',
        'rib_fracture',
        'rib_lesion',
        'scoliosis',
        'subcutaneous_emphysema',
        'technical_issue',
        'undefined_sentence',
        'unfolded_aorta',
        'upper_lobe_blood_diversion',
        'volume_loss'
    ]

    return l