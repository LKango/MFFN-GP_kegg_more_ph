import pandas as pd
import numpy as np
import re
import json

def parse_reaction_formula_side(s):
    """
        Parses the side formula, e.g. '2 C00001 + C00002 + 3 C00003'
        Ignores stoichiometry. 

        Returns:
            The set of CIDs.
    """

    if s.strip() == "null":
        return {}

    compound_bag = {}
    for member in re.split('\s+\+\s+', s):
        tokens = member.split(None, 1)
        if len(tokens) == 0:
            continue
        if len(tokens) == 1:
            amount = 1
            key = member
        else:
            amount = float(tokens[0])
            key = tokens[1] 

        compound_bag[key] = compound_bag.get(key, 0) + amount

    return compound_bag 

def parse_formula(formula, arrow='=', rid=None):
    """
        Parses a two-sided formula such as: 2 C00001 = C00002 + C00003

        Return:
            The set of substrates, products and the direction of the reaction
    """
    tokens = formula.split(arrow)

    if len(tokens) > 2:
        print(('Reaction contains more than one arrow sign (%s): %s'
               % (arrow, formula)))

    sparse_reaction = {}
    sparse_reaction['Substrates'] = {}
    sparse_reaction['Products'] = {}

    if len(tokens) == 1:
        cid = tokens[0].strip()
        sparse_reaction['Substrates'][cid] = 1

    if len(tokens) == 2:
        left = tokens[0].strip()
        right = tokens[1].strip()

        for cid, count in parse_reaction_formula_side(left).items():
            sparse_reaction['Substrates'][cid] = sparse_reaction.get(cid, 0) + count

        for cid, count in parse_reaction_formula_side(right).items():
            sparse_reaction['Products'][cid] = sparse_reaction.get(cid, 0) + count
   
    """
        {'Substrates': {'C00002': 1, 'C00059': 1, 'C00001': 1},'Products': {'C00009': 2.0, 'C00224': 1}}
    """
    return sparse_reaction 

def split_compounds(S, compounds):
    S1 = S.T
    S1.index = compounds

    rxn_dict1 = dict()
    rxn_dict2 = dict() 

    for rid, value in S1.items():
        rxn_substrates = {}
        rxn_products = {}
        rxn = S1[rid].to_frame() 
        # print("=================== rxn ====================")
        # print(rxn)
        r = rxn[~(rxn == 0).any(axis=1)] 
        # print("=================== r ====================")
        # print(r)
        rxns = r.index.tolist() 
       
        if len(r)==1: 
            rxn_dict1[rid] = 'None Compound!'
            rxn_products[rxns[0]] = round (r.values[0, 0], 3) 
            rxn_dict2[rid] = rxn_products
        if len(r)==0:
            rxn_dict1[rid] = 'None Compound!'
            rxn_dict2[rid] = 'None Compound!'
        else:
            M = abs(r.values[:, 0]).max()
            for i in range(len(r)):
                r.values[i, 0] = r.values[i, 0] / M  

                if r.values[i, 0] < 0: 
                    rxn_substrates[rxns[i]] = round(-r.values[i, 0], 3)
                    rxn_dict1[rid] = rxn_substrates
                else: 
                    rxn_products[rxns[i]] = round(r.values[i, 0], 3)
                    rxn_dict2[rid] = rxn_products

def generate_data(features,compounds):

    rootx = "./test1/mlp_data/sub_channel0/"
    rooty = "./test1/mlp_data/pro_channel0/"

    features1 = features
    features1.index = compounds
    features1 = features1.T
    substrates_dict = json.load(open('./test1/Substrates0.json'))
    products_dict = json.load(open('./test1/Products0.json'))
    
    maccs = 167
    maccs_add = 1
    
    PubChem = 881
    # PubChem_add = 19
    
    FP4 = 307
    # FP4_add = 17
    
    max = 1
    # label = []
    print("The product processing begins.")
    for key1, value1 in list(substrates_dict.items()):

        sub_channel = np.zeros((1, maccs+maccs_add))
        temp_channel = np.zeros((1, maccs + maccs_add))
        full_sub_channel = []
        for i in range(len(value1)):

            comp = list(value1.keys())[i] 
            fea = features1[comp].to_frame() 
            maccs_keys = fea.values[1188:1355,0] 
            PubChem_keys = fea.values[0:881,0] 
            FP4_keys = fea.values[881:1188, 0] 
            coefficient = list(value1.values())[i]
            maccs_keys = maccs_keys * coefficient
            temp_channel[0, 1:maccs+1] = maccs_keys 

            sub_channel += temp_channel
        sub_channel[0,0] = ph
        np.savetxt(rootx+'rxn_sub'+str(key1)+ '_' +str(ph)+'.txt', sub_channel, fmt='%.3f')
    print("The substrate processing is completed, and the product processing begins.")
    # np.savetxt(root + 'label.txt', label, fmt='%.7f')
    for key2, value2 in list(products_dict.items()):
        pro_channel = np.zeros((1, maccs))  
        temp_channel = np.zeros((1, maccs))
  
        if len(value2) < 10:
            for i in range(len(value2)):
                comp2 = list(value2.keys())[i]
                fea2 = features1[comp2].to_frame()

                maccs_keys2 = fea2.values[1188:1355,0]
                PubChem_keys2 = fea2.values[0:881, 0] 
                FP4_keys2 = fea2.values[881:1188, 0]
                coefficient = list(value2.values())[i]
                maccs_keys2 = maccs_keys2 * coefficient
                temp_channel[0, 0:maccs] = maccs_keys2 
                pro_channel += temp_channel

        else:
            pro_channel = pro_channel
        np.savetxt(rooty + 'rxn_pro' + key2 + '_' + str(ph) + '.txt', pro_channel, fmt='%.3f')
    print("The product processing is completed.")
if __name__ == '__main__':
    ph_list = ['5.0', '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5', '9.0']
    # ph_list = ['5.0']
    for ph in ph_list:
        print("===============================Process ", ph, " data===============================")
       
        compounds_df = pd.read_csv("./test1/compound_has_smiles.tsv", header=None) 
        features = pd.read_csv("./test1/features.csv", header=None) 
        compounds = compounds_df[0].tolist()
        generate_data(features,compounds)