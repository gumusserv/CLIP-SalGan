import json

with open('general_score_dic.json', 'r') as f:
    sal_score_dic = json.load(f)

print("total model on pure text:")
print(sum(sal_score_dic['pure']['AUC']) / len(sal_score_dic['pure']['AUC']))
print(sum(sal_score_dic['pure']['sAUC']) / len(sal_score_dic['pure']['sAUC']))
print(sum(sal_score_dic['pure']['CC']) / len(sal_score_dic['pure']['CC']))
print(sum(sal_score_dic['pure']['NSS']) / len(sal_score_dic['pure']['NSS']))
print("total model on general text:")
print(sum(sal_score_dic['general']['AUC']) / len(sal_score_dic['general']['AUC']))
print(sum(sal_score_dic['general']['sAUC']) / len(sal_score_dic['general']['sAUC']))
print(sum(sal_score_dic['general']['CC']) / len(sal_score_dic['general']['CC']))
print(sum(sal_score_dic['general']['NSS']) / len(sal_score_dic['general']['NSS']))
