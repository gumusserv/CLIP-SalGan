import json

with open('sal_score_dic.json', 'r') as f:
    sal_score_dic = json.load(f)

print("sal model on pure text:")
print(sum(sal_score_dic['pure']['AUC']) / len(sal_score_dic['pure']['AUC']))
print(sum(sal_score_dic['pure']['sAUC']) / len(sal_score_dic['pure']['sAUC']))
print(sum(sal_score_dic['pure']['CC']) / len(sal_score_dic['pure']['CC']))
print(sum(sal_score_dic['pure']['NSS']) / len(sal_score_dic['pure']['NSS']))
print("sal model on sal text:")
print(sum(sal_score_dic['sal']['AUC']) / len(sal_score_dic['sal']['AUC']))
print(sum(sal_score_dic['sal']['sAUC']) / len(sal_score_dic['sal']['sAUC']))
print(sum(sal_score_dic['sal']['CC']) / len(sal_score_dic['sal']['CC']))
print(sum(sal_score_dic['sal']['NSS']) / len(sal_score_dic['sal']['NSS']))
print("sal model on nonsal text:")
print(sum(sal_score_dic['nonsal']['AUC']) / len(sal_score_dic['nonsal']['AUC']))
print(sum(sal_score_dic['nonsal']['sAUC']) / len(sal_score_dic['nonsal']['sAUC']))
print(sum(sal_score_dic['nonsal']['CC']) / len(sal_score_dic['nonsal']['CC']))
print(sum(sal_score_dic['nonsal']['NSS']) / len(sal_score_dic['nonsal']['NSS']))

with open('nonsal_score_dic.json', 'r') as fp:
    nonsal_score_dic = json.load(fp)

print("nonsal model on pure text:")
print(sum(nonsal_score_dic['pure']['AUC']) / len(nonsal_score_dic['pure']['AUC']))
print(sum(nonsal_score_dic['pure']['sAUC']) / len(nonsal_score_dic['pure']['sAUC']))
print(sum(nonsal_score_dic['pure']['CC']) / len(nonsal_score_dic['pure']['CC']))
print(sum(nonsal_score_dic['pure']['NSS']) / len(nonsal_score_dic['pure']['NSS']))
print("nonsal model on sal text:")
print(sum(nonsal_score_dic['sal']['AUC']) / len(nonsal_score_dic['sal']['AUC']))
print(sum(nonsal_score_dic['sal']['sAUC']) / len(nonsal_score_dic['sal']['sAUC']))
print(sum(nonsal_score_dic['sal']['CC']) / len(nonsal_score_dic['sal']['CC']))
print(sum(nonsal_score_dic['sal']['NSS']) / len(nonsal_score_dic['sal']['NSS']))
print("nonsal model on nonsal text:")
print(sum(nonsal_score_dic['nonsal']['AUC']) / len(nonsal_score_dic['nonsal']['AUC']))
print(sum(nonsal_score_dic['nonsal']['sAUC']) / len(nonsal_score_dic['nonsal']['sAUC']))
print(sum(nonsal_score_dic['nonsal']['CC']) / len(nonsal_score_dic['nonsal']['CC']))
print(sum(nonsal_score_dic['nonsal']['NSS']) / len(nonsal_score_dic['nonsal']['NSS']))