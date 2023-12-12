import json

with open('total_score_dic.json', 'r') as f:
    sal_score_dic = json.load(f)

print("total model on pure text:")
print(sum(sal_score_dic['pure']['AUC']) / len(sal_score_dic['pure']['AUC']))
print(sum(sal_score_dic['pure']['sAUC']) / len(sal_score_dic['pure']['sAUC']))
print(sum(sal_score_dic['pure']['CC']) / len(sal_score_dic['pure']['CC']))
print(sum(sal_score_dic['pure']['NSS']) / len(sal_score_dic['pure']['NSS']))
print("total model on sal text:")
print(sum(sal_score_dic['sal']['AUC']) / len(sal_score_dic['sal']['AUC']))
print(sum(sal_score_dic['sal']['sAUC']) / len(sal_score_dic['sal']['sAUC']))
print(sum(sal_score_dic['sal']['CC']) / len(sal_score_dic['sal']['CC']))
print(sum(sal_score_dic['sal']['NSS']) / len(sal_score_dic['sal']['NSS']))
print("total model on nonsal text:")
print(sum(sal_score_dic['nonsal']['AUC']) / len(sal_score_dic['nonsal']['AUC']))
print(sum(sal_score_dic['nonsal']['sAUC']) / len(sal_score_dic['nonsal']['sAUC']))
print(sum(sal_score_dic['nonsal']['CC']) / len(sal_score_dic['nonsal']['CC']))
print(sum(sal_score_dic['nonsal']['NSS']) / len(sal_score_dic['nonsal']['NSS']))
print("total model on general text:")
print(sum(sal_score_dic['general']['AUC']) / len(sal_score_dic['general']['AUC']))
print(sum(sal_score_dic['general']['sAUC']) / len(sal_score_dic['general']['sAUC']))
print(sum(sal_score_dic['general']['CC']) / len(sal_score_dic['general']['CC']))
print(sum(sal_score_dic['general']['NSS']) / len(sal_score_dic['general']['NSS']))

print("total model on total text:")
print(sum(sal_score_dic['total']['AUC']) / len(sal_score_dic['total']['AUC']))
print(sum(sal_score_dic['total']['sAUC']) / len(sal_score_dic['total']['sAUC']))
print(sum(sal_score_dic['total']['CC']) / len(sal_score_dic['total']['CC']))
print(sum(sal_score_dic['total']['NSS']) / len(sal_score_dic['total']['NSS']))

