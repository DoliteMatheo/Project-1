#coding:utf8
import jieba
def cut(essay_str):return list(jieba.cut(str(essay_str)))
import pandas as pd

content=pd.read_csv('C:/Users/trans02/AppData/Local/Programs/Python/Python36/Scripts/dataset/sqlResult_1558435.csv',encoding='gb18030')
#测试集
convals=content['content'].values
convals_list=convals.tolist()

train_num=4000
test_num=400
all_num=train_num+test_num

kind_list=[]
for k in range(int(0.5*train_num)):kind_list.append(0)
for k in range(int(0.5*train_num)):kind_list.append(1)

xhs_num=0
oth_num=0
xhs_content=[]
oth_content=[]
for k in range(len(content)):
	if content['source'].values[k]=='新华社':#把1000个新华社的文章放入xhs_content中
		if xhs_num>=0.5*all_num:continue
		xhs_content.append(content['content'].values[k])
		xhs_num += 1
	else:#把1000个不是新华社的文章放入oth_content中
		if oth_num>=0.5*all_num:continue
		oth_content.append(content['content'].values[k])
		oth_num += 1

corpus_train=oth_content[:int(0.5*train_num)]+xhs_content[:int(0.5*train_num)]
corpus_test=oth_content[int(0.5*train_num):]+xhs_content[int(0.5*train_num):]
corpus_all=corpus_train+corpus_test

#清洗程序
def wash(str_original,ls_delete):
    for k in range(len(ls_delete)):
        str_original=str(str_original).replace(ls_delete[k],'')
    return str_original
#要清洗删除的字符
ls_delete=['\u3000','\\n',',','。',':','：','，','“','”','（','）','《','》','！','!','、','(',')','·']


#执行清洗
corpus=[wash(str,ls_delete) for str in corpus_all]
#分词并隔开
essay=[' '.join(cut(s)) for s in corpus]
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
victorize=TfidfVectorizer()#定义一个向量生成器
x=victorize.fit_transform(essay)#是一个csr_matrix类型的变量
x_ndarray=x.toarray()#是一个以原始数据每篇文本为行，以每个词（所有文本中出现过的词）为列的TF-IDF值的矩阵

mat_train=x_ndarray[:train_num]
mat_test=x_ndarray[train_num:]

from sklearn.naive_bayes import MultinomialNB
modle=MultinomialNB()
modle.fit(mat_train,kind_list)

lst_predict=modle.predict(mat_test)
print(lst_predict)

error_num1=[]
error_num2=[]
for k in range(int(0.5*test_num)):
    if lst_predict[k]!=0:error_num1.append(k)#将非的判定为新华社的
for k in range(int(0.5*test_num),test_num):
    if lst_predict[k]!=1:error_num2.append(k)#将新华社判定为非的
print(error_num1)
print(error_num2)
precision=1-((len(error_num1)+len(error_num2))/test_num)
print('Precision：'+str(precision))

print('将非的判定为新华社的：')
for k in error_num1:
    print(corpus_test[k])
    print('*'*40)
'''
print('将新华社判定为非的：')
for k in error_num2:
    print(corpus_test[k])
    print('*' * 40)    

'''
import numpy as np
def cos_tfidf(str1,str2):
    corpus=[str1,str2]
    essay=[' '.join(cut(s)) for s in corpus]
    vectorizer=TfidfVectorizer()
    vec=vectorizer.fit_transform(essay)
    ls=vec.toarray()
    vec1,vec2=ls[0],ls[1]
    return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))

print('余弦距离比对')

dic={k:cos_tfidf(corpus_test[162],convals_list[k]) for k in range(len(convals_list))}
'''
for k,v in dic.items():
	if v!=v:dic[k]=0

#sorted_ls=sorted(dic.items(),key=lambda xx:xx[1],reverse=True)#按余弦相似度从大到小排序
lss=[v for k,v in dic.items()]
max_v=max(lss)
[k for k,v in dic.items() if v==max_v]
similar_num,similar_cos=sorted_ls[0]
print('最相近的文章是第'+str(similar_num)+'篇')
print('相似度为：'+str(similar_cos))
print('相似文章内容为：'+corpus_train[similar_num])
'''