import scipy as sp
import pandas as pd
import numpy as np
import sys
import io
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from konlpy.tag import Twitter
import os

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
matplotlib.rcParams.update({'font.size': 10})

warnings.filterwarnings('ignore')
#분석위한 전체 데이터
with open("C:/Users/dmsdk/Desktop/대학/3학년/2학기/소프트웨어응용/프로젝트/데이터/피부과.csv",'r',encoding='ANSI') as xf:
    data = pd.read_csv(xf,sep=',')
    star_and_review = data[['score','reviewBody']]
    score_org = data['score']
    review = data['reviewBody']
    hostital= data['bookingBusinessName']

#data=pd.read_csv('C:/Users/dmsdk/Desktop/대학/3학년/2학기/소프트웨어응용/프로젝트/데이터/네이버_csv.csv')
print(data.shape)

#별점 분석
count = score_org.value_counts(sort=True, ascending=False)

count.plot(kind='bar', figsize=(8,4),color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
plt.title("score counts")
plt.xlabel("score")
plt.ylabel("counts")
plt.show()

#병원 리뷰 분포 분석
hostitalrate= hostital.value_counts(sort=True, ascending=False)
hostitalrate.plot(kind='pie', figsize=(20,20))
plt.title("hospital")
plt.xlabel("")
plt.ylabel("")
plt.show()

"""
#추천 결과 평가 위해 - 단어 비율 확안 => 추천 키워드 등장횟수 / 리뷰에 등장하는 전체 키워드
twitter = Twitter()
path = "C:/Users/dmsdk/Desktop/대학/3학년/2학기/소프트웨어응용/프로젝트/데이터/csv/치과/"
file_list=os.listdir(path)
data = pd.DataFrame(columns=['hospitalID','hospital', 'text'])
count = pd.Series([])

index=0
i=0
reviews=""

#병원 별로 리뷰 모음
for i in file_list:
    file_name = path + i
#   print(i)
    with open(file_name,'r',encoding='ANSI') as xf:
        file = pd.read_csv(xf,sep=',')
        name_and_review = file[['bookingBusinessName','reviewBody']]
        name = file['bookingBusinessName']
        review = file['reviewBody']
        ID=file['bookingBusinessId']
#    print (s[1])
#    print(type(review))
    size=review.size
#    print(size)
    hID=ID.values[0]
    hName=name.values[0]
#    print(hName)
    #리뷰개수
    for j in review:
        reviews=reviews+str(j)
    if(size>20):
        data.loc[index]={'hospitalID':hID,'hospital':hName,'text':reviews}
        count.loc[index]=size
        index+=1
    reviews=""
#리뷰 합치기 종료

indices = pd.Series(data.index, index=data['hospital']).drop_duplicates()

#추천 테스트
tokens = []
rank={}
#추천 키워드
recommendkey = '사랑니'
count=0
#병원 리뷰에 존재하는 키워드 분리
for i in data['text']:
    keycount=0
    tokencount=0
    hos=data.iloc[count,1]
    tokens.append(twitter.nouns(i))
    #print(tokens)
#해당 병원에 존재하는 키워드 = tokens
    for re in tokens: #한 토큰에 여러 키워드 뭉치
        for word in re: #각 키워드 뭉치의 한 단어마다
            tokencount+=1 #검사한 한 단어에 대해=검사횟수
            #print(word)
            if(word == recommendkey):
                keycount+=1
            #병원에 존재하는 키워드의 수를 구함
        #print (keycount)
        #print (tokencount)
    #병원에 등장한 키워드 수 구함
    rank[hos] = keycount/tokencount
    count+=1
    tokens=[]
#모든 병원에 대해 계산후 등수 확인
rank=sorted(rank.items(), key=lambda x:x[1],reverse=True)
print(type(rank))
for result in rank[:10]:
    print(result)
#print(rank[:10])
"""
