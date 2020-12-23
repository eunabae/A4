import sys
import io
import os
import warnings
from konlpy.tag import Twitter
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log10

np.set_printoptions(threshold=sys.maxsize)

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

warnings.simplefilter("ignore") #경고 문장 삭제
twitter = Twitter()

def findfield(field): #csv 폴더 안에 치과 폴더, 피부과 폴더 구분해서 존재
    dir = "C:/Users/dmsdk/Desktop/대학/3학년/2학기/소프트웨어응용/프로젝트/데이터/csv/"
    #dir = "C:/Users/sena/Desktop/csv/csv/"
    dir_list = os.listdir(dir)
    if field in dir_list:
        return field

#찾으려는 병원 카테고리
hos = findfield('치과')

path = "C:/Users/dmsdk/Desktop/대학/3학년/2학기/소프트웨어응용/프로젝트/데이터/csv/"+hos+"/"
#path = "C:/Users/sena/Desktop/csv/csv/"+hos+"/"
file_list=os.listdir(path)
#print(path)

path2 = "C:/Users/dmsdk/Desktop/대학/3학년/2학기/소프트웨어응용/프로젝트/데이터/address_csv.csv"
#path2 = "C:/Users/sena/Desktop/csv/address_csv.csv"
data = pd.DataFrame(columns=['hospitalID','hospital', 'text'])
count = pd.Series([])
index=0
i=0
reviews=""

for i in file_list:
    file_name = path + i
#   print(i)
    with open(file_name,'r',encoding='ANSI') as xf:
        file = pd.read_csv(xf,sep=',')
        name_and_review = file[['bookingBusinessName','reviewBody']]
        name = file['bookingBusinessName']
        review = file['reviewBody']
        ID=file['bookingBusinessId']
    s =name.values[0]
#    print(type(review))
    size=review.size
    #print(size)

    #리뷰개수
    hID=ID.values[0]
    for j in review:
        reviews=reviews+str(j)
    if(size>25):
        #모든 데이터 사용시 평균이 약 50
        data.loc[index]={'hospitalID':hID,'hospital':s,'text':reviews}
        count.loc[index]=size
        index+=1
    reviews=""

with open(path2,'r',encoding='ANSI') as ad_csv:
    address_csv = pd.read_csv(ad_csv,sep=',')
    hID2 = address_csv['bookingBusinessId']
    city_ = address_csv['구']
    ad_csv.close()
city = pd.DataFrame(data={'hospitalID':hID2,'city_name':city_},columns=['hospitalID','city_name'])

avg = count.mean(axis = 0) # 열 단위 평균
#print(avg) #= 62

#리뷰 수에 따라 가중치 업데이트
def makeweight(avg=avg, count=count):
    weight=[]
    for j in count:
        if j>avg:
            weight.append(2)
        else:
            weight.append(1)
    return weight
#print(type(weight))
def test(weight):
    test=0
    for i in data['hospital']:
        print(i+str(weight[test]))
        test=test+1
#test(makeweight())

#리뷰 가중치로 업데이트
def update_matrix(tfmatrix):
    weight=makeweight()
    for i in range(len(weight)):
        tfmatrix[i,:]=tfmatrix[i,:]*weight[i]
        #print(weight[i])
    return tfmatrix

#코사인 유사도 계산
def cosine_similar(matrix):
    A = matrix
    B = np.transpose(matrix)
    #print(type(A))
    dotpruduct=np.dot(A,B)
    lenA = np.sqrt(np.dot(A,A.T))
    length = np.multiply(lenA,lenA)
    similar = np.divide(dotpruduct,length)
    return np.asarray(similar)

#주소 추출
def address_(filter_address):
    ind = []
    c = []
    filter_city = city[city['city_name']==filter_address]
    pr1 = filter_city['hospitalID'].values
#    print(pr1)
    for i in pr1:
        ind.append(data[data['hospitalID']==i].index)
    for i in ind:
        countind = str(i)
        countind = countind.replace("Int64Index([", "")
        countind = countind.replace("], dtype='int64')", "")
        if countind == '': continue
        countind = int(countind)
        c.append(countind)
    return c

#추소 가중치
def address_update(tfmatrix,filter_address):
    c = address_(filter_address)
    tmp_matrix = tfmatrix.toarray()
    for i in c:
        tmp_matrix[i,:] = tmp_matrix[i,:]*1.5
    tmp_matrix = sparse.csr_matrix(tmp_matrix)
    return tmp_matrix

#TF-IDF계산
def calculate_TFIDF():
    tfidf_TFs = []
    counttok = 0
    tokensall = []
    tokens = []
    for i in data['text']:
        tfidf_TF = []
        tokens.append(twitter.nouns(i))
        token_no_dup = list(set(tokens[counttok])) #중복 지운 병원 별 단어 개수
        total_words_count = len(tokens[counttok])
        word_count = []
        for word in token_no_dup :
            wc = tokens[counttok].count(word)
            if total_words_count == 1:
                word_count.append((1,word))
                continue
            elif wc <= 1:                          #단어가 전체 리뷰에서 한 번 나올경우 제거
                continue

            word_count.append((wc,word))
        word_count = sorted(word_count,reverse=True)
        # print(word_count)
        maxcount = word_count[0][0]
        for str in word_count:
            tfidf_TF.append((0.5+0.5*str[0]/maxcount,str[1]))
            #TF계산
        for j in word_count :
            tokensall.append(j[1])
        tfidf_TFs.append(tfidf_TF)
        tokens[counttok] = token_no_dup
        counttok += 1
    #word_count 순서가 자꾸 뒤바뀌어서 정렬시킴 -> tf 단어 순서가 달라져버림
    tokensall = list(set(tokensall))
    tfidf_IDFs=[]

    D = len(data['text'])
    T = len(tokensall)
    for word in tokensall :
        countdt = 0
        for i in tokens:
            for str in i:
                if word in str:
                    countdt += 1
                    break
        tfidf_IDFs.append((log10(D/(countdt+1))+1,word))
        #IDF계산
        # tfidf_IDFs.append((log10(D/countdt)+1,word))
    tfidf_TFIDF = []
    token_num = 0
    for token in tfidf_IDFs:
        rev_num = 0
        for tf in tfidf_TFs:
            for word in tf:
                if token[1] == word[1]:
                    tfidf_TFIDF.append((rev_num,token_num,token[0]*word[0],token[1]))
                    #TF-IDF 값
            rev_num += 1
        token_num += 1
    # print(tfidf_TFIDF)

    tt = pd.DataFrame(tfidf_TFIDF,columns = ['row','column','TFIDFvalue','word'])
    rowtt = tt['row'].values
    coltt = tt['column'].values
    datatt = tt['TFIDFvalue'].values
    sparse_matrix = sparse.csr_matrix((datatt,(rowtt,coltt)))

    tmp_matrix = sparse_matrix.toarray()
    for i in tmp_matrix:
        l2 = i**2
        l2 = np.sum(l2)
        l2 = np.sqrt(l2)
        i = i / l2
    tmp_matrix = sparse.csr_matrix(tmp_matrix)
#TF-IDF 정규화
#tfidf vec 함수가 return 값이 단순히 np.array 가 아니라 sparse_matrix 형태라 교체
    return tmp_matrix

indices = pd.Series(data.index, index=data['hospital']).drop_duplicates()
#병원이름과 인덱스 연결(인덱스,병원이름)-인덱스객체
# print(indices.head())

#추천
def get_recommendations(keyword,filter_address,index=index):
    data.loc[index,'text']=keyword
    tfidf_matrix = calculate_TFIDF()
    # review에 대해서 tf-idf 수행
    tfidf_matrix = update_matrix(tfidf_matrix)
    tfidf_matrix = address_update(tfidf_matrix,filter_address)
#    print(tfidf_matrix.shape)
    cosine_sim= cosine_similar(tfidf_matrix)
    #코사인 유사도 계산
    #print(type(cosine_sim))
    # 선택한 병원 해당되는 인덱스 받아옴.
    sim_scores = list(enumerate(cosine_sim[index]))
    # 유사도에 따라 정렬
    # print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # 가장 유사한 5개
    sim_scores = sim_scores[1:6]
    # 가장 유사한 5개의 인덱스
    # 0번 인덱스에는 추천에 사용된 값 = 입력 단어에 대한 정보
    hospital_indices = [i[0] for i in sim_scores]
    # 병원이름을 리턴
    return data[['hospitalID','hospital']].iloc[hospital_indices]

print(get_recommendations('충치','중구'))
