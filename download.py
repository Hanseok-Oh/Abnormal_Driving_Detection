# API requests
import requests
import urllib.request
from bs4 import BeautifulSoup

# data
import pandas as pd
import numpy as np

# etc
import os
import time
import re
import pytz
import time
from datetime import datetime

#argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", help='다운로드 경로', type=str, nargs='?', default='C:/Users/Yoon/Desktop/프로젝트/이상운전/data/frame/')
parser.add_argument("--num", help='다운로드 개수', type=int, nargs='?', default=105)
parser.add_argument("--road", help='ex(고속도로) / its(국도)', type=str, nargs='?', default='ex')

args = parser.parse_args()
path = args.path
num  = args.num
road = args.road

class Download():
    def __init__(self, path=path, num=num, road=road):
        self.path = path # data 저장 path
        self.num = num # random 다운로드 파일 수

        auth = '1554523699265'
        ex_url = 'http://openapi.its.go.kr:8081/api/NCCTVInfo?key={}&ReqType=2&MinX=124&MaxX=132&MinY=33&MaxY=43&type=ex'.format(auth) # 고속도로 url
        its_url = 'http://openapi.its.go.kr:8081/api/NCCTVInfo?key={}&ReqType=2&MinX=124&MaxX=132&MinY=33&MaxY=43&type=its'.format(auth) # 국도 url

        if road == 'ex':
            self.url = ex_url
        else:
            self.url = its_url

    def request_api(self):
        req = requests.get(self.url)
        html = req.text
        soup = BeautifulSoup(html, 'html.parser')

        # cctv 이름 / url / 좌표 데이터프레임
        name = [i.text for i in soup.findAll("cctvname")]
        url = [i.text for i in soup.findAll("cctvurl")]
        coordx = [i.text for i in soup.findAll("coordx")]
        coordy = [i.text for i in soup.findAll("coordy")]

        self.cctv_df = pd.DataFrame(data={'name':name, 'url':url, 'coordx':coordx, 'coordy':coordy})

    # Timezone 설정
    def set_timezone(self):
        now = datetime.utcnow()
        seoul_tz = pytz.timezone('Asia/Seoul')
        now = pytz.utc.localize(now).astimezone(seoul_tz)
        self.now = now.strftime("%Y%m%d_%H%M_")

    # random하게 cctv 선택
    def random_cctv(self):
        random_idx = np.random.choice(len(self.cctv_df))
        name = self.cctv_df.loc[random_idx, 'name']

        # 파일로 저장될 수 없는 이름 수정
        p = re.compile('[\w\[\]]+')
        regex_name = re.findall(p, name)
        new_name = ''
        for i in range(len(regex_name)):
            new_name += regex_name[i]

        self.cctv_name = new_name
        self.cctv_url = self.cctv_df.loc[random_idx, 'url']

    # 다운로드
    def download_cctv(self):
        # 디렉토리 존재여부 확인
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        urllib.request.urlretrieve(self.cctv_url, self.path + self.now + self.cctv_name + '.mp4')

    # dataset 만들기
    def make_dataset(self):
        self.request_api()
        for _ in range(self.num):
            self.set_timezone()
            self.random_cctv()
            self.download_cctv()
        print('다운로드 완료')

# 실행
if __name__ == '__main__':
    d = Download()
    d.make_dataset()
