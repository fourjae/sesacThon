# 베이스 이미지로 Python 3.9 사용
FROM python:3.7.9

# 작업 디렉토리 설정
WORKDIR /app

# 로컬의 requirements.txt를 컨테이너에 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 전체 복사
COPY . .

# Flask가 0.0.0.0에서 실행되도록 환경변수 설정 (외부 접속 가능하게)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# 컨테이너가 열어놓을 포트
EXPOSE 5000

# Flask 앱 실행
CMD ["flask", "run"]
