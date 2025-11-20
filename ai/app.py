import os
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import requests
import openai
import re
from dotenv import load_dotenv

# -----------------------
# 환경변수 로드
# -----------------------
load_dotenv()
CLOVA_ID = os.getenv("CLOVA_CLIENT_ID")
CLOVA_SECRET = os.getenv("CLOVA_CLIENT_SECRET")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_KEY


# -----------------------
# 1) CLOVA STT
# -----------------------
def clova_stt(audio_path: str) -> str:
    url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"

    headers = {
        "X-NCP-APIGW-API-KEY-ID": CLOVA_ID,
        "X-NCP-APIGW-API-KEY": CLOVA_SECRET,
        "Content-Type": "application/octet-stream",
    }

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    response = requests.post(url, headers=headers, data=audio_data)
    result = response.json()

    return result.get("text", "").strip()


# -----------------------
# 2) 전처리
# -----------------------
FILLERS = ["음", "어", "저기요", "그니까", "아니", "음...", "어...", "저기"]

def clean_text(text: str) -> str:
    t = text
    for f in FILLERS:
        t = t.replace(f, " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^가-힣0-9a-zA-Z ,.!?]", "", t)
    return t.strip()


# -----------------------
# 3) SBERT 임베딩
# -----------------------
embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def embed_texts(texts):
    return embedder.encode(texts, show_progress_bar=False)


# -----------------------
# 4) HDBSCAN 클러스터링
# -----------------------
def cluster_embeddings(embeddings):
    """여러 건 입력용 (batch)"""
    if len(embeddings) < 3:
        # 3개 미만이면 군집화 불가 → 모두 -1로 처리
        return [-1] * len(embeddings)

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
    return clusterer.fit_predict(emb_scaled)


# -----------------------
# 5) GPT 요약
# -----------------------
def summarize_cluster(texts):
    prompt = f"""
    아래 민원 텍스트들의 공통 주제를 한 줄로 요약하고,
    적절한 카테고리 라벨을 생성해줘.

    {texts}
    """

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content


# -----------------------
# 6) Batch 군집화 (여러 민원)
# -----------------------
def batch_cluster(text_list):
    cleaned = [clean_text(t) for t in text_list]
    embeddings = embed_texts(cleaned)
    labels = cluster_embeddings(embeddings)

    # cluster ID 기준으로 묶기
    clusters = {}
    for text, label in zip(cleaned, labels):
        clusters.setdefault(label, []).append(text)

    results = []
    for cluster_id, items in clusters.items():
        summary = summarize_cluster(items)
        results.append({
            "cluster": int(cluster_id),
            "count": len(items),
            "items": items,
            "summary": summary
        })

    return results


# -----------------------
# 7) 단일 민원 처리
# -----------------------
def single_complaint(audio_path):
    print("📌 STT 변환 중...")
    text_raw = clova_stt(audio_path)

    print("📌 전처리...")
    text_clean = clean_text(text_raw)

    embeddings = embed_texts([text_clean])

    # 단일 민원은 cluster=-1 처리
    label = -1

    summary = summarize_cluster([text_clean])

    return {
        "raw": text_raw,
        "clean": text_clean,
        "cluster": label,
        "summary": summary
    }


# -----------------------
# 8) 자동 분기 처리 (단일 vs 여러 건)
# -----------------------
def process(data, webhook_url=None):
    """
    data:
      - str (음성 파일 경로) → 단일 처리
      - list[str] (민원 텍스트 목록) → batch 처리
    """
    # 단일 음성 파일인 경우
    if isinstance(data, str):
        result = single_complaint(data)

    # 여러 텍스트 (batch)인 경우
    elif isinstance(data, list):
        result = batch_cluster(data)

    else:
        raise ValueError("지원하지 않는 데이터 타입입니다.")

    # webhook 보내기(optional)
    if webhook_url:
        requests.post(webhook_url, json=result)

    return result


# -----------------------
# 실행 예시
# -----------------------
if __name__ == "__main__":

    # 1) 단일 민원 (음성 파일)
    single_result = process("C:/Users/82104/Downloads/tests.m4a")
    print("단일 민원 결과:", single_result)

    # 2) 여러 건(batch) 민원
    batch_data = [
        "지하철 소음 너무 심해요",
        "전철에서 기계음이 계속 납니다",
        "골목 가로등이 꺼져 있습니다",
        "가로등 고장 났어요",
        "버스 시간표 알려주세요"
    ]

    batch_result = process(batch_data)
    print("Batch 민원 결과:", batch_result)
import os
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import requests
import openai
import re
from dotenv import load_dotenv

# -----------------------
# 환경변수 로드
# -----------------------
load_dotenv()
CLOVA_ID = os.getenv("CLOVA_CLIENT_ID")
CLOVA_SECRET = os.getenv("CLOVA_CLIENT_SECRET")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_KEY


# -----------------------
# 1) CLOVA STT
# -----------------------
def clova_stt(audio_path: str) -> str:
    url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"

    headers = {
        "X-NCP-APIGW-API-KEY-ID": CLOVA_ID,
        "X-NCP-APIGW-API-KEY": CLOVA_SECRET,
        "Content-Type": "application/octet-stream",
    }

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    response = requests.post(url, headers=headers, data=audio_data)
    result = response.json()

    return result.get("text", "").strip()


# -----------------------
# 2) 전처리
# -----------------------
FILLERS = ["음", "어", "저기요", "그니까", "아니", "음...", "어...", "저기"]

def clean_text(text: str) -> str:
    t = text
    for f in FILLERS:
        t = t.replace(f, " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^가-힣0-9a-zA-Z ,.!?]", "", t)
    return t.strip()


# -----------------------
# 3) SBERT 임베딩
# -----------------------
embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def embed_texts(texts):
    return embedder.encode(texts, show_progress_bar=False)


# -----------------------
# 4) HDBSCAN 클러스터링
# -----------------------
def cluster_embeddings(embeddings):
    """여러 건 입력용 (batch)"""
    if len(embeddings) < 3:
        # 3개 미만이면 군집화 불가 → 모두 -1로 처리
        return [-1] * len(embeddings)

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
    return clusterer.fit_predict(emb_scaled)


# -----------------------
# 5) GPT 요약
# -----------------------
def summarize_cluster(texts):
    prompt = f"""
    아래 민원 텍스트들의 공통 주제를 한 줄로 요약하고,
    적절한 카테고리 라벨을 생성해줘.

    {texts}
    """

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content


# -----------------------
# 6) Batch 군집화 (여러 민원)
# -----------------------
def batch_cluster(text_list):
    cleaned = [clean_text(t) for t in text_list]
    embeddings = embed_texts(cleaned)
    labels = cluster_embeddings(embeddings)

    # cluster ID 기준으로 묶기
    clusters = {}
    for text, label in zip(cleaned, labels):
        clusters.setdefault(label, []).append(text)

    results = []
    for cluster_id, items in clusters.items():
        summary = summarize_cluster(items)
        results.append({
            "cluster": int(cluster_id),
            "count": len(items),
            "items": items,
            "summary": summary
        })

    return results


# -----------------------
# 7) 단일 민원 처리
# -----------------------
def single_complaint(audio_path):
    print("📌 STT 변환 중...")
    text_raw = clova_stt(audio_path)

    print("📌 전처리...")
    text_clean = clean_text(text_raw)

    embeddings = embed_texts([text_clean])

    # 단일 민원은 cluster=-1 처리
    label = -1

    summary = summarize_cluster([text_clean])

    return {
        "raw": text_raw,
        "clean": text_clean,
        "cluster": label,
        "summary": summary
    }


# -----------------------
# 8) 자동 분기 처리 (단일 vs 여러 건)
# -----------------------
def process(data, webhook_url=None):
    """
    data:
      - str (음성 파일 경로) → 단일 처리
      - list[str] (민원 텍스트 목록) → batch 처리
    """
    # 단일 음성 파일인 경우
    if isinstance(data, str):
        result = single_complaint(data)

    # 여러 텍스트 (batch)인 경우
    elif isinstance(data, list):
        result = batch_cluster(data)

    else:
        raise ValueError("지원하지 않는 데이터 타입입니다.")

    # webhook 보내기(optional)
    if webhook_url:
        requests.post(webhook_url, json=result)

    return result


# -----------------------
# 실행 예시
# -----------------------
if __name__ == "__main__":

    # 1) 단일 민원 (음성 파일)
    single_result = process("C:/Users/82104/Downloads/tests.m4a")
    print("단일 민원 결과:", single_result)

    # 2) 여러 건(batch) 민원
    batch_data = [
        "지하철 소음 너무 심해요",
        "전철에서 기계음이 계속 납니다",
        "골목 가로등이 꺼져 있습니다",
        "가로등 고장 났어요",
        "버스 시간표 알려주세요"
    ]

    batch_result = process(batch_data)
    print("Batch 민원 결과:", batch_result)
