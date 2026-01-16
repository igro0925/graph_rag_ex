<<<<<<< HEAD
# graph_rag_ex
Graph RAG에 대한 실습
=======
# graph_rag

# graph DB(neo4j) 활용 영화 추천
## graph RAG 소개
- Graph 데이터베이스를 기반으로 한 질의 응답 시스템(GraphRAG)은 전통적인 벡터 기반 RAG 시스템보다 더 정확하고 연관성 있는 답변을 제공함. 
- 자연어 질문을 Neo4j Cypher 쿼리로 변환하여 지식 그래프를 효과적으로 탐색함.


### 특장점:
- 정확한 관계 검색: 그래프 데이터베이스의 관계 중심 구조를 활용해 복잡한 연결 패턴을 찾을 수 있음.
- 컨텍스트 유지: 엔티티 간의 관계를 유지하여 더 풍부한 컨텍스트를 제공함.
구조화된 정보 검색: 단순 텍스트 검색이 아닌 구조화된 방식으로 정보를 검색함.


# 설치 라이브러리
```
pip install langchain, langchain-neo4j, langchain-openai
pip install pandas
```


전체 흐름 요약
```
1. 그래프 데이터 구축 (CSV → Neo4j)
2. 기본 Cypher 탐색
3. Full-text Search 기반 의미 검색
4. Vector Search 기반 Semantic Search
5. Text-to-Cypher를 통한 자연어 질의
6. Graph + Vector RAG 통합 시스템 완성
```


# 실습 설명
## 실습1
- 파일 : 1.neo4j_movie_graphdb구축_csv.ipynb
### 목표
```
CSV 형식의 영화 데이터를 기반으로 Neo4j GraphDB를 구축하고,
영화–장르–배우–감독 간의 관계를 지식 그래프(Knowledge Graph) 형태로 모델링 및 저장하는 실습
```
### 내용
```
Neo4j 접속을 위한 환경 변수 로딩 및 연결 설정

LangChain을 활용한 Neo4j Graph 객체 생성

Cypher 쿼리를 이용한 연결 테스트(Query Test) 수행

영화 도메인에 대한 온톨로지 정의 및 그래프 스키마 설계
- Movie, Genre, Actor, Director 등의 노드 정의
- ACTED_IN, DIRECTED, HAS_GENRE 등의 관계 설계

CSV 데이터를 로드하여 노드(Node)와 관계(Relationship)로 변환
- 중복 노드 생성을 방지하기 위한 ID 기반 처리
- 배치 처리 방식으로 대용량 데이터 안정적 적재

LangChain의 GraphDocument 객체를 생성하여 Neo4j GraphDB에 저장

전체 그래프 구축 과정의 실행 시간 측정 및 검증
```


## 실습2
- 파일 : 2.neo4j_movie_basic_search.ipynb
### 목표
```
Neo4j에 구축된 영화 GraphDB를 대상으로
Cypher 기반의 기본 검색(Query)과 그래프 탐색 방식을 이해하고,
이를 활용한 그래프 기반 영화 추천 로직의 기초를 학습하는 실습
```
### 내용
```
Neo4j 및 LangChain 연동을 위한 DB 환경 설정

Cypher의 핵심 개념 학습
- 노드(Node)와 관계(Relationship)를 활용한 그래프 패턴 매칭
- MATCH, WHERE, RETURN을 이용한 기본 조회
- 관계 방향(->, <-)과 관계 타입을 활용한 탐색

영화 GraphDB를 활용한 기본 검색 쿼리 작성
- 특정 영화, 배우, 장르 기준 조회
- 다중 관계를 따라가는 경로 기반 탐색

데이터 집계 및 변환
- COUNT, COLLECT 등을 활용한 결과 집계
- 추천 후보 영화 목록 구성

그래프 기반 추천 시스템 기초 구현
- 장르 기반 영화 추천
    - 동일 장르를 공유하는 영화 탐색
    - 자기 자신을 제외한 추천 후보 필터링
- 복합 추천 로직
    - 장르 + 배우 정보 결합
    - 평점(rating) 조건을 활용한 추천 품질 개선
```



## 실습3
- 파일 : 3.neo4j_movie_full-text_search.ipynb
### 목표
```
Neo4j GraphDB에서 제공하는 Full-text Search 기능을 활용하여
정확 일치 검색이 아닌 키워드 기반, 부분 일치, 유사 검색을 수행하고
기본 Cypher 검색 방식과의 차이를 이해하는 실습
```
### 내용
```
Neo4j Full-text Index 개념 학습
- 기존 MATCH 기반 패턴 매칭의 한계 이해
- 텍스트 속성(title, description, overview 등)에 대한 검색 필요성 인식

Full-text Index 생성
- db.index.fulltext.createNodeIndex를 활용한 인덱스 정의
- 영화 노드의 제목, 설명 필드를 대상으로 검색 인덱스 구성

Full-text Search 쿼리 실습
- db.index.fulltext.queryNodes를 활용한 키워드 검색
- 단어 일부 입력 시에도 검색 결과가 반환되는 동작 확인
- 검색 결과에 포함된 score를 통해 유사도 기반 정렬 이해

Full-text Search + Graph 탐색 결합
- 키워드로 검색된 영화 노드를 시작점으로 장르, 배우, 감독 등과 연결된 그래프 구조를 추가 탐색

검색 결과 필터링 및 정렬
- 검색 점수(score) 기준 정렬
- 불필요한 결과 제거 및 추천 후보 정제
```



## 실습4
- 파일 : 4.neo4j_movie_vector_search.ipynb
### 목표
```
Neo4j에 저장된 영화 데이터를 대상으로
Vector Embedding 기반 의미 검색(Vector Search) 을 수행하고,
기존의 키워드 검색(Full-text Search)과 의미 검색의 차이와 장점을 이해하는 실습
```
### 내용
```
Vector Search 개념 학습
- 키워드 일치 기반 검색의 한계 인식
- 문장의 의미(Semantic Similarity)를 벡터 공간에서 비교하는 방식 이해

OpenAI Embedding 모델을 활용한 텍스트 임베딩 생성
- 영화 제목, 설명(overview)을 벡터로 변환
- 동일한 의미를 가진 문장들이 가까운 벡터로 표현됨을 확인

Neo4j Vector Index 생성
- 벡터 차원(dimension) 및 유사도 측정 방식(cosine similarity) 설정
- 영화 노드를 대상으로 벡터 인덱스 구성

Vector Similarity Search 쿼리 실습
- 사용자 질의 문장을 임베딩으로 변환
- Neo4j의 vector.similarity 기반 검색 수행
- 의미적으로 유사한 영화 노드 검색

Vector Search + Graph 확장
- 검색된 영화 노드를 시작점으로 장르, 배우, 감독 등 연결된 그래프 관계를 추가 탐색

검색 결과 비교
- 기본 Cypher 검색 vs Full-text Search vs Vector Search 결과 비교
- 의미 기반 검색이 추천 품질을 어떻게 개선하는지 확인
```



## 실습5
- 파일 : 5.neo4j_movie_text2cypher.ipynb
### 목표
```
사용자의 자연어 질문(Natural Language Query) 을
Neo4j에서 실행 가능한 Cypher 쿼리로 자동 변환(Text-to-Cypher) 하고,
이를 통해 그래프 DB를 LLM 기반으로 질의하는 방식을 학습하는 실습
```
### 내용
```
Text-to-Cypher 개념 이해
- 사용자는 Cypher 문법을 몰라도 자연어로 질문 가능
- LLM이 그래프 스키마를 이해하고 적절한 Cypher 쿼리를 생성

영화 GraphDB 스키마 정보 정의
- 노드 타입(Movie, Genre, Actor, Director 등)
- 관계 타입(ACTED_IN, DIRECTED, HAS_GENRE 등)
- LLM이 올바른 쿼리를 생성하도록 스키마를 명시적으로 제공

LangChain을 활용한 Text-to-Cypher Chain 구성
- 사용자 질문 → 프롬프트 템플릿
- LLM을 통한 Cypher 쿼리 생성

생성된 Cypher 쿼리 실행 및 결과 반환
- Neo4j GraphDB에 쿼리 실행
- 결과를 사람이 이해하기 쉬운 형태로 출력

질문 예제 실습
- 특정 장르의 영화 조회
- 특정 배우가 출연한 영화 목록
- 조건 기반(평점, 장르 등) 영화 탐색

잘못된 쿼리 생성을 방지하기 위한 프롬프트 제어 및 가이드라인 적용
```



## 실습6
- 파일 : 6.neo4j_movie_graphVector_RAG.ipynb
### 목표
```
Neo4j GraphDB의 그래프 구조 탐색(Graph Traversal) 과
벡터 기반 의미 검색(Vector Search) 을 결합한
Graph + Vector RAG 시스템을 구축하고,
LLM이 그래프 컨텍스트를 활용해 정확하고 맥락 있는 답변을 생성하도록 하는 실습
```
### 내용
```
Graph Vector RAG 개념 이해
- 기존 Vector RAG의 한계(관계 정보 부족) 인식
- 그래프 구조를 컨텍스트로 활용하는 Graph RAG의 필요성 이해

사용자 질문 처리 파이프라인 설계
- 자연어 질문을 임베딩으로 변환
- Neo4j Vector Index를 활용한 의미 기반 노드 검색

그래프 확장(Graph Expansion)
- 벡터 검색으로 찾은 핵심 영화 노드를 시작점으로 장르, 배우, 감독 등 연결된 이웃 노드를 탐색
- 그래프 관계를 기반으로 컨텍스트를 구조적으로 확장

RAG 컨텍스트 구성
- 벡터 검색 결과 + 그래프 관계 정보를 결합
- LLM 입력에 적합한 텍스트 형태로 변환

LLM을 활용한 최종 응답 생성
- 그래프에서 추출된 컨텍스트만을 기반으로 답변 생성
- 환각(Hallucination)을 줄이기 위한 컨텍스트 제약 적용

질문 예제 실습
- 특정 장르·분위기에 맞는 영화 추천
- 특정 배우·감독과 관련된 영화 설명
- 단순 검색이 아닌 “이유를 포함한 추천” 응답 확인
```



## 실습7
- 파일 : 7.neo4j_movie_graphRAG_hybrid.ipynb

### Neo4j 기반 하이브리드 RAG
- Vector RAG + Graph RAG를 활용한 서비스 구현하기
>>>>>>> 2e07fc8 (RAG Graph 실습)
