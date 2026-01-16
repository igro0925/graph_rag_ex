# movie_recommend.py
import os
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv(override=True)

# -----------------------------
# 1) Neo4j Graph 연결
# -----------------------------
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
    enhanced_schema=True,
)

# -----------------------------
# 2) Vector Index 연결 (Neo4jVector)
# -----------------------------
embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

vector_store = Neo4jVector.from_existing_index(
    embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name=os.getenv("NEO4J_VECTOR_INDEX", "movie_content_embeddings"),
    text_node_property=os.getenv("NEO4J_TEXT_PROPERTY", "overview"),
)

# -----------------------------
# 3) Graph Search 함수들
# -----------------------------
def get_movie_details_and_actors(movie_titles: list[str]):
    query = """
    MATCH (m:Movie)
    WHERE ANY(t IN $titles WHERE m.title CONTAINS t)
    OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
    RETURN 
        m.title as title, 
        m.released as released, 
        m.rating as rating, 
        m.overview as overview,
        collect(a.name) as actor_names,
        collect(elementId(a)) as actor_ids
    """
    return graph.query(query, params={"titles": movie_titles})


def get_actor_filmography(actor_ids: list[str], exclude_titles: list[str]):
    query = """
    MATCH (a:Person)
    WHERE elementId(a) IN $actor_ids
    MATCH (a)-[:ACTED_IN]->(m:Movie)
    WHERE NOT ANY(t IN $exclude_titles WHERE m.title CONTAINS t)
    RETURN a.name as actor, collect(DISTINCT m.title) as other_movies
    """
    return graph.query(query, params={"actor_ids": actor_ids, "exclude_titles": exclude_titles})


def format_context_for_llm(movie_data, film_data) -> str:
    lines = []

    for m in movie_data:
        lines.append(f"- 영화: {m['title']} ({m['released']})")
        lines.append(f"  평점: {m['rating']}")
        if m.get("overview"):
            lines.append(f"  개요: {m['overview']}")
        if m.get("actor_names"):
            actors = [a for a in m["actor_names"] if a]
            if actors:
                lines.append(f"  배우: {', '.join(actors[:10])}")
        lines.append("")

    if film_data:
        lines.append("[배우들의 다른 작품]")
        for f in film_data:
            actor = f.get("actor")
            other_movies = f.get("other_movies", [])
            other_movies = [x for x in other_movies if x]
            if actor and other_movies:
                lines.append(f"- {actor}: {', '.join(other_movies[:10])}")

    return "\n".join(lines)


# -----------------------------
# 4) Orchestrator (Vector → Graph → Format)
# -----------------------------
def movie_graph_search_orchestrator(user_query: str) -> str:
    docs = vector_store.similarity_search(user_query, k=3)
    found_titles = [doc.metadata.get("title") for doc in docs if doc.metadata.get("title")]

    if not found_titles:
        return "관련 정보를 찾을 수 없습니다."

    movie_data = get_movie_details_and_actors(found_titles)

    all_actor_ids = []
    for m in movie_data:
        all_actor_ids.extend(m.get("actor_ids", []))
    all_actor_ids = list(set([x for x in all_actor_ids if x]))

    film_data = []
    if all_actor_ids:
        film_data = get_actor_filmography(all_actor_ids, found_titles)

    return format_context_for_llm(movie_data, film_data)


# -----------------------------
# 5) Main Chain (LLM Answer)
# -----------------------------
def main_chain(query: str) -> str:
    llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"), temperature=0.0)

    template = """
당신은 영화 추천 전문가로서 오직 주어진 정보에 기반하여 객관적이고 정확한 답변을 제공합니다.

[주어진 영화 정보]
{context}

[질문]
{question}

[답변 규칙]
1. 반드시 위 컨텍스트에 있는 정보만 사용하세요.
2. 영화 제목, 평점, 배우 정보가 있으면 포함하세요.
3. 질문이 "추천"이면 컨텍스트 기반으로 추천 근거를 포함하세요.
4. 한국어로 자연스럽고 이해하기 쉽게 답변하세요.
"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_input = {
        "context": RunnableLambda(movie_graph_search_orchestrator),
        "question": RunnablePassthrough(),
    }

    graph_rag_chain = rag_input | prompt | llm | StrOutputParser()

    return graph_rag_chain.invoke(query)
