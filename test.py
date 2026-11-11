from parsePDF import parsepdf
from summarizer import text_table_summary, image_summary
from vector_store import VectorStoreManager
from rag_chain import MultimodalRAGChain


tables, texts, images, formulas, captions = parsepdf("./content/")

text_summaries, table_summaries = text_table_summary(texts, tables)
print("text summary done")
image_summaries = image_summary(images, captions)
print("img summary done")
# Initialize once
db = VectorStoreManager()

# After parsing your PDF and generating summaries:
db.create_db(
    texts=texts,
    text_summaries=text_summaries,
    tables=tables,
    table_summaries=table_summaries,
    images=images,
    image_summaries=image_summaries,
)

# Retrieve relevant parent documents
#docs = db.retrieve("Who are the authors of this paper?")

rag = MultimodalRAGChain(
    retriever=db.retriever,
    model_name="qwen2.5vl:3b",
    temperature=0.5,
)
'''
response = rag.ask(
    "What is attention?"
)

print(response)
'''
'''
result = rag.ask_with_context(
    "Who are the authors of this paper?"
)

print("Answer:")
print(result["response"])

print("\nRetrieved text:")
for text in result["context"]["texts"]:
    print(text)

print("\nNumber of retrieved images:")
print(len(result["context"]["images"]))
'''


while True:
    question = input("\nYou: ").strip()

    if question.lower() in {"exit", "quit"}:
        break

    if not question:
        continue

    try:
        answer = rag.ask(question)
        print(f"\nAssistant: {answer}")

    except Exception as e:
        print(f"Error: {e}")

