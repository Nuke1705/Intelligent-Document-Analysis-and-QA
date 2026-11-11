
# rag_chain.py

from base64 import b64decode

from langchain_ollama import ChatOllama
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MultimodalRAGChain:

    def __init__(
        self,
        retriever,
        model_name="qwen2.5vl:3b",
        temperature=0.5,
    ):
        self.retriever = retriever

        self.model = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

        # Build reusable chains
        self.chain = self._build_chain()
        self.chain_with_sources = self._build_chain_with_sources()

    @staticmethod
    def parse_docs(docs):
        """Split base64-encoded images and texts"""
        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc["data"])
                b64.append(doc["data"])
            except Exception as e:
                text.append(doc["data"])
        return {"images": b64, "texts": text}

    @staticmethod
    def build_prompt(kwargs):
        """Build a multimodal prompt from retrieved context."""

        docs_by_type = kwargs["context"]
        question = kwargs["question"]

        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text

        prompt_text = f"""
        Answer the question based only on the following context, which can include text, tables, and the below image.

        If the context does not contain enough information,
        state that explicitly. Do not invent facts.

        Text and table context:
        {context_text}

        Question:
        {question}
        """

        prompt_content = [{"type": "text", "text": prompt_text}]

        if len(docs_by_type["images"]) > 0:
               for image in docs_by_type["images"]:
                   prompt_content.append(
                       {
                           "type": "image_url",
                           "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                       }
                   )
       
        return ChatPromptTemplate.from_messages(
            [
                HumanMessage(content=prompt_content),
            ]
        )

    def _build_chain(self):
        """Build the question -> answer chain."""

        return (
            {
                "context": self.retriever
                | RunnableLambda(self.parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.build_prompt)
            | self.model
            | StrOutputParser()
        )

    def _build_chain_with_sources(self):
        """Return retrieved context and generated answer."""

        return (
            {
                "context": self.retriever
                | RunnableLambda(self.parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough().assign(
                response=(
                    RunnableLambda(self.build_prompt)
                    | self.model
                    | StrOutputParser()
                )
            )
        )

    def ask(self, question):
        """Return only the answer."""
        return self.chain.invoke(question)

    def ask_with_context(self, question):
        """Return the parsed context, question, and answer."""
        return self.chain_with_sources.invoke(question)
