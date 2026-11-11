import base64
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

#Prompt
prompt_text = """
You are an assistant tasked with summarizing tables and text
from scientific research papers.

Give a concise, accurate summary of the supplied table or text.
Preserve technical terminology, numerical values, and key findings, mention authors of the paper when required.

Respond only with the summary.

Table or text chunk:
{element}
"""

#Summary chain
model = ChatOllama(
    model="qwen2.5vl:3b",
    temperature=0.5
)

def text_table_summary(texts, tables):
    
    prompt = ChatPromptTemplate.from_template(prompt_text)

    summarize_chain = (
        {"element": lambda x: x}
        | prompt
        | model
        | StrOutputParser()
    )

    text_summaries = summarize_chain.batch(
        [text["data"] for text in texts],
        config={"max_concurrency": 3}
    )

    tables_html = [
        table["data"].metadata.text_as_html
        for table in tables
    ]

    table_summaries = summarize_chain.batch(
        tables_html,
        config={"max_concurrency": 2}
    )

    return text_summaries, table_summaries

prompt_template = """
You are analyzing a figure extracted from a scientific research paper.

Your task is to produce a scientifically accurate description of the image, using the provided figure caption as essential context.

FIGURE CAPTION:
{caption}

INSTRUCTIONS:

1. Read and interpret the figure caption BEFORE analyzing the image. The caption describes the scientific purpose, subject, and context of the figure. Use this information to guide your interpretation of the visual content.

2. Identify the type of figure: plot, diagram, schematic, photograph, equation, table, or a combination of these.

3. Connect the visible content to the caption:
   - Identify the objects, physical quantities, processes, or theoretical concepts mentioned in the caption that are actually visible in the image.
   - Explain how the visible elements relate to the scientific subject described in the caption.
   - Use terminology from the caption where appropriate.
   - If the caption specifies a physical process, model, experimental setup, or theoretical relationship, interpret the image in that context.

4. If the image contains graphs, describe the axes, variables, units, curves, legends, trends, important features, and relationships. Use the caption to interpret what the plotted quantities represent. Do not invent axis labels, numerical values, or trends that cannot be established from the image.

5. If the image contains a schematic or scientific diagram, describe its components, labels, arrows, connections, and the physical or mathematical process they represent, informed by the caption.

6. If the image contains multiple panels, describe each panel separately and explain how the panels relate to the caption and to each other.

7. Distinguish between information explicitly provided by the caption and information directly observable in the image. Do not invent scientific conclusions that are unsupported by either source.

8. If the caption describes something that is not clearly visible, acknowledge the uncertainty rather than ignoring the caption or hallucinating visual details.

OUTPUT REQUIREMENTS:
Write a detailed, coherent scientific description suitable for indexing and retrieval in a retrieval-augmented generation (RAG) system. Prioritize scientific meaning and relationships over superficial visual appearance.

Return only the image summary. Do not repeat these instructions.
"""

def summarize_image(image, caption):
    # Strip the data URL prefix if present
    if image.startswith("data:"):
        header, image = image.split(",", 1)
        mime_type = header.split(";")[0].split(":")[1]
    else:
        mime_type = "image/jpeg"

    # Validate that the image is base64
    base64.b64decode(image, validate=True)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt_template.format(
                    caption=caption or ""
                )
            },
            {
                "type": "image",
                "base64": image,
                "mime_type": mime_type
            }
        ]
    )

    response = model.invoke([message])
    return StrOutputParser().invoke(response)

def image_summary(images, captions):
    assert len(images) == len(captions)

    image_summaries = [
        summarize_image(image["data"], caption["data"])
        for image, caption in zip(images, captions)
    ]

    return image_summaries