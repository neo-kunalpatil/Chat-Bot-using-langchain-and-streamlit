from langchain_core.prompts import PromptTemplate, load_prompt

template= PromptTemplate(template="""
You are a research paper assistant.

Explain the following research paper:
Paper Name: {paper_input}

Explanation Style: {style_input}

Explanation Length: {length_input}

Prompting Technique: {technique_input}

{examples}

Instructions:
1. Explain the main idea of the paper.
2. Describe the problem the paper tries to solve.
3. Explain its key methodology and architecture.
4. Mention important results or contributions.
5. Adapt the explanation to the selected style and length.
6. Use simple language when the Beginner-Friendly style is selected.
If examples are provided, use them as guidance for
the format and level of detail, not as facts about
the selected paper.

Generate the explanation now.
""",
    input_variables=["paper_input", "style_input", "length_input", "technique_input", "examples"])



template.save("template.json")