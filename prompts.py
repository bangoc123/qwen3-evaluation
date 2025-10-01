PROMPT_SPLIT_STATEMENTS = """
You are given a user question and an AI-generated answer.

### Task
Decompose the answer into a list of **standalone factual statements**.

### Rules
- Do not use pronouns (e.g., "he", "she", "it", "they").
- Each statement must be a complete and understandable sentence that conveys **one atomic factual idea**.
- Treat a statement as **atomic** when it describes properties that belong to the same entity at the same time (example: "The main camera has 50MP resolution and an f/1.8 aperture." → keep as one statement).
- If a statement expresses **alternatives, options, or multiple versions** (example: "The phone has 50MP main camera with f/1.8 aperture, 5MP ultra-wide camera with f/2.2 aperture."), split into separate statements (→ "The phone has 50MP main camera with f/1.8 aperture." and "The phone has 5MP ultra-wide camera with f/2.2 aperture.").
- Do not merge distinct entities or features into a single statement.
- Provide a short `"reason"` for each statement that explains why it is one atomic idea.
- The language of the statement must be in the same language as the question.

### Output format
Return the result in the following JSON format:
{{
  "statements": [
    {{
      "statement": "...",
      "reason": "..."
    }},
    ...
  ]
}}

### Example

**Question:**  
What are the key features of the display on the TCL 40 NXTPAPER 8GB/256GB mobile phone?

**Answer:**  
the display of the "TCL 40 NXTPaper 8GB/256GB" phone stands out with the following features:

90Hz refresh rate  
Resolution of 2460x1080 pixels  
50MP main camera with f/1.8 aperture, 5MP ultra-wide camera with f/2.2 aperture, 2MP macro camera with f/2.4 aperture  
256GB of storage, 8GB of RAM.  
These detailed specifications are taken from the first product listing.

**Output:**
{{
  "statements": [
    {{
      "statement": "The display of the TCL 40 NXTPaper 8GB/256GB phone has a 90Hz refresh rate.",
      "reason": "This is a single specification about the display refresh rate."
    }},
    {{
      "statement": "The display of the TCL 40 NXTPaper 8GB/256GB phone has a resolution of 2460x1080 pixels.",
      "reason": "This is a single specification about the display resolution."
    }},
    {{
      "statement": "The TCL 40 NXTPaper 8GB/256GB phone has a 50MP main camera with an f/1.8 aperture.",
      "reason": "This combines resolution and aperture which describe the same entity (main camera)."
    }},
    {{
      "statement": "The TCL 40 NXTPaper 8GB/256GB phone has a 5MP ultra-wide camera with an f/2.2 aperture.",
      "reason": "This combines resolution and aperture which describe the same entity (ultra-wide camera)."
    }},
    {{
      "statement": "The TCL 40 NXTPaper 8GB/256GB phone has a 2MP macro camera with an f/2.4 aperture.",
      "reason": "This combines resolution and aperture which describe the same entity (macro camera)."
    }},
    {{
      "statement": "The TCL 40 NXTPaper 8GB/256GB phone has 256GB of internal storage.",
      "reason": "This is a single specification about storage capacity."
    }},
    {{
      "statement": "The TCL 40 NXTPaper 8GB/256GB phone has 8GB of RAM.",
      "reason": "This is a single specification about RAM."
    }}
  ]
}}

---

Input:
Question: {}
Answer: {}
"""

F1_SCORE_LABEL_STATEMENT_PROMPT = """
                    You are given a user question, the list AI-generated statements of answer, and ground_truth.

                    1. **Evaluate accuracy:**
                    For each extracted AI-generated statements of answer, determine whether it is supported by ground_truth:
                    - Assign `"verdict": 1` if the statement can be inferred from the ground_truth.
                    - Assign `"verdict": 0` if the statement cannot be inferred.
                    - Provide a short `"reason"` for each verdict.

                    Return the result in the following JSON format:
                    {{
                    "statements": [
                        {{
                        "statement": "...",
                        "reason": "...",
                        "verdict": 1 or 0
                        }},
                        ...
                    ]
                    }}
                    ---

                    ### ✅ Example

                    **Question:**  
                    Who is Marie Curie and what is she famous for?

                    **Answer:**  
                    ["Marie Curie was a physicist.", "Marie Curie was a chemist.", "Marie Curie won two Nobel Prizes.", "Marie Curie discovered radium.", "Marie Curie discovered polonium.", "Marie Curie taught at Sorbonne University in Paris."]

                    **Ground Truth:**  
                    Marie Curie was a pioneering physicist and chemist who conducted research on radioactivity. She was awarded two Nobel Prizes: one in Physics and one in Chemistry. She is known for discovering the radioactive elements radium and polonium.

                    **Output:**
                    {{
                    "statements": [
                        {{
                        "statement": "Marie Curie was a physicist.",
                        "reason": "The ground truth confirms that Marie Curie was a physicist.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie was a chemist.",
                        "reason": "The ground truth confirms that Marie Curie was a chemist.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie won two Nobel Prizes.",
                        "reason": "The ground truth confirms that Marie Curie won two Nobel Prizes in Physics and Chemistry.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie discovered radium.",
                        "reason": "The ground truth clearly states that she discovered the radioactive elements radium and polonium.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie discovered polonium.",
                        "reason": "The ground truth clearly states that she discovered the radioactive elements radium and polonium.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie taught at Sorbonne University in Paris.",
                        "reason": "The ground truth does not mention anything about Marie Curie teaching at the Sorbonne or any other university.",
                        "verdict": 0
                        }}
                    ]
                    }}

                    ---

                    Input:
                    Question: {}

                    Answer: {}

                    Ground Truth: {}
                    """

GROUNDEDNESS_LABEL_STATEMENT_PROMPT = """
                    You are a helpful and harmless AI assistant. You will be provided with a question, a textual
                    context and the list model-generated statements of response.
                    Your task is to analyze the list statements of response and classify each
                    statement according to its relationship with the provided context.

                    **Instructions:**
                    1. **For each statement, assign one of the following labels:**
                        * **`supported`**: The statement is entailed by the given context.  Provide a
                        supporting excerpt from the context. The supporting excerpt must *fully*
                        entail the statement. If you need to cite multiple supporting excerpts,
                        simply concatenate them.
                        * **`unsupported`**: The statement is not entailed by the given context. No
                        excerpt is needed for this label.
                        * **`contradictory`**: The statement is falsified by the given context.
                        Provide a contradicting excerpt from the context.
                        * **`no_rad`**: The statement does not require factual attribution (e.g.,
                        opinions, greetings, questions, disclaimers).  No excerpt is needed for
                        this label.
                    2. **For each label, provide a short rationale explaining your decision.**
                    The rationale should be separate from the excerpt.
                    3. **Be very strict with your `supported` and `contradictory` decisions.**
                    Unless you can find straightforward, indisputable evidence excerpts *in the
                    context* that a sentence is `supported` or `contradictory`, consider it
                    `unsupported`. You should not employ world knowledge unless it is truly
                    trivial.

                    **Input Format:**

                    The input will consist of three parts, clearly separated:
                    * **Question:** The original user query that the model is supposed to answer, typically related to the given context.,
                    * **Context:**  The textual context used to generate the response.
                    * **Response:** The list model-generated statements of response.

                    **Output Format:**
                    Return the result in the following JSON format:
                    {{
                    "statements": [
                        {{
                        "sentence": "...",
                        "label": "...",
                        "rationale": "...",
                        "excerpt": "..."
                        }},
                        ...
                    ]
                    }}
                    For each sentence in the response, output with the following
                    fields:

                    * `"sentence"`: The sentence being analyzed.
                    * `"label"`: One of `supported`, `unsupported`, `contradictory`, or `no_rad`.
                    * `"rationale"`: A brief explanation for the assigned label.
                    * `"excerpt"`:  A relevant excerpt from the context. Only required for
                    `supported` and `contradictory` labels.

                    **Example:**

                    **Input:**

                    ```
                    Question:
                    What are the colors of apples and bananas?
                    
                    Response:
                    [Apples are red, Bananas are green, Bananas are cheaper than apples,Enjoy your fruit!]

                    Context:
                    Apples are red fruits. Bananas are yellow fruits.

                    ```

                    **Output:**
                    {{
                    "statements":
                        [
                        {{"sentence": "Apples are red.", "label": "supported", "rationale": "The context explicitly states that apples are red.", "excerpt": "Apples are red fruits."}}
                        {{"sentence": "Bananas are green.", "label": "contradictory", "rationale": "The context states that bananas are yellow, not green.", "excerpt": "Bananas are yellow fruits."}}
                        {{"sentence": "Bananas are cheaper than apples.", "label": "unsupported", "rationale": "The context does not mention the price of bananas or apples.", "excerpt": null}}
                        {{"sentence": "Enjoy your fruit!", "label": "no_rad", "rationale": "This is a general expression and does not require factual attribution.", "excerpt": null}}
                        ]
                    }}
                    **Now, please analyze the following context and response:**
                    
                    **Question:**
                    {}

                    **Response:**
                    {}

                    **Context:**
                    {}
                    
                    """

NOISESENSITIVY_LABEL_STATEMENT_PROMPT = """
                    You are given a user question, the list AI-generated statements of answer, and a reference context.

                    1. **Evaluate Faithfulness:**
                    For each extracted statement, determine whether it is supported by the context:
                    - Assign `"verdict": 1` if the statement can be directly inferred from the context.
                    - Assign `"verdict": 0` if the statement cannot be directly inferred.
                    - Provide a short `"reason"` for each verdict.

                    Return the result in the following JSON format:
                    {{
                    "statements": [
                        {{
                        "statement": "...",
                        "reason": "...",
                        "verdict": 1 or 0
                        }},
                        ...
                    ]
                    }}
                    ---

                    ### ✅ Example

                    **Question:**  
                    Who is Marie Curie and what is she famous for?

                    **Answer:**  
                    ["Marie Curie was a physicist.", "Marie Curie was a chemist.", "Marie Curie won two Nobel Prizes.", "Marie Curie discovered radium.", "Marie Curie discovered polonium.", "Marie Curie taught at Sorbonne University in Paris."]

                    **Context:**  
                    Marie Curie was a pioneering physicist and chemist who conducted research on radioactivity. She was awarded two Nobel Prizes: one in Physics and one in Chemistry. She is known for discovering the radioactive elements radium and polonium.

                    **Output:**
                    {{
                    "statements": [
                        {{
                        "statement": "Marie Curie was a physicist.",
                        "reason": "The ground truth confirms that Marie Curie was a physicist.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie was a chemist.",
                        "reason": "The ground truth confirms that Marie Curie was a chemist.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie won two Nobel Prizes.",
                        "reason": "The ground truth confirms that Marie Curie won two Nobel Prizes in Physics and Chemistry.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie discovered radium.",
                        "reason": "The ground truth clearly states that she discovered the radioactive elements radium and polonium.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie discovered polonium.",
                        "reason": "The ground truth clearly states that she discovered the radioactive elements radium and polonium.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie taught at Sorbonne University in Paris.",
                        "reason": "The ground truth does not mention anything about Marie Curie teaching at the Sorbonne or any other university.",
                        "verdict": 0
                        }}
                    ]
                    }}
                    ---

                    Input:
                    Question: {}

                    Answer: {}

                    Context: {}
                    """