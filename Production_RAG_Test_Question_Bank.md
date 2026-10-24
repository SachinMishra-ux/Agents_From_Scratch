# Production RAG & LangGraph Assessment
### 15 Straightforward Questions with Code Snippets & Answer Key

---

**Q1. When deploying Google's Gemini embedding models in an enterprise production RAG system, what is a primary consideration when choosing between a public preview model (e.g., 'models/gemini-embedding-2') and an enterprise-provisioned embedding service (e.g., Azure OpenAI Embeddings or Google Vertex AI)?**

- **[A]** Preview models lack cosine similarity support, whereas enterprise models support vector arithmetic.
- **[B]** Enterprise-provisioned services provide guaranteed SLAs, private VPC networking, zero data retention for training, and immutable version pinning to prevent embedding drift.
- **[C]** Gemini embedding models can only produce binary (0 or 1) vector representations.
- **[D]** Enterprise embedding models do not enforce any request-per-minute (RPM) or token rate limits.

---

**Q2. Consider the following code snippet from the repository used to resolve a batch serialization bug in langchain-google-genai:

What is the primary operational trade-off of this code implementation in production?**

```python
class PatchedGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Workaround for batch embedding bug
        return [self.embed_query(t) for t in texts]

embeddings = PatchedGoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
```

- **[A]** It alters the vector dimensionality from 768 to 1536, breaking FAISS compatibility.
- **[B]** It converts concurrent batch requests into sequential single-item HTTP calls, increasing total ingestion time and consuming RPM quota faster.
- **[C]** It causes Google AI Studio to reject queries because embed_query requires a different API key.
- **[D]** It stores all embedded vectors in a local Redis cache automatically.

---

**Q3. In the vectorstore setup, FAISS indexes are loaded using:

vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

Why does LangChain mandate setting allow_dangerous_deserialization=True, and what risk does it carry in production?**

- **[A]** It allows FAISS to exceed 32 GB of system RAM without operating system warnings.
- **[B]** FAISS uses Python's pickle module to deserialize index metadata; unpickling an untrusted or tampered file can execute arbitrary code (Remote Code Execution) on the server.
- **[C]** It automatically disables SSL/TLS encryption for all network traffic.
- **[D]** It converts dense floating-point vectors into uncompressed plaintext strings.

---

**Q4. In RAG_embedding/embedding.py, document chunks are embedded in small batches of 5 with an explicit delay (time.sleep(10)) before calling vectorstore.merge_from(db). Why is this pattern necessary for production document ingestion?**

- **[A]** To prevent HTTP 429 (Too Many Requests / Quota Exhaustion) errors from embedding APIs and ensure partial ingestion progress is preserved.
- **[B]** To allow the operating system to clear GPU video memory after each batch.
- **[C]** Because FAISS crashes if it indexes more than 5 documents in a single index.
- **[D]** To give the PDF text loader enough time to parse the next set of pages.

---

**Q5. In the codebase, documents are chunked using chunk_size_words=2500 (~15,000 characters). When building a financial RAG system querying specific metrics (e.g., quarterly revenue figures), what is the main risk of such large chunks?**

- **[A]** RecursiveCharacterTextSplitter will throw an uncaught integer overflow exception.
- **[B]** Semantic dilution: pinpoint numerical data is averaged over 2,500 words, lowering vector similarity scores and increasing LLM context distraction.
- **[C]** FAISS indexes cannot store any vector computed from text longer than 512 characters.
- **[D]** It forces the embedding model to drop all numerical digits from the text.

---

**Q6. In a LangGraph StateGraph, how does an individual node function (e.g., question_rewriter or retrieve) update the centralized AgentState?**

- **[A]** It modifies a global Python variable directly without returning anything.
- **[B]** It returns a dictionary containing only the modified state keys, which LangGraph merges into the shared state based on defined channel reducers.
- **[C]** It must serialize the entire state dictionary to a JSON file on disk.
- **[D]** It writes raw SQL UPDATE statements to the SQLite checkpoint database.

---

**Q7. Why is a question_rewriter node placed immediately before the vector retrieval step in a conversational multi-turn RAG pipeline?**

- **[A]** Vector databases require queries to be written strictly in uppercase letters.
- **[B]** Follow-up questions often contain pronouns or implicit context (e.g., 'What was their profit?'); rewriting reformulates them into standalone, unambiguous queries optimized for vector search.
- **[C]** It translates natural language into structured SQL syntax for FAISS.
- **[D]** It removes all question marks so that the query matches document titles.

---

**Q8. Consider the following LangGraph conditional routing snippet from Advance_RAG/workflow.py:

Which LangGraph method call correctly connects this conditional router to the question_classifier node?**

```python
def on_topic_router(state: AgentState):
    on_topic = state.get("on_topic", "").strip().lower()
    if on_topic == "yes":
        return "retrieve"
    else:
        return "off_topic_response"

# How to connect to question_classifier node in StateGraph 'g'?
```

- **[A]** g.add_edge('question_classifier', on_topic_router)
- **[B]** g.add_conditional_edges('question_classifier', on_topic_router, {'retrieve': 'retrieve', 'off_topic_response': 'off_topic_response'})
- **[C]** g.set_entry_point(on_topic_router)
- **[D]** g.add_node('on_topic_router', ['retrieve', 'off_topic_response'])

---

**Q9. In the retrieval_grader node, relevance scoring uses llm.with_structured_output(GradeDocument) where GradeDocument is a Pydantic model with a binary score. Why is this preferred over free-form prompt engineering (e.g., 'Reply with YES or NO')?**

- **[A]** Free-form strings always crash the Python runtime with a TypeError.
- **[B]** Structured output guarantees schema validation via API-level tool/function calling, avoiding failures caused by conversational filler, explanations, or irregular capitalization.
- **[C]** Structured output runs locally on the CPU without consuming LLM API tokens.
- **[D]** It enables the LLM to search the internet without an external API wrapper.

---

**Q10. In LangGraph cyclical workflows (e.g., refining queries when documents are graded irrelevant), what safeguard prevents an infinite loop if the exit condition fails to trigger?**

- **[A]** The operating system forcibly kills the process after 60 seconds.
- **[B]** LangGraph enforces a recursion_limit (default 25 steps) configured in invoke/stream, raising a GraphRecursionError if exceeded.
- **[C]** FAISS automatically deletes the vector index upon 5 failed queries.
- **[D]** The Python garbage collector terminates all while loops after 100 iterations.

---

**Q11. In Corrective RAG (CRAG), as implemented in Advance_RAG/workflow.py, what corrective action occurs when local document retrieval and query refinement fail to yield relevant documents?**

- **[A]** The system aborts and returns an unhandled 500 error to the user.
- **[B]** The graph routes to a search_internet node (using a web search tool like SerpAPI) to retrieve fresh external web results.
- **[C]** The system automatically retrains the embedding model on the fly.
- **[D]** It prompts the user to enter their personal database credentials.

---

**Q12. When comparing Self-RAG and Corrective RAG (CRAG) for an air-gapped financial environment where external internet access is strictly prohibited and hallucinations carry heavy regulatory penalties, which architecture is preferred and why?**

- **[A]** CRAG, because it always requires live Google search fallback.
- **[B]** Self-RAG, because its internal reflection checkpoints (e.g., is_sup / hallucination check) verify that answers are strictly grounded in internal documents without outbound web reliance.
- **[C]** Neither, because both require open internet access to run vector similarity.
- **[D]** CRAG, because Self-RAG only works on non-financial documents.

---

**Q13. In Advance_RAG/workflow.py, role-based access control (RBAC) is implemented by retrieving top-K documents and then filtering doc.metadata['role'] in Python. What is the primary production flaw of this post-retrieval filtering?**

- **[A]** Python dictionaries cannot store string metadata keys.
- **[B]** Top-K Starvation: If the top-K retrieved chunks belong to unauthorized roles, post-filtering leaves zero documents for generation; filtering should instead be pushed into the vector store as pre-filtering.
- **[C]** FAISS throws a C++ segmentation fault whenever doc.metadata is accessed in Python.
- **[D]** Post-filtering exposes user credentials in the application logs.

---

**Q14. In Advance_RAG/new_workflow.py, when len(state['messages']) > 8, a summarize_conversation node condenses prior messages into state['summary'] and retains only the last 2 messages. What production problem does this solve?**

- **[A]** It prevents context window overflow, reduces inference latency, and caps LLM token costs in long multi-turn sessions while retaining conversation context.
- **[B]** It converts chat messages into FAISS vector files automatically.
- **[C]** It stops the user from asking any questions with more than 8 words.
- **[D]** It encrypts the chat history using AES-256 for database storage.

---

**Q15. For a multi-container, horizontally scaled production deployment of LangGraph (e.g. running on Kubernetes), why is SqliteSaver replaced with a database checkpointer like PostgresSaver?**

- **[A]** SQLite cannot store strings containing numbers or punctuation.
- **[B]** SQLite is an in-process, single-file database that suffers from write-lock contention under concurrency and cannot be shared across stateless pod replicas; PostgresSaver provides distributed, concurrent state storage.
- **[C]** SqliteSaver is limited to a maximum of 10 conversational turns per day.
- **[D]** PostgresSaver eliminates the need for vector embeddings in RAG systems.

---

## Answer Key & Explanations

| Question | Correct Answer | Explanation |
| :--- | :---: | :--- |
| **Q1** | **[B]** | Enterprise services provide formal SLAs, private VPC endpoints, compliance certifications, and static model version pinning. Public preview models may experience silent updates or breaking changes, which cause embedding drift and invalidate pre-built vector databases. |
| **Q2** | **[B]** | Overriding embed_documents with a list comprehension over embed_query fixes client-side batch formatting bugs by issuing N individual HTTP requests. However, this serializes the requests, increasing network latency and consuming per-minute request limits (RPM). |
| **Q3** | **[B]** | FAISS serializes docstore metadata into a Python .pkl (pickle) file. Deserializing untrusted pickle files exposes the host to arbitrary bytecode execution (RCE). In production, indexes must be cryptographically verified or loaded from trusted, locked storage. |
| **Q4** | **[A]** | Embedding APIs enforce strict Request-Per-Minute (RPM) and Token-Per-Minute (TPM) quotas. Bulk ingestion without batching and rate-limiting leads to 429 errors that fail the entire job. Incremental batching with sleep prevents rate-limit errors and saves intermediate progress. |
| **Q5** | **[B]** | Averaging 2,500 words into a single embedding vector dilutes fine-grained semantic details (like financial metrics), hurting retrieval accuracy. Smaller chunks (e.g., 200–500 words) or parent-document retrieval yield higher precision for fact-based lookup. |
| **Q6** | **[B]** | LangGraph nodes receive the current state and return a dictionary of partial updates. LangGraph updates the state channels by either overwriting values or applying reducers (like operator.add for lists). |
| **Q7** | **[B]** | In multi-turn chat, users frequently ask ambiguous follow-up questions ('How did that change in Q4?'). Without rewriting against chat history, dense vector search fails to retrieve relevant documents due to missing contextual keywords. |
| **Q8** | **[B]** | LangGraph uses add_conditional_edges(source_node, routing_function, path_map) to evaluate a decision function and direct flow to target nodes based on its returned string. |
| **Q9** | **[B]** | Prompting an LLM for free text often yields conversational preambles (e.g., 'Sure! The answer is: Yes'). In production, structured output enforces a strict JSON schema via API tool-calling protocols, guaranteeing clean, parseable outputs. |
| **Q10** | **[B]** | LangGraph includes a configurable recursion_limit in the invoke/stream config (defaulting to 25). If a cyclic graph fails to exit within that step budget, it raises a GraphRecursionError, preventing runaway API costs and infinite loops. |
| **Q11** | **[B]** | Corrective RAG (CRAG) evaluates retrieval quality. When internal vector retrieval fails to produce relevant context, CRAG falls back to external web search engines to rescue the query rather than returning a dead-end failure. |
| **Q12** | **[B]** | Self-RAG uses self-reflection tokens to check whether the answer is strictly supported (is_sup) by retrieved internal documents. CRAG's primary feature is an external web search fallback, which cannot be used in an air-gapped, zero-internet environment. |
| **Q13** | **[B]** | Post-filtering fetches K documents globally and then discards unauthorized ones. If all K items belong to restricted roles, the user receives an empty result despite valid authorized documents existing deeper in the index. Pre-filtering inside the vector search avoids this. |
| **Q14** | **[A]** | Unbounded conversation history causes linear-to-quadratic token growth, driving up costs and latency. Summarizing past turns into a concise state summary while keeping only recent messages bounds token usage without losing conversational continuity. |
| **Q15** | **[B]** | SQLite locks the database file during writes and cannot be accessed concurrently across multiple stateless container instances (Kubernetes pods). Production multi-tenant systems require a centralized networked database like PostgreSQL. |
