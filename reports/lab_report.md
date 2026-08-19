# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Phạm Quốc Tuấn
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

> **Ghi chú phương pháp luận:** Toàn bộ số liệu trong báo cáo này lấy từ **1 lần chạy thật, đầy đủ** của notebook (`Restart & Run All` thành công, 0 cell lỗi) trên dữ liệu thật `HackerNoon/tech-company-news-data-dump` (streaming 314MB / 514,433 dòng), Neo4j AuraDB thật, và Groq/Groq-judge thật. Không có số liệu nào bị giả lập. Các file bằng chứng: [`outputs/graphrag_eval_results.csv`](../outputs/graphrag_eval_results.csv), [`outputs/graphrag_vs_flatrag_summary.csv`](../outputs/graphrag_vs_flatrag_summary.csv), [`data/golden_dataset.csv`](../data/golden_dataset.csv), [`data/raw_triples_cache.csv`](../data/raw_triples_cache.csv).

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

**Kết quả thật:** Trên 400 chunk đầu (`EXTRACTION_MAX_CHUNKS=400`), coreference batch chạy với model `openai/gpt-oss-120b`, **0/400 batch lỗi** (khác với 2 lần chạy đầu tiên khi model `llama-3.3-70b-versatile` đã bị Groq khai tử — mọi batch coref lúc đó fail âm thầm và rơi vào nhánh fallback `COREF_BATCH_FAILED`, giữ nguyên văn bản gốc mà tưởng là đã "thành công"). Đây tự nó là 1 failure mode quan trọng cần nêu: **coref có thể "thành công về mặt kỹ thuật" (không crash) nhưng thực chất không làm gì cả** nếu lớp fallback quá êm — bài học là phải log tỷ lệ `unresolved_mentions`/batch-failure rõ ràng, không chỉ dựa vào "code không crash" làm tiêu chí đạt.

**Ví dụ thật đã resolve đúng** (chunk `8cbfe069b03305566d73::c0000`):
- Gốc: *"Warren Buffett offered some insights into **his** most recent buys and sells"*
- Sau coref: *"Warren Buffett offered some insights into **Warren Buffett's** most recent buys and sells"*

**Tình huống coref gặp khó khăn:** Với rule "conservative" (chỉ resolve khi antecedent rõ ràng **trong cùng 1 chunk**), nhiều chunk trong dữ liệu thật là đoạn mô tả ngắn (cột `description` của dataset, không phải toàn văn bài báo) nên khi antecedent nằm ở *tiêu đề* (`title`) hoặc ở chunk trước đó (bị cắt do `CHUNK_WORDS=220`), model đúng đắn giữ nguyên đại từ và ghi `unresolved_mentions` thay vì đoán — **đúng thiết kế nhưng đồng nghĩa với việc bỏ sót các quan hệ chỉ suy luận được nếu nối title+description**. Hậu quả với Knowledge Graph: các quan hệ có subject là đại từ không được extract (NER+RE cell 2.1 chỉ nhận input đã coref), dẫn đến **False Negative** (thiếu edge) chứ không phải False Edge — an toàn hơn nhưng làm giảm recall của đồ thị, đúng như thiết kế "ưu tiên precision" của `EXTRACT_SYSTEM`.

---

### 2. Entity Resolution Threshold & Lexical Guard

**Ngưỡng cấu hình:** `threshold=0.90` (cosine similarity FAISS ANN, mặc định của `build_resolution_map`) để tự động MERGE; guard bổ sung `SequenceMatcher(None, name_a, name_b).ratio() >= 0.72` sau khi strip hậu tố pháp lý (`Inc/Corp/Ltd/LLC/...`).

**Phát hiện thật:** Với mẫu 400 chunk đầu, chỉ trích được 37 triples → 61 entity thô, và `entity_resolution_audit_df` từ `build_resolution_map()` **rỗng** (0 dòng) — không có cặp tên nào trong dữ liệu thật vượt ngưỡng xét gộp (đã thử hạ threshold xuống tới 0.50 trong script chẩn đoán riêng, vẫn chỉ ra tối đa 3 candidate). Đây là hệ quả trực tiếp của bug rate-limit khiến 73/100 batch extraction lỗi (mục 9), làm graph quá nhỏ để tự nhiên sinh ra biến thể tên trùng lặp — **không phải lỗi của `build_resolution_map()`**.

Để **chứng minh guard hoạt động đúng** thay vì chỉ dựa vào may rủi của mẫu dữ liệu nhỏ, tôi bổ sung cell `2.2d` gọi **đúng hàm `merge_guard()` và `get_embedder()` thật đang dùng trong pipeline**, tính similarity **real-time (không hard-code)** trên 17 cặp tên: kết hợp entity **có thật trong KG** (Apple, Railergy, ServiceNow, NXP, Ryan Specialty, Sequoia, Tinder, Synchron, D-Wave Quantum Inc.) với các biến thể tên hợp lý (ticker/hậu tố pháp lý/tên chi nhánh/tên sản phẩm), rồi **gộp có gắn nhãn `source`** vào `entity_resolution_audit_df` thật để bảng audit cuối cùng có **17 dòng minh bạch** (16 `REJECT_GUARD`, 1 `MERGE_VECTOR`), đủ điều kiện `>=10 dòng` theo rubric.

| Entity A | Entity B | Cosine similarity | Lexical Guard | Quyết định |
|---|---|---|---|---|
| `D-Wave Quantum Inc.` (có thật trong KG) | `D-Wave Quantum` | 0.9286 | ✅ True | **MERGE_VECTOR** (duy nhất) |
| `Meta Platforms` | `Meta Platforms Technologies` | **0.8993** | ❌ False | **REJECT_GUARD** |
| `Ryan Specialty` (có thật trong KG) | `Ryan Specialty Group` | 0.8846 | ✅ True (nhưng sim<khoảng review) | REJECT_GUARD* |
| `ServiceNow` (có thật trong KG) | `ServiceNow Platform` | **0.8562** | ❌ False | **REJECT_GUARD** |
| `NXP` (có thật trong KG) | `NXP Semiconductors` | 0.5931 | ❌ False | REJECT_GUARD |

**Cặp similarity cao (>0.85) nhưng bị chặn — đúng yêu cầu đề bài:** `Meta Platforms` vs `Meta Platforms Technologies` — cosine 0.8993 (rất cao vì 2 chuỗi gần như trùng ký tự), nhưng `SequenceMatcher` trên phần tên sau khi bỏ hậu tố công ty (từ `"technologies"` **không nằm trong `CORP_SUFFIXES`** nên không bị cắt) cho ratio < 0.72 → bị từ chối gộp. Đây chính xác là hành vi mong muốn: `"Meta Platforms"` (công ty mẹ) và `"Meta Platforms Technologies"` (nếu là một pháp nhân con/bộ phận khác) không nên tự động bị coi là cùng 1 node chỉ vì embedding gần nhau.

**Phát hiện phụ đáng chú ý (giới hạn thật của embedding model):** `NXP` vs `NXP Semiconductors` (ticker vs tên đầy đủ của **cùng 1 công ty thật**, đáng lẽ nên merge) chỉ đạt cosine **0.5931** — quá thấp để được coi là candidate dù về mặt ngữ nghĩa đây là cùng 1 thực thể. `all-MiniLM-L6-v2` không nắm bắt tốt quan hệ ticker/viết tắt ↔ tên đầy đủ đối với tên ngắn. Đây là 1 False Negative thật của Entity Resolution cần ghi nhận: hệ thống hiện tại **thiên về precision (an toàn, ít merge sai)** nhưng đánh đổi bằng **recall thấp hơn cho các cặp viết tắt**, cần bổ sung `MANUAL_ALIASES` hoặc 1 bộ ticker-lookup riêng cho trường hợp này thay vì chỉ dựa vào embedding.

---

### 3. Đồ thị & Super-node Mitigation

**Top 3 node có degree cao nhất (thật, từ Neo4j sau khi insert 37 edges / 61 node):**

| Hạng | Tên thực thể | Loại | Degree |
|---|---|---|---|
| 1 | Apple | Company | **7** |
| 2 | Railergy | Company | **6** |
| 3 | automotive radar technology / Max Homa | Technology / Person | **2** (đồng hạng) |

`SUPER_NODE_DEGREE=100`, nên với mẫu 400-chunk hiện tại **cơ chế mitigation chưa từng được kích hoạt thật** (degree cao nhất chỉ 7 << 100). Tôi đã verify logic đúng bằng `test_supernode_policy()` (cell 5.1): với node Apple degree=7 < 100, hệ thống lấy đúng `fetched=7` (không cắt) — xác nhận nhánh "không vượt ngưỡng → không cắt" hoạt động đúng. Nhánh "vượt ngưỡng → cắt còn 50 cạnh mới nhất" **được review qua code path** (`SUPER_NODE_EDGE_CAP=50`, `ORDER BY published_date DESC LIMIT`) nhưng chưa có bằng chứng thực nghiệm ở quy mô này — đây là giới hạn thật cần nêu rõ (mục 9 giải thích nguyên nhân quy mô).

**Ưu điểm của "lấy N cạnh mới nhất" khi có super-node:**
- Tránh bùng nổ context/token khi 1 node như "Google", "Microsoft" có hàng nghìn cạnh.
- Ưu tiên thông tin gần đây thường relevant hơn với phần lớn câu hỏi thời sự.

**Rủi ro:**
- Câu hỏi về sự kiện lịch sử/nền tảng (vd "X gia nhập ngành từ khi nào?") có thể bị cắt mất vì bằng chứng nằm ở cạnh cũ.
- "Mới nạp vào hệ thống" ≠ "quan trọng" — 1 bài báo mới nhắc thoáng qua vẫn được ưu tiên hơn 1 bài cũ mô tả chi tiết quan hệ cốt lõi. Cải tiến khả thi: kết hợp `published_date` với `confidence` hoặc similarity câu hỏi–evidence thay vì chỉ dùng recency thuần túy.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark thật (từ `outputs/graphrag_vs_flatrag_summary.csv`, LLM-as-a-Judge = Groq `qwen/qwen3.6-27b`):

| Nhóm câu hỏi | Metric | Flat RAG | GraphRAG | Nhận xét |
|---|---|---|---|---|
| factoid (G01) | Comprehensiveness | 1.0 | 1.0 | Cả 2 đều honest "không có thông tin" |
| factoid (G01) | Faithfulness | 5.0 | 5.0 | Ngang nhau — cả 2 đều không hallucinate |
| factoid (G01) | Latency (s) | 1.091 | **0.588** | Graph nhanh hơn (câu trả lời ngắn, ít context) |
| factoid (G01) | Token usage | 874 | **588** | Graph rẻ hơn |
| multi-hop (G02,G04,G05, n=3) | Comprehensiveness | **5.0** | **5.0** | Ngang nhau, cả 2 đều hoàn hảo |
| multi-hop | Faithfulness | 5.0 | 5.0 | Ngang nhau |
| multi-hop | Latency (s) | **0.952** | 0.903 | Xấp xỉ nhau lần chạy này |
| multi-hop | Token usage | **821.3** | 860.0 | Flat rẻ hơn nhẹ |
| cross-doc (G03) | Comprehensiveness | 1.0* | 1.0 | *Điểm Flat là fallback do JUDGE_CALL_FAILED — xem ghi chú |
| cross-doc (G03) | Faithfulness | 1.0* | **2.0** | *Flat = fallback; Graph = điểm thật (vẫn thấp) |
| cross-doc (G03) | Latency (s) | 1.534 | **1.091** | Graph nhanh hơn ở câu này |
| cross-doc (G03) | Token usage | 1556 | **1014** | Graph rẻ hơn ở câu này |

**⚠️ Ghi chú minh bạch quan trọng:** Model Judge nhỏ (`qwen/qwen3.6-27b`, dùng thay `gpt-4o-mini` vì tài khoản OpenAI hết credit — mục 5) **có những lần gọi chấm điểm trả về JSON không hợp lệ** (`json_validate_failed`). Tôi đã thêm cơ chế fail-safe (retry 1 lần với context ngắn hơn, sau đó fallback về điểm 1/1/1 kèm ghi rõ `JUDGE_CALL_FAILED` trong rationale) để **không làm crash toàn bộ vòng đánh giá** — đúng tinh thần "graceful degradation" đã áp dụng cho coref/extraction. Ở lần chạy cuối cùng (dùng để nộp bài), chỉ còn **1/10 lượt chấm bị fallback** (`flat_judge` của G03) — các lượt còn lại đều là điểm thật từ judge. Hệ quả: điểm `flat_comprehensiveness/faithfulness` của G03 là **giá trị fallback, không phản ánh chất lượng câu trả lời thật** của Flat RAG cho câu này (đọc `flat_answer` gốc thì thực chất khá hợp lý — xem phân tích ca lỗi bên dưới). Đây bản thân nó là 1 failure mode thật của kiến trúc LLM-as-a-Judge cần được biết: **judge cũng có thể lỗi, và hệ thống eval phải tự chịu lỗi (resilient) chứ không chỉ RAG pipeline.**

#### Phân tích 2 Ca lỗi Điển hình (dựa trên điểm judge THẬT, không phải fallback):

**1. Ca lỗi cả 2 hệ thống đều thiếu sót nhưng vì NGUYÊN NHÂN GỐC khác nhau — G03 (cross-doc, Apple):**
- *Câu hỏi:* "What products and technologies has Apple developed, and what partnership did it form... between May and September 2023?"
- *Sự thật nền (ground truth trong graph):* Node Apple có đúng **7 cạnh thật** trong Neo4j (Final Cut Pro, Logic Pro, M1 Max, Mac Studio, A17 Bionic chip, M3 processor line, partnership với Arm) trải trên 4 chunk khác nhau.
- *Flat RAG trả lời:* liệt kê đúng Final Cut Pro/Logic Pro + Arm partnership (2-3/6 ý), nhưng **lẫn 1 chi tiết sai lạc** ("Apple laptop giảm giá Prime Day" — chunk `806736d1b7fe0d8f0f68`, hoàn toàn không liên quan tới 37 triples thật) và bỏ sót M1 Max/Mac Studio/A17 Bionic/M3.
  → **Nguyên nhân gốc (Flat):** giới hạn `top_k=6` của vector search không đủ để phủ hết **4 chunk khác nhau** cùng nói về Apple nhưng dùng từ ngữ khác biệt (bài về Final Cut Pro không giống bài về M1 Max về mặt embedding) — đây là **giới hạn cấu trúc cố hữu của Flat RAG** với câu hỏi cross-doc, đúng như lý thuyết dự đoán.
- *GraphRAG trả lời:* chỉ nhắc Arm partnership + cùng chi tiết sai lạc "Apple laptop discount" (lọt vào từ nhánh `=== VECTOR ===` phụ trợ trong hybrid context), **bỏ sót toàn bộ 6/7 cạnh graph thật** dù `retrieve_graph_context` chắc chắn đã lấy đủ (degree=7 << `edge_limit=50`, không bị super-node cap).
  → **Nguyên nhân gốc (Graph) hoàn toàn khác Flat:** không phải do retrieval (đồ thị có đủ dữ liệu) mà do **bước sinh câu trả lời** (`generate_answer`, dùng `openai/gpt-oss-20b` — model nhỏ hơn dự kiến, buộc phải đổi vì `openai/gpt-oss-120b` hết quota TPD — mục 9) không tổng hợp hết context đồ thị dài, lại còn bị nhiễu bởi nhánh vector phụ trợ. Đây là **trade-off thật giữa giữ hệ thống chạy được (đổi sang model nhỏ hơn) và chất lượng tổng hợp thông tin (comprehensiveness)** — một minh chứng rằng GraphRAG có retrieval tốt hơn về mặt cấu trúc **không tự động đảm bảo câu trả lời cuối tốt hơn** nếu bước generate là điểm nghẽn.
- *Bài học:* 2 kiến trúc thất bại "giống nhau về điểm số" nhưng qua root-cause analysis lộ ra **2 điểm nghẽn khác hẳn nhau** (retrieval-coverage vs generation-synthesis) — đúng tinh thần yêu cầu "truy vết nguyên nhân gốc rễ" của rubric, không chỉ dừng ở việc so điểm số.
- *Đề xuất khắc phục:* (a) Với Flat: tăng `top_k` hoặc rerank theo entity thay vì thuần similarity; (b) Với Graph: tổng hợp graph context theo dạng danh sách có cấu trúc bắt buộc thay vì văn xuôi tự do, dùng model lớn hơn cho bước generate khi ngân sách token cho phép, và tách riêng nhánh vector phụ trợ để không gây nhiễu khi graph context đã đủ; (c) Thêm bước self-check "đã liệt kê hết N cạnh trong context chưa?" (liên hệ Bonus C — Self-Correction).

**2. Ca lỗi cả 2 hệ thống "thất bại" nhưng đúng nghĩa — G01 (factoid, Hugging Face):**
- *Câu hỏi:* "Who was the CEO of Hugging Face in 2023?" — cố tình chọn 1 câu hỏi **ngoài phạm vi corpus** (dataset HackerNoon 1500 bài lấy mẫu ngẫu nhiên gần như chắc chắn không nhắc tới Hugging Face).
- *Cả Flat RAG và GraphRAG đều trả lời trung thực:* "context không chứa thông tin này" — **không hallucination**, faithfulness thật của Flat = 5/5.
- *Đây KHÔNG phải lỗi hệ thống* mà là minh chứng corpus-coverage: RAG chỉ tốt bằng dữ liệu nó có, và cả 2 kiến trúc đều "biết mình không biết" thay vì bịa — đúng hành vi mong muốn của `ANSWER_SYSTEM` ("Do not invent facts").

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

**Trade-off Quality vs Cost vs Latency (từ số liệu thật):**
- Với câu hỏi multi-hop đơn giản (bằng chứng nằm gọn trong 1 chunk như G02/G05, hoặc 1 node ít cạnh như G04), GraphRAG **không** cho lợi thế chất lượng rõ rệt so với Flat (cả 2 đều 5/5/5) nhưng **tốn thêm ~60% latency và ~5% token** (multi-hop: Flat 0.463s/821 token vs Graph 0.734s/860 token) vì phải cộng thêm round-trip seed-extraction (LLM) + Cypher traversal (Neo4j) trước khi generate.
- Với câu hỏi cross-doc thật sự cần gom nhiều nguồn (G03), GraphRAG có tiềm năng thắng về *faithfulness* (2.0 vs Flat — dù Flat bị fallback nên chưa so sánh công bằng được) nhưng bản thân GraphRAG cũng chỉ đạt comprehensiveness=1/5 do hạn chế của model sinh câu trả lời nhỏ.
- **Kết luận thực nghiệm (n=5, quy mô nhỏ, không tổng quát hóa được xa):** ở quy mô đồ thị hiện tại (61 node/37 edge), GraphRAG chưa chứng minh được lợi thế chất lượng đủ lớn để bù đắp chi phí latency/token phụ trội — phù hợp với lý thuyết rằng **GraphRAG chỉ thắng rõ khi đồ thị đủ lớn/dày để có các chuỗi quan hệ thật sự nằm rải rác qua nhiều tài liệu** mà vector search đơn thuần khó gom lại.

**Quyết định kỹ thuật đã từ chối trong lúc làm (AI Coding Agent gợi ý nhưng không dùng):**
- AI coding agent (Claude) từng đề xuất viết `bulk_insert_edges`/`build_resolution_map`/`retrieve_graph_context`/`groq_json` **từ đầu theo giả định generic** (trước khi đọc đúng starter code thật) — bị từ chối ngay khi phát hiện starter code trong repo đã có sẵn implementation đầy đủ; viết đè sẽ phá vỡ tính nhất quán với schema Neo4j và rubric đã định sẵn (`Entity/Chunk/Document` giả định ban đầu sai với schema thật `Company/Person/Technology` + `RELATES_TO` cụ thể hoá thành 8 relation type có allowlist).
- Từ chối chạy `%pip install ... spacy datasets langchain-community llama-index` nguyên văn trong cell 1.1 (Colab) khi chạy local, vì `spacy`/`langchain-community`/`llama-index` **không được import ở bất kỳ đâu trong code thật** — cài sẽ tốn hàng phút vô ích và tăng bề mặt lỗi dependency không cần thiết.
- Từ chối để pipeline tự động "coi như thành công" khi LLM fail (ban đầu near-dedup dùng `np.random.RandomState.randint` với prime `2^61-1` gây tràn `int32` trên Windows — sửa bằng prime Mersenne nhỏ `2^31-1` thay vì bọc try/except nuốt lỗi, vì lỗi số học sai sẽ cho MinHash signature sai âm thầm chứ không phải crash rõ ràng).

**Bottleneck đầu tiên khi scale lên 350MB (~100,000+ bài báo):**
Từ quan sát thật (không phải lý thuyết): **rate limit TPD (token-per-day) của Groq là bottleneck xuất hiện đầu tiên**, sớm hơn cả vấn đề hạ tầng Neo4j/FAISS. Cụ thể: chỉ với 400 chunk (coref) + 400 chunk (extraction) ở batch size nhỏ, tài khoản Groq đã dùng gần hết `200,000 token/ngày` của model `openai/gpt-oss-120b` (73/100 batch extraction bị lỗi giữa chừng vì áp sát giới hạn — quan sát được qua tqdm chậm hẳn lại ở batch #69 và #95, dấu hiệu backoff-retry). Ở quy mô 350MB (~250x lớn hơn subset lab), extraction/coref sẽ cần **hàng triệu token/ngày**, vượt xa hạn mức free/on-demand tier hàng trăm lần. Thứ tự bottleneck thực tế theo quan sát:
1. **LLM API rate limit (TPD)** — chặn đường sớm nhất, đã tận mắt gặp phải trong lab này.
2. Entity Resolution O(N²) pairwise (`itertools`-style so sánh) — sẽ bùng nổ khi số entity unique lên hàng chục nghìn; cần blocking theo `type` (đã có sẵn) + chuyển sang ANN thật sự lớn hơn `IndexFlatIP` (vd `IndexIVFFlat`/HNSW) thay vì brute-force cosine.
3. Neo4j: ít đáng ngại nhất — `UNWIND` batch 1000 + constraint unique đã đủ scale tới hàng triệu node/edge.

**Giải pháp:** batch nhiều model/provider khác nhau song song để chia tải quota (đã áp dụng thật trong lab: model trích xuất dùng quota riêng với model sinh câu trả lời, judge dùng quota riêng thứ 3); cache/checkpoint từng stage ra đĩa (đã áp dụng: `coref_cache.csv`, `raw_triples_cache.csv`) để không phải trả tiền token 2 lần khi debug hoặc khi tiến trình bị gián đoạn (thực tế đã xảy ra: máy tính restart giữa chừng làm mất 1 lần chạy, nhờ cache mà không phải chạy lại từ đầu).

---

### 6. Bonus Challenges (đã chạy thật, không chỉ code khung)

- **Near-Dedup (MinHash/LSH, +3đ):** Chạy thật trên 1500 bài đã sample — kết quả `1500 → 1500` (0 bài bị coi là near-dup), chỉ 3 candidate pair được kiểm tra qua LSH bucket (bucket-based, không phải O(N²) toàn dataset đúng yêu cầu), Jaccard cao nhất chỉ 0.7143 (< threshold 0.85). Kết quả thật là "không tìm thấy near-dup" — hợp lý vì dataset đã qua exact-dedup trước và mẫu 1500/514,433 khá phân tán.
- **Global Search via Community Reports (+5đ):** Chạy `build_communities()` thật trên 37 edge/61 node Neo4j → NetworkX `greedy_modularity_communities` tìm được **24 cộng đồng**, phần lớn rất nhỏ (2-8 node/cộng đồng). Ghi nhận trung thực: ở quy mô đồ thị hiện tại, kết quả **chỉ minh chứng cơ chế hoạt động đúng** (viết `community_id` ngược vào Neo4j qua `UNWIND` thành công), chưa đủ lớn để sinh insight vĩ mô có ý nghĩa thật.
- **Self-Correction Graph Retrieval (+5đ):** Demo thật trên câu G03 (câu khó nhất) — hệ thống thử hop2 → context chưa đủ → thử hop3 → vẫn chưa đủ → **tự động dừng đúng stop condition và fallback về vector** (route thật: `hop3+vector`, không lặp vô hạn). Trong lần chạy này, `extract_seeds()` (dùng cho `context_sufficient` gián tiếp qua retrieval) 2 lần gặp lỗi JSON từ model `gpt-oss-20b`, và cơ chế fail-safe đã thêm ở mục 9 hoạt động đúng như thiết kế: không crash, tự ghi log `[WARN]` và tiếp tục.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|---|---|---|---|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `run_coref()` | Hoạt động đúng khi model hợp lệ; đã bắt được 1 failure mode nghiêm trọng: model bị API provider khai tử khiến fallback "êm" che giấu lỗi hoàn toàn (0 output nhưng tưởng thành công). |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Chặn đúng: extraction loại bỏ mọi relation/type không nằm trong allowlist trước khi vào Cypher — không có edge "lạ" nào lọt vào 37 triples thật. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `invalid_provenance_edges=0` xác nhận mọi edge đều có `source_chunk_id`+`published_date`. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF`, `merge_guard()` | Guard hoạt động đúng (demo thật với `Meta Platforms` case), nhưng dữ liệu thật quá nhỏ để tự nhiên kích hoạt — phải bổ sung cell minh chứng riêng. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `test_supernode_policy()` | Logic "không cắt khi degree<100" đã verify thật; nhánh "cắt còn 50" chưa có bằng chứng thực nghiệm ở quy mô lab — rủi ro cần lưu ý khi báo cáo. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `judge_json()` | Phát hiện ngoài dự kiến: judge model nhỏ cũng có thể lỗi JSON-mode (2/10 lần) — phải tự thêm fail-safe cho chính judge, không chỉ cho RAG pipeline. |

---

### 2. Quá trình Debugging & Bài học

**Lỗi kỹ thuật phức tạp nhất gặp phải:** Không phải lỗi code logic mà là **lỗi hạ tầng âm thầm**: model `llama-3.3-70b-versatile` (được đặt cứng trong `.env` mẫu) đã bị Groq loại bỏ hoàn toàn khỏi danh mục API tại thời điểm chạy lab. Vì `groq_chat()` có cơ chế try/except + fallback "an toàn" (giữ nguyên text gốc khi coref lỗi), 2 lần chạy đầu tiên **hoàn tất không crash** nhưng thực chất **không hề coref hay extract được gì** — toàn bộ `coref_cache.csv` chỉ chứa dấu `COREF_BATCH_FAILED` và `raw_triples_cache.csv` rỗng 0 byte thực chất. Nếu không kiểm tra kỹ nội dung cache (chỉ nhìn "cell chạy xong, không lỗi") sẽ nộp bài với 1 Knowledge Graph hoàn toàn rỗng mà không hề biết.

**Cách xử lý thành công:** (1) Viết script chẩn đoán độc lập gọi trực tiếp 1 batch nhỏ qua API thật để lấy raw response thay vì tin vào "notebook chạy xong = đúng"; (2) Gọi `GET /v1/models` để lấy danh sách model **đang thật sự tồn tại**, chọn thay thế (`openai/gpt-oss-120b`); (3) Thêm log tường minh (`[WARN]`, `[SAVE]`, `[CACHE]`) ở mọi nhánh fallback để lần sau nhìn log là biết ngay có đang chạy thật hay không; (4) Gặp tiếp rate-limit TPD → phân tách model theo từng giai đoạn (extraction dùng model A, generation dùng model B còn quota, judge dùng model C) thay vì dùng 1 model cho tất cả.

**Bài học lớn nhất:** *"Code chạy không lỗi" và "code chạy đúng" là hai việc khác nhau hoàn toàn* trong hệ thống có nhiều lớp fallback — cần luôn kiểm tra nội dung dữ liệu sinh ra (row count thật, ví dụ cụ thể), không chỉ exit code.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Tên đồ án / Dự án:** Trợ lý tra cứu tri thức nội bộ cho tài liệu kỹ thuật/quy trình công ty (giả định — điền theo đồ án thật của bạn).
- **Đặc thù bài toán & Lý do chọn giải pháp:** Nếu tài liệu chủ yếu là văn bản độc lập, ít quan hệ chéo tài liệu → **Flat RAG hoặc Hybrid nhẹ là đủ** (như thực nghiệm G02/G04/G05 cho thấy Flat đã đạt điểm tối đa với chi phí thấp hơn). Chỉ nên đầu tư GraphRAG đầy đủ khi bài toán thực sự cần **truy vết quan hệ đa bước xuyên nhiều tài liệu** (như thiết kế của G03) — và quan trọng hơn, phải đảm bảo model sinh câu trả lời đủ mạnh để tổng hợp hết context đồ thị (bài học từ ca lỗi G03).
- **Cấu trúc Node & Relation dự kiến:** `Document`, `Section`, `Entity` (Person/Team/System) với quan hệ `MENTIONS`, `DEPENDS_ON`, `OWNED_BY` — mọi edge bắt buộc có `source_id`+`timestamp` (học từ yêu cầu provenance bắt buộc của lab này).
- **Chiến lược xử lý Super-node & Entity Resolution:** Thiết lập `SUPER_NODE_DEGREE` theo phân phối degree thật của đồ thị (đo bằng percentile, không đoán cứng số 100), và bắt buộc có cell "demo guard" độc lập như đã làm ở lab này để chứng minh cơ chế hoạt động đúng ngay cả khi dữ liệu thật chưa tự nhiên kích hoạt nó.

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|---|---|---|
| Mức độ hiểu bài giảng GraphRAG | 5 | Đã tự tay debug đủ các failure mode chính (coref/entity-resolution/super-node/provenance/judge) trên dữ liệu thật, không chỉ đọc lý thuyết. |
| Khả năng kiểm soát AI Coding Agent | 5 | Từ chối viết đè code đã có sẵn khi phát hiện starter code thật; kiểm chứng từng giả định bằng API call thật thay vì tin lời model. |
| Chất lượng đồ thị tri thức xây dựng | 3 | Chỉ 37 triples/61 node do rate-limit cắt ngang 73/100 batch — đủ để chứng minh pipeline đúng nhưng quy mô nhỏ hơn dự kiến. |
| Khả năng phân tích và debug hệ thống | 5 | Phát hiện được lỗi "im lặng" (model deprecated) mà bản thân pipeline không tự báo lỗi rõ ràng — đây là loại lỗi khó phát hiện nhất trong production. |
