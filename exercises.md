# Ngày 7 — Bài tập
## Nền tảng Dữ liệu: Embedding & Vector Store | Bài tập thực hành

---

## Phần 1 — Khởi động (Cá nhân)

### Bài tập 1.1 — Cosine Similarity (Độ tương tự Cosine) bằng ngôn ngữ đời thường

Không yêu cầu toán học — hãy giải thích về mặt khái niệm:

- Điều gì xảy ra khi hai đoạn văn bản có độ tương tự cosine cao?
- Đưa ra một ví dụ cụ thể về hai câu sẽ có độ tương tự CAO và hai câu sẽ có độ tương tự THẤP.
- Tại sao độ tương tự cosine lại được ưu tiên hơn khoảng cách Euclid (Euclidean distance) đối với text embeddings?

> **Ghi kết quả vào:** Báo cáo — Phần 1 (Khởi động) -1. Ý nghĩa khái niệm: Khi hai đoạn văn bản có độ tương tự cosine cao (tiến gần về 1), điều đó có nghĩa là nội dung, ý nghĩa hoặc chủ đề của chúng rất giống nhau, dù chúng có thể dùng các từ ngữ khác nhau đôi chút. Bản chất vector: Trong không gian nhiều chiều của AI, mỗi đoạn văn bản được biểu diễn bằng một vector. Góc tạo bởi hai vector này rất nhỏ (hướng về cùng một phía). Điều này phản ánh việc hai đoạn văn bản cùng chia sẻ một ngữ nghĩa cốt lõi hoặc đang nói về cùng một vấn đề

>-2. Ví dụ: Cặp câu có độ tương tự CAO:
>Câu A: "Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc."Câu B: "Công nghệ AI đang cách mạng hóa phương thức lao động của con người." Giải thích: Hai câu này dùng từ ngữ khác nhau (AI vs. Trí tuệ nhân tạo, làm việc vs. lao động) nhưng có chung một ý nghĩa nền tảng, do đó góc giữa hai vector rất nhỏ và độ tương tự rất cao

>Cặp câu có độ tương tự THẤP:
>Câu A: "Hôm nay thời tiết Hà Nội rất đẹp và có nắng."Câu B: "Thuật toán sắp xếp nhanh (Quicksort) có độ phức tạp trung bình là $O(N \log N)$."Giải thích: Hai câu này thuộc hai lĩnh vực hoàn toàn khác nhau (thời tiết và khoa học máy tính/lập trình), hướng đi của vector trong không gian số hoàn toàn lệch hướng, dẫn đến độ tương tự cosine thấp (tiến gần về 0 hoặc âm).

>3. Độ tương tự cosine được ưu tiên hơn khoảng cách euclid vì: 
>Khắc phục vấn đề độ dài văn bản: Khoảng cách Euclid (Euclidean distance) tính độ dài đường thẳng nối hai điểm. Nếu một đoạn văn rất dài, nó chứa nhiều từ hơn, khiến tọa độ vector bị kéo dài ra xa gốc tọa độ (magnitude lớn), dù nội dung của nó hoàn toàn tương đồng với một đoạn văn ngắn. Khoảng cách Euclid sẽ cho kết quả xa nhau (sai lệch).
>Cosine chỉ quan tâm đến "Góc" (Hướng): Độ tương tự cosine chỉ đo góc giữa hai vector mà không quan tâm đến độ dài của chúng. Điều này cực kỳ hoàn hảo cho xử lý văn bản vì nó giúp so sánh chính xác ý nghĩa ngữ nghĩa (semantic) của hai đoạn văn ngay cả khi một bên viết dài dòng, giải thích ý nhiều hơn, còn một bên viết ngắn gọn, cô đọng.

---

### Bài tập 1.2 — Bài toán tính toán Chunking

- Một tài liệu có độ dài 10,000 ký tự. Bạn tiến hành chia nhỏ (chunk) với `chunk_size=500` (kích thước chunk), `overlap=50` (độ chồng chéo). Bạn dự kiến sẽ có bao nhiêu chunks?
- Công thức: `số lượng chunk = làm_tròn_lên((độ_dài_tài_liệu - độ_chồng_chéo) / (kích_thước_chunk - độ_chồng_chéo))`
- Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk sẽ thay đổi như thế nào? Tại sao bạn lại muốn tăng độ chồng chéo?

> **Ghi kết quả vào:** Báo cáo — Phần 1 (Khởi động) -Dự kiến sẽ có 23 chunks. Nếu độ chồng chéo tăng lên 100 thì số lương chunks tăng lên 25
>-Muốn tăng độ chồng chéo để: Bảo toàn ngữ cảnh (Context preservation): Khi một câu hoặc một ý quan trọng nằm ngắt quãng ngay ranh giới giữa hai chunk, việc tăng overlap giúp giữ lại phần đuôi của chunk trước ở phần đầu của chunk sau. Nhờ đó, mô hình Embedding không bị mất bối cảnh (context) khi đoạn văn bản bị cắt cụt. Tăng độ chính xác khi truy xuất (Retrieval accuracy): Giúp các câu hỏi tìm kiếm dễ dàng khớp (match) được thông tin nằm rải rác ở phần giao nhau giữa các khối dữ liệu.

---

## Phần 2 — Lập trình cốt lõi (Cá nhân)

Hoàn thành tất cả các TODOs trong `src/chunking.py`, `src/store.py`, và `src/agent.py`. `Document` dataclass và `FixedSizeChunker` đã được triển khai sẵn làm ví dụ — hãy đọc kỹ để hiểu cấu trúc trước khi lập trình phần còn lại.

Chạy `pytest tests/` để kiểm tra tiến độ.

### Danh sách cần làm (Checklist)
- [x] `Document` dataclass — ĐÃ TRIỂN KHAI SẴN
- [x] `FixedSizeChunker` — ĐÃ TRIỂN KHAI SẴN
- [X] `SentenceChunker` — tách dựa trên ranh giới câu, nhóm lại thành các chunks
- [X] `RecursiveChunker` — thử nghiệm các dấu phân cách (separators) theo thứ tự, thực hiện đệ quy trên các đoạn có kích thước quá lớn
- [X] `compute_similarity` — công thức tính độ tương tự cosine kèm cơ chế bảo vệ chia cho 0
- [X] `ChunkingStrategyComparator` — gọi cả ba chiến lược, tính toán các chỉ số thống kê
- [X] `EmbeddingStore.__init__` — khởi tạo store (lưu trữ trong bộ nhớ hoặc ChromaDB)
- [X] `EmbeddingStore.add_documents` — nhúng (embed) và lưu trữ từng tài liệu
- [X] `EmbeddingStore.search` — nhúng truy vấn, xếp hạng theo tích vô hướng (dot product)
- [X] `EmbeddingStore.get_collection_size` — trả về số lượng
- [X] `EmbeddingStore.search_with_filter` — lọc theo siêu dữ liệu (metadata), sau đó tìm kiếm
- [X] `EmbeddingStore.delete_document` — xóa tất cả các chunks của một doc_id
- [X] `KnowledgeBaseAgent.answer` — truy xuất (retrieve) + tạo prompt + gọi LLM

> **Nộp code:** thư mục `src/`
> **Ghi lại hướng tiếp cận vào:** Báo cáo — Phần 4 (Hướng tiếp cận của tôi)

---

## Phần 3 — So Sánh Chiến Lược Truy Xuất (Nhóm)

### Bài tập 3.0 — Chuẩn Bị Tài Liệu (Giờ đầu tiên)

Mỗi nhóm chọn một chủ đề (domain) và chuẩn bị bộ tài liệu:

**Bước 1 — Chọn chủ đề:** FAQ (Câu hỏi thường gặp), SOP (Quy trình chuẩn), chính sách, tài liệu kỹ thuật, công thức nấu ăn, luật, y tế, v.v.

**Bước 2 — Thu thập 5-10 tài liệu.** Chỉ dùng nguồn công khai hoặc nguồn nhóm có quyền sử dụng; lưu dưới dạng `.txt` hoặc `.md` vào thư mục `data/`.

**Quy tắc dữ liệu bắt buộc:**
- Không đưa dữ liệu cá nhân, thông tin đăng nhập, hồ sơ nội bộ hoặc nội dung có quyền sử dụng không rõ ràng vào repo.
- Với mỗi tài liệu, ghi `source_url`, `retrieved_at` (ngày lấy) và `document_version` hoặc ngày hiệu lực nếu nguồn có nêu.
- Đưa ba trường trên vào siêu dữ liệu (metadata) khi nạp (ingest); chúng giúp kiểm tra độ mới và truy vết câu trả lời.

> **Mẹo chuyển PDF sang Markdown:**
> - `pip install marker-pdf` → `marker_single input.pdf output/` (chất lượng cao, giữ cấu trúc)
> - `pip install pymupdf4llm` → `pymupdf4llm.to_markdown("input.pdf")` (nhanh, đơn giản)
> - Hoặc sao chép-dán (copy-paste) nội dung từ PDF/web vào file `.txt`

Ghi vào bảng:

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bước 3 — Thiết kế cấu trúc metadata (metadata schema):** Mỗi tài liệu cần `source_url`, `retrieved_at`, `document_version` và ít nhất 2 trường hữu ích cho việc truy xuất (ví dụ: `audience`, `department`, `category`, `language`, `difficulty`).

> **Ghi kết quả vào:** Báo cáo — Phần 2 (Lựa chọn tài liệu)

---

### Bài tập 3.1 — Thiết Kế Chiến Lược Truy Xuất (Mỗi người thử riêng)

Mỗi thành viên **tự chọn chiến lược riêng** để thử nghiệm trên cùng bộ tài liệu của nhóm.

**Bước 1 — Đường cơ sở (Baseline):** Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu. Ghi lại kết quả.

**Bước 2 — Chọn hoặc thiết kế chiến lược của bạn:**
- Dùng 1 trong 3 chiến lược có sẵn (built-in strategies) với tham số tối ưu, HOẶC
- Thiết kế chiến lược tùy chỉnh cho chủ đề của bạn (ví dụ: chia nhỏ theo cặp Câu hỏi-Đáp án, theo các phần (sections), theo tiêu đề (headers))
- Mỗi thành viên nên thử một chiến lược **khác nhau** để có cơ sở so sánh

```python
class CustomChunker:
    """Chiến lược chia nhỏ tùy chỉnh cho [chủ đề của bạn].

    Lý do thiết kế: [giải thích tại sao chiến lược này phù hợp với dữ liệu của bạn]
    """

    def chunk(self, text: str) -> list[str]:
        # Viết mã nguồn của bạn ở đây
        ...
```

**Bước 3 — So sánh:** So sánh chiến lược tùy chỉnh/được tinh chỉnh (custom/tuned strategy) với đường cơ sở (baseline) trên cùng tài liệu.

> **Ghi kết quả vào:** Báo cáo — Phần 3 (Chiến lược chia nhỏ - Chunking Strategy)

---

### Bài tập 3.2 — Chuẩn Bị Câu Hỏi Đánh Giá (Benchmark Queries)

Mỗi nhóm viết **đúng 5 câu hỏi đánh giá** kèm theo **câu trả lời chuẩn (gold answers)**.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

**Yêu cầu:**
- Câu hỏi phải đa dạng (không hỏi 5 câu có nội dung/cấu trúc giống hệt nhau)
- Câu trả lời chuẩn phải cụ thể và có thể kiểm chứng (verify) từ tài liệu
- Ít nhất 1 câu hỏi yêu cầu lọc bằng metadata (metadata filtering) để trả lời tốt

> **Ghi kết quả vào:** Báo cáo — Phần 6 (Kết quả — Câu hỏi đánh giá & Câu trả lời chuẩn)

---

### Bài tập 3.3 — Dự Đoán Độ Tương Tự Cosine (Cá nhân)

Gọi hàm `compute_similarity()` trên 5 cặp câu. **Trước khi chạy**, hãy dự đoán xem cặp câu nào sẽ có độ tương tự cao nhất/thấp nhất. Ghi lại các dự đoán của bạn và kết quả thực tế. Suy ngẫm xem điều gì khiến bạn ngạc nhiên nhất.

> **Ghi kết quả vào:** Báo cáo — Phần 5 (Dự đoán độ tương tự)

---

### Bài tập 3.4 — Chạy Đánh Giá & So Sánh Trong Nhóm

**Bước 1:** Mỗi thành viên chạy 5 câu hỏi đánh giá với chiến lược riêng. Ghi lại kết quả top-3 cho mỗi câu hỏi.

**Bước 2:** So sánh kết quả trong nhóm:
- Chiến lược nào cho việc truy xuất tốt nhất? Tại sao?
- Có câu hỏi nào mà chiến lược A tốt hơn B nhưng lại ngược lại ở câu hỏi khác không?
- Lọc bằng metadata (Metadata filtering) có giúp ích không?

**Bước 3:** Thảo luận và rút ra bài học — chuẩn bị cho phần demo (thuyết trình) với các nhóm khác.

> **Ghi kết quả vào:** Báo cáo — Phần 6 (Kết quả)
> **Gợi ý đánh giá:** xem danh sách kiểm tra ngắn trong `README.md` mục **Cách Tự Đánh Giá Kết Quả Retrieval** hoặc chi tiết hơn trong file `docs/EVALUATION.md`.

---

### Bài tập 3.5 — Phân Tích Lỗi (Failure Analysis)

Tìm ít nhất **1 trường hợp lỗi (failure case)** trong quá trình so sánh. Mô tả:
- Câu hỏi nào mà quá trình truy xuất gặp thất bại?
- Tại sao? (do chunk quá nhỏ/quá lớn, thiếu metadata, câu hỏi mơ hồ, v.v.)
- Đề xuất cải thiện?

> **Ghi kết quả vào:** Báo cáo — Phần 7 (Những gì tôi học được)
> **Gợi ý:** phân tích lỗi nên tham chiếu từ các góc nhìn như độ chính xác (precision), tính mạch lạc của chunk (chunk coherence), tính hữu dụng của metadata, và chất lượng thông tin nền (grounding quality).

---

## Danh Sách Kiểm Tra Nộp Bài (Submission Checklist)

- [ ] Vượt qua tất cả các bài kiểm thử (tests): `pytest tests/ -v`
- [ ] Cập nhật thư mục `src/` (cá nhân)
- [ ] Hoàn thành báo cáo nhóm (`report/REPORT_NHOM.md` — 1 file/nhóm)
- [ ] Hoàn thành báo cáo cá nhân (`report/REPORT_CANHAN.md` — 1 file/sinh viên)
