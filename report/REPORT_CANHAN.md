# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** [Vũ Văn Hà]
**Nhóm:** [L3A-01]
**Ngày:** [19/09/2026]

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Khi hai đoạn văn bản có độ tương tự cosine cao, các vector biểu diễn chúng có hướng gần nhau trong không gian embedding. Điều đó thường cho thấy hai đoạn có nội dung, chủ đề hoặc ý nghĩa ngữ nghĩa tương đồng, dù cách dùng từ có thể khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc."
- Câu B: "Công nghệ AI đang cách mạng hóa phương thức lao động của con người."
- Tại sao tương đồng: Hai câu cùng diễn đạt ý nghĩa rằng AI đang làm thay đổi cách con người làm việc, chỉ khác nhau về cách dùng từ.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Hôm nay thời tiết Hà Nội rất đẹp và có nắng."
- Câu B: "Thuật toán Quicksort có độ phức tạp trung bình là O(n log n)."
- Tại sao khác: Hai câu thuộc hai chủ đề không liên quan là thời tiết và thuật toán, nên hướng của các vector embedding thường khác nhau đáng kể.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine similarity tập trung vào góc giữa hai vector, nên chủ yếu đo mức độ giống nhau về hướng/ngữ nghĩa và ít bị ảnh hưởng bởi độ dài văn bản. Vì vậy, nó phù hợp hơn Euclidean distance khi so sánh một đoạn văn dài với một đoạn văn ngắn nhưng có cùng nội dung.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính: `ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.11) = 23`.
>
> Đáp án: **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100: `ceil((10,000 - 100) / (500 - 100)) = ceil(9,900 / 400) = ceil(24.75) = 25`, nên số lượng tăng lên **25 chunks**. Overlap lớn hơn giúp giữ lại ngữ cảnh ở ranh giới giữa hai chunk, từ đó giảm nguy cơ một câu hoặc ý quan trọng bị chia cắt và cải thiện khả năng truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Hàm dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách sau dấu chấm, chấm than hoặc chấm hỏi khi theo sau là khoảng trắng hoặc xuống dòng. Các câu được làm sạch khoảng trắng rồi nhóm tối đa theo `max_sentences_per_chunk`; đầu vào rỗng hoặc chỉ có khoảng trắng trả về danh sách rỗng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán thử các separator theo thứ tự ưu tiên: đoạn văn, dòng, câu, khoảng trắng và cuối cùng là từng ký tự. Base case là đoạn đã nhỏ hơn hoặc bằng `chunk_size`; nếu không còn separator phù hợp, hàm cắt trực tiếp theo kích thước để luôn tạo được chunk hợp lệ.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mỗi `Document` được chuyển thành record gồm `id`, `content`, `metadata` và vector embedding, sau đó lưu trong bộ nhớ; nếu ChromaDB khả dụng thì record cũng được thêm vào collection. Khi tìm kiếm, query được embed và so sánh với mọi vector bằng dot product, rồi sắp xếp điểm giảm dần và lấy `top_k` kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Metadata được lọc trước bằng điều kiện tất cả cặp khóa-giá trị phải khớp, sau đó chỉ các record còn lại mới được tìm kiếm vector. `delete_document` tạo lại danh sách bằng cách loại bỏ các record có `id` trùng `doc_id`, đồng thời xóa các ID tương ứng trong ChromaDB nếu backend này đang hoạt động.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Agent gọi `store.search()` để lấy các chunk liên quan, ghép chúng thành phần `Context` có kèm ID tài liệu, rồi đặt câu hỏi ở cuối prompt. Prompt yêu cầu LLM chỉ dựa vào context và nói rõ khi context không đủ; kết quả trả về chính là chuỗi do `llm_fn` sinh ra.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
42 passed in 0.19s
```

**Số lượng bài test vượt qua (pass):** **42 / 42**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | What is course registration? | How do I register for a course? | cao | -0.2118 | Không* |
| 2 | What is course registration? | How do I borrow a library book? | thấp | -0.1386 | Không* |
| 3 | Scholarship eligibility requirements | Financial aid requirements | cao | -0.3705 | Không* |
| 4 | Library opening hours | The quick brown fox jumps over the lazy dog | thấp | -0.0168 | Đúng* |
| 5 | Student withdrawal policy | How can a student drop a class? | cao | -0.1882 | Không* |

\* Điểm được tính bằng `_mock_embed`, là embedder giả lập xác định nhưng không hiểu ngữ nghĩa; vì vậy dấu hiệu cao/thấp của con số không nên được xem là đánh giá chất lượng mô hình ngôn ngữ thực.

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Điều bất ngờ nhất là các cặp câu có ý nghĩa gần nhau vẫn có điểm âm và không nhất thiết cao hơn cặp không liên quan. Điều này cho thấy `_mock_embed` chỉ phục vụ kiểm thử tính ổn định của pipeline, không đại diện cho embedding ngữ nghĩa; khi đánh giá retrieval thực tế cần dùng model embedding phù hợp với ngôn ngữ và chủ đề dữ liệu.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên đăng ký học phần như thế nào? | Dịch vụ thư viện | -0.0595 | Không | Không thể trả lời chắc chắn từ chunk top-1. |
| 2 | Học phần có thể yêu cầu điều kiện gì? | Dịch vụ thư viện | -0.0521 | Không | Context top-1 không chứa điều kiện học phần. |
| 3 | Sinh viên xử lý lỗi trùng lịch ra sao? | Đăng ký học phần | 0.0473 | Có | Cần điều chỉnh lớp học phần trước thời hạn được công bố. |
| 4 | Thư viện cung cấp những dịch vụ nào? | Dịch vụ thư viện | -0.1430 | Có | Thư viện cho mượn tài liệu và cung cấp không gian học tập. |
| 5 | Cần mang gì khi mượn tài liệu? | Dịch vụ thư viện | 0.1100 | Có | Người dùng cần mang thẻ định danh hợp lệ. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **3 / 5 (thử nghiệm sơ bộ)**

> Lưu ý: Đây là kết quả chạy trên 2 tài liệu khởi động trong `data/university/`, chưa phải bộ 5 câu hỏi chính thức của nhóm. Bộ crawl 10 URL VinUni chưa tạo được tài liệu vì các URL bị `robots.txt` từ chối.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Metadata chỉ hữu ích khi tài liệu được gán nhất quán và có nhiều giá trị để phân biệt, chẳng hạn `student` và `faculty`. Ngoài ra, chất lượng embedding và độ sạch của corpus ảnh hưởng trực tiếp đến top-k, nên không nên đánh giá hệ thống chỉ dựa trên việc code chạy thành công.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 3 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 6 / 10* |
| **Tổng phần cá nhân** | **54 / 60** |

\* Điểm tự đánh giá phần retrieval chỉ là sơ bộ vì chưa có corpus nhóm đủ 5 tài liệu và bộ benchmark chính thức.
