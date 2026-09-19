from pathlib import Path
from src.chunking import ChunkingStrategyComparator

def main():
    # Chọn đường dẫn tới file tài liệu mẫu trong data/ của bạn
    file_path = Path("data/university-policy-crawl/academic-regulations-undergraduate.md") # Thay bằng tên file thực tế của nhóm
    
    if not file_path.exists():
        print(f"Lỗi: Không tìm thấy file tại {file_path}. Hãy kiểm tra lại thư mục data/!")
        return

    text = file_path.read_text(encoding="utf-8")
    
    print(f"Đang phân tích file: {file_path.name}...")
    comparator = ChunkingStrategyComparator()
    comparison_results = comparator.compare(text)
    
    print("\n--- KẾT QUẢ SO SÁNH CHIẾN LƯỢC CHUNKING ---")
    for strategy_name, metrics in comparison_results.items():
        print(f"Chiến lược: {strategy_name}")
        for key, value in metrics.items():
            print(f"  - {key}: {value}")
        print("-" * 30)

if __name__ == "__main__":
    main()