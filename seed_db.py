import json
import os
import sqlite3

from config import DB_PATH
from generator import generate

def seed_questions():
    if not os.path.exists(DB_PATH):
        print(f"Không tìm thấy file Database tại {DB_PATH}. Hãy chạy python init_db.py trước.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Danh sách các chủ đề muốn AI tạo câu hỏi
    topics = [
        "Mã độc File Infector cơ bản", 
        "Thuật toán đồ thị BFS", 
        "Ngữ pháp TOEIC: Thì hiện tại đơn",
        "Bảo mật hệ thống thông tin"
    ]
    
    print("🚀 Đang khởi động Ollama sinh câu hỏi. Quá trình này sẽ phụ thuộc vào tốc độ máy tính của bạn...")
    
    for topic in topics:
        print(f"\n⏳ Đang tạo câu hỏi Level 1 cho chủ đề: {topic}")
        try:
            # Gọi trực tiếp Ollama qua file generator.py
            q_dict = generate(topic=topic, difficulty=1)
            
            # Chuyển mảng options thành chuỗi JSON để lưu vào DB
            options_json = json.dumps(q_dict.get('options'), ensure_ascii=False)
            
            # Bắn thẳng dữ liệu vào bảng Questions
            cursor.execute('''
                INSERT INTO Questions ( content, difficulty, subject, options, answer, explanation)
                VALUES ( ?, ?, ?, ?, ?, ?)
            ''', (
                q_dict.get('content'),
                q_dict.get('difficulty'),
                q_dict.get('subject'),
                options_json,
                q_dict.get('answer'),
                q_dict.get('explanation')
            ))
            conn.commit()
            print("✅ Đã lưu thành công vào Database!")
        except Exception as e:
            print(f"❌ Lỗi khi tạo/lưu câu hỏi: {e}")
            
    conn.close()
    print("\n🎉 Hoàn tất nạp dữ liệu! Giờ bạn có thể chạy lại lệnh 'streamlit run app.py' để test UI.")

if __name__ == "__main__":
    seed_questions()