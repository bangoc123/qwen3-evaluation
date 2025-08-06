from split_statements import SplitStatements

split = SplitStatements(MODEL_ID = "gemini-2.0-flash")
query = "Xiaomi Redmi 12 có bao nhiêu màu sắc lựa chọn?"
model_answer =  "Xiaomi Redmi 12 có 3 màu sắc lựa chọn: Bạc, Đen, Xanh Dương"
statements_response = split.split_statements(query, model_answer)