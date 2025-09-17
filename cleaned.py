import pandas as pd
import re
from html.parser import HTMLParser


df = pd.read_csv("./hoanghamobile-question-answer.csv")

def split_product(text: str):
	# Tách các sản phẩm dựa trên "Reference id "
	chunks = text.strip().split("Reference id ")
	return [("Reference id " + p.strip()) for p in chunks if p.strip()]

df['reference_str_cleaned'] = df['reference_str'].apply(split_product)

# Bước 2: Hàm tách từng mục thông tin trong mỗi sản phẩm
def extract_product_info(chunk):
	reference_id = re.search(r"Reference id ([\w-]+):", chunk)
	title = re.search(r"Product Title:\s*(.*)", chunk)
	specs = re.search(r"Product Specifications:(.*?)Promotions:", chunk, re.DOTALL)
	promotions = re.search(r"Promotions:(.*?)Price:", chunk, re.DOTALL)
	price = re.search(r"Price:\s*(.*?)Colors:", chunk, re.DOTALL)
	colors = re.search(r"Colors:\s*(.*)", chunk)

	return {
		"reference_id": reference_id.group(1).strip() if reference_id else None,
		"title": title.group(1).strip() if title else None,
		"specifications": specs.group(1).strip() if specs else None,
		"promotions": promotions.group(1).strip() if promotions else None,
		"price": price.group(1).strip() if price else None,
		"colors": colors.group(1).strip() if colors else None,
	}



product_infos_all = []

for idx, row in df.iterrows():
	chunks = row['reference_str_cleaned']  # list các chuỗi sản phẩm

	product_infos = []

	for chunk in chunks:
		if chunk.strip():
			product_info = extract_product_info(chunk)
			product_infos.append(product_info)
	
	# Gán danh sách product_info của dòng này vào list tổng
	product_infos_all.append(product_infos)

# Sau vòng lặp, gán lại vào DataFrame
df["reference_str_cleaned"] = product_infos_all

def clean_specs(text):
	if not isinstance(text, str):
		return ""
	
	text = text.split("\n")
	text = ' '.join(text)
	
	text = re.sub(r'<br>', '\n', text) 
	text = re.sub(r'<br/>', '\n', text)
	text = re.sub(r'<BR>', '\n', text)
	
	# Tách thành các dòng
	lines = [line.strip() for line in text.split('\n') if line.strip()]
	
	cleaned = []
	
	for line in lines:
		if line:
			# Thêm dấu gạch đầu dòng nếu chưa có
			if not line.startswith('-'):
				line = f"- {line}"
			cleaned.append(line)
	
	return '\n'.join(cleaned)

class MLStripper(HTMLParser):
	def __init__(self):
		super().__init__()
		self.reset()
		self.fed = []

	def handle_starttag(self, tag, attrs):
		if tag in ["br", "p", "div"]:
			self.fed.append(" ")

	def handle_data(self, d):
		self.fed.append(d)

	def get_data(self):
		return ' '.join(''.join(self.fed).split())

def remove_html_tags(text) -> str:
	if not isinstance(text, str):
		return ""
	s = MLStripper()
	s.feed(text)
	return s.get_data()


def split_promotions(text):
	if not isinstance(text, str):
		return ""
	text = text.split("\n")
	text = ' '.join(text) 
	lines = [item.strip().lstrip('- ') for item in text.split(" - ") if item.strip()]
	return '\n'.join(f"- {line}" for line in lines)

def extract_product_name_clean(title):
    # Chuyển về chữ thường và loại bỏ các từ khóa chung
    title = title.lower().replace("điện thoại", "").replace("chính hãng", "").strip()
    
    # Tách theo dấu gạch ngang
    parts = [p.strip() for p in title.split("-") if p.strip()]
    
    # Tìm phần có nhiều từ nhất (thường là tên sản phẩm)
    candidate = max(parts, key=lambda x: len(x.split()), default="")
    
    # Loại bỏ thông tin dung lượng RAM/Storage với regex chính xác hơn
    # Xử lý các pattern: 4gb/128gb, 12gb/512gb, 4gb+8gb/128gb, etc.
    candidate = re.sub(r'\(\d+gb[+/]\d+gb[/]\d+gb\)', '', candidate)  # (4gb+8gb/128gb)
    candidate = re.sub(r'\(\d+gb[/]\d+gb\)', '', candidate)           # (4gb/128gb)
    candidate = re.sub(r'\b\d+gb[+/]\d+gb[/]\d+gb\b', '', candidate)  # 4gb+8gb/128gb
    candidate = re.sub(r'\b\d+gb[/]\d+gb\b', '', candidate)           # 4gb/128gb, 12gb/512gb
    candidate = re.sub(r'\(\s*\)', '', candidate)                     # Loại bỏ dấu ngoặc trống ()
    
    # Loại bỏ khoảng trắng thừa
    candidate = re.sub(r'\s+', ' ', candidate).strip()
    
    return candidate

product_infos_all = []
product_infos_all_2 = []

for idx, row in df.iterrows():
	row_product_contents = []
	row_product_contents_2 = []
	for product_info in row["reference_str_cleaned"]:
		if product_info and extract_product_name_clean(row['title']).lower() in extract_product_name_clean(product_info["title"]).lower():
			content = (
				(f"Tên sản phẩm: {product_info['title']}\n" if product_info['title'] else "Tên sản phẩm: Không có thông tin\n") +
				(f"Màu: {product_info['colors']}\n" if product_info['colors'] else "Màu: Không có thông tin\n") +
				(f"Giá: {product_info['price']}\n" if product_info['price'] else "Giá: Không có thông tin\n") +
				(f"Thông số kỹ thuật: {clean_specs(product_info['specifications'])}\n" if product_info['specifications'] else "Thông số kỹ thuật: Không có thông tin\n") +
				(f"Khuyến mãi: {split_promotions(remove_html_tags(product_info['promotions']))}" if product_info['promotions'] else "Khuyến mãi: Không có khuyến mãi")
			)

			row_product_contents.append(content)
		
		if product_info:
			row_product_contents_2.append(
				(f"Tên sản phẩm: {product_info['title']}\n" if product_info['title'] else "Tên sản phẩm: Không có thông tin\n") +
				(f"Màu: {product_info['colors']}\n" if product_info['colors'] else "Màu: Không có thông tin\n") +
				(f"Giá: {product_info['price']}\n" if product_info['price'] else "Giá: Không có thông tin\n") +
				(f"Thông số kỹ thuật: {clean_specs(product_info['specifications'])}\n" if product_info['specifications'] else "Thông số kỹ thuật: Không có thông tin\n") +
				(f"Khuyến mãi: {split_promotions(remove_html_tags(product_info['promotions']))}" if product_info['promotions'] else "Khuyến mãi: Không có khuyến mãi")
			)
	product_infos_all.append('\n\n'.join(row_product_contents))
	product_infos_all_2.append('\n\n'.join(row_product_contents_2))

df["reference_str_cleaned"] = product_infos_all
df["reference_str"] = product_infos_all_2

df.to_csv("hoanghamobile-question-answer-cleaned.csv", index=False)