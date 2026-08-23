import os
from pypdf import PdfReader

def extract_pdf_pages(file_path):
    """
    PDF 파일에서 텍스트를 페이지별로 추출하여 딕셔너리 리스트로 반환합니다.
    이 함수는 멀티프로세싱 자식 프로세스에서 실행하기 좋게 설계되었습니다.
    """
    pages = []
    if not os.path.exists(file_path):
        return pages
        
    try:
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    # 윈도우 한글 경로 깨짐 방지를 위해 absolute path의 슬래시화
                    clean_path = file_path.replace("\\", "/")
                    pages.append({
                        "text": text,
                        "metadata": {
                            "source": clean_path,
                            "page": page_num
                        }
                    })
            except Exception as page_err:
                # 특정 페이지 로드 실패 시 로깅 후 건너뛰기
                print(f"Error reading page {page_num} of {file_path}: {page_err}")
    except Exception as e:
        print(f"Error opening or reading PDF file {file_path}: {e}")
        
    return pages
