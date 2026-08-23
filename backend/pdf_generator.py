import os
from fpdf import FPDF

class UnicodePDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        try:
            self.set_font("Malgun", size=9)
            self.cell(0, 10, f"페이지 {self.page_no()} / {{nb}}", align="C")
        except:
            self.set_font("Helvetica", size=9)
            self.cell(0, 10, f"Page {self.page_no()} / {{nb}}", align="C")

def clean_emojis(text):
    """맑은 고딕 폰트에서 미지원하는 이모지 및 특수 기호를 텍스트 설명으로 치환하거나 정화합니다."""
    if not text:
        return ""
    replacements = {
        "💡": "[팁] ",
        "🔍": "[출처] ",
        "🏫": "[초등] ",
        "📝": "[계획] ",
        "🚀": "[시작] ",
        "📊": "[보고] ",
        "💬": "[대화] ",
        "✨": "[안내] ",
        "🟢": "● ",
        "🔴": "○ ",
        "⚪": "○ ",
        "📥": "[다운로드] ",
        "▪": "▪ ",
        "•": "• ",
        "🗓️": "[일정] ",
        "🎯": "[목표] ",
        "✅": "[완료] ",
        "📰": "[뉴스] ",
        "📚": "[도서] ",
        "🌍": "[지구] ",
        "🧭": "[가이드] ",
        "🌿": "[환경] ",
        "🌱": "[새싹] ",
        "🏆": "[성과] ",
        "⭐": "★ ",
        "🌟": "★ ",
        "🎨": "[활동] ",
        "🔬": "[실험] ",
        "📖": "[독서] "
    }
    for emoji, repl in replacements.items():
        text = text.replace(emoji, repl)
    
    # HTML 개행 태그 잔재 정화
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    return text

def wrap_korean_text(text, limit=48):
    """한글 단어 잘림 및 우측 마진 침범 방지를 위해 지정 글자수 단위로 강제 줄바꿈을 수행합니다."""
    if not text:
        return ""
    
    lines = text.split('\n')
    wrapped_lines = []
    
    for line in lines:
        if len(line) <= limit:
            wrapped_lines.append(line)
            continue
            
        current_line = ""
        current_len = 0
        
        words = line.split(' ')
        for word in words:
            if len(word) > limit:
                for char in word:
                    if current_len + 1 > limit:
                        wrapped_lines.append(current_line)
                        current_line = char
                        current_len = 1
                    else:
                        current_line += char
                        current_len += 1
            elif current_len + len(word) + (1 if current_len > 0 else 0) > limit:
                wrapped_lines.append(current_line)
                current_line = word
                current_len = len(word)
            else:
                if current_len > 0:
                    current_line += " " + word
                    current_len += 1 + len(word)
                else:
                    current_line = word
                    current_len = len(word)
                    
        if current_line:
            wrapped_lines.append(current_line)
            
    return '\n'.join(wrapped_lines)

def markdown_to_pdf_bytes(md_text, title="수업계획서"):
    # HTML 개행 태그를 실제 개행 문자(\n)로 사전 변환하여 줄바꿈 반영
    md_text = md_text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    
    pdf = UnicodePDF()
    pdf.alias_nb_pages()
    
    # 기본 여백 선언
    pdf.set_margins(15, 20, 15)
    pdf.add_page()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    font_path = os.path.join(base_dir, "fonts", "malgun.ttf")
    font_bold_path = os.path.join(base_dir, "fonts", "malgunbd.ttf")
    
    has_font = False
    if os.path.exists(font_path):
        try:
            pdf.add_font("Malgun", "", font_path)
            if os.path.exists(font_bold_path):
                pdf.add_font("Malgun", "B", font_bold_path)
            has_font = True
        except Exception as e:
            print(f"Font load error: {e}")
            
    # --- Y축 좌표 수동 통제 및 강제 개행 절대 좌표 렌더링 ---
    # fpdf2 한글 폰트 행간 붕괴 버그(겹침 현상)를 완벽히 격파합니다.
    current_y = 30
    left_margin = 15
    line_height = 6.2  # 줄 간격 고정 수치
    
    def write_safe_line(txt, size=10, is_bold=False):
        nonlocal current_y
        
        # 하단 도달 시 페이지 넘김 및 Y축 리셋
        if current_y > 270:
            pdf.add_page()
            current_y = 25
            
        if has_font:
            pdf.set_font("Malgun", "B" if is_bold else "", size=size)
        else:
            pdf.set_font("Helvetica", "B" if is_bold else "", size=size)
            
        txt = clean_emojis(txt)
        
        # text() 함수를 사용하여 X좌표 l_margin(15mm)에 강제 고정 렌더링
        pdf.text(left_margin, current_y + (size * 0.22), txt)
        current_y += line_height

    # 1. 제목 출력 (중앙 정렬 좌표 동적 계산)
    clean_title = clean_emojis(title)
    if has_font:
        pdf.set_font("Malgun", "B", size=16)
    else:
        pdf.set_font("Helvetica", "B", size=16)
        
    title_lines = wrap_korean_text(clean_title, limit=35).split('\n')
    for tl in title_lines:
        str_w = pdf.get_string_width(tl)
        center_x = left_margin + (pdf.epw - str_w) / 2
        pdf.text(center_x, current_y + 4, tl)
        current_y += 10
        
    current_y += 4
    
    # 2. 본문 한 줄씩 파싱하여 절대 좌표 렌더링
    lines = md_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        line = line.replace('\u2013', '-').replace('\u2014', '-').replace('\u2022', '*').replace('\u2019', "'").replace('\u2018', "'").replace('\u201d', '"').replace('\u201c', '"')
        line = clean_emojis(line)
        
        # --- 표(Table)를 카드 요약 단락으로 변환하여 순차 Y축 출력 ---
        if line.strip().startswith('|'):
            if '-|-' in line or '---|---' in line or ':---' in line:
                i += 1
                continue
                
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if not parts or all(p == '' for p in parts):
                i += 1
                continue
                
            header_lbl = parts[0].replace('**', '').replace('*', '').strip()
            write_safe_line(f"▪ {header_lbl} :", size=10, is_bold=True)
            
            desc_val = " | ".join([p for p in parts[1:] if p]).replace('**', '').replace('*', '').strip()
            if desc_val:
                wrapped_desc = wrap_korean_text(desc_val, limit=48)
                for dl in wrapped_desc.split('\n'):
                    write_safe_line(f"   {dl}", size=9.5, is_bold=False)
                    
            current_y += 1.5
            i += 1
            continue
            
        # 일반 라인 처리
        if line.startswith('# '):
            current_y += 4
            write_safe_line(line[2:], size=13.5, is_bold=True)
            current_y += 2
        elif line.startswith('## '):
            current_y += 3
            write_safe_line(line[3:], size=11.5, is_bold=True)
            current_y += 1.5
        elif line.startswith('### ') or line.startswith('#### '):
            current_y += 2
            prefix_len = 4 if line.startswith('### ') else 5
            write_safe_line(line[prefix_len:], size=10, is_bold=True)
            current_y += 1
        else:
            clean_line = line.replace('**', '').replace('*', '').strip()
            if not clean_line:
                current_y += 3
                i += 1
                continue
                
            wrapped_line = wrap_korean_text(clean_line, limit=52)
            for wl in wrapped_line.split('\n'):
                write_safe_line(wl, size=10, is_bold=False)
                
        i += 1
        
    return bytes(pdf.output())
