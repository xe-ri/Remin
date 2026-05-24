import re
from typing import List


def _looks_like_layout_noise(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return True

    tokens = re.findall(r"[A-Za-z0-9]+", normalized)
    if not tokens:
        return True

    numeric_count = sum(token.isdigit() for token in tokens)
    alpha_count = sum(any(char.isalpha() for char in token) for token in tokens)
    numeric_ratio = numeric_count / max(len(tokens), 1)
    lower_text = normalized.lower()
    axis_terms = (
        "time of a day",
        "traffic flow volume",
        "ground truth",
        "x-axis",
        "y-axis",
    )

    if numeric_count >= 8 and any(term in lower_text for term in axis_terms):
        return True
    if numeric_count >= 12 and numeric_ratio >= 0.25:
        return True
    if alpha_count < 18 and numeric_count >= 8:
        return True

    return False


class TextProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        初始化切片器
        :param chunk_size: 每个文本块的最大字符数
        :param chunk_overlap: 相邻两个块之间重叠的字符数（防止语义在切分点断裂）
        """
        self.chunk_size = max(chunk_size, 50)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))

    def clean_text(self, text: str) -> str:
        """
        基础清洗：去除多余空格、特殊的换行符等，这些会干扰向量化
        """
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in (text or "").splitlines()
        ]
        lines = [line for line in lines if line and not _looks_like_layout_noise(line)]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_text(self, text: str) -> List[str]:
        """
        递归切分算法：尝试在段落、句子、空格处切分，尽量保持语义完整
        """
        cleaned_text = self.clean_text(text)
        
        chunks = []
        start = 0
        text_len = len(cleaned_text)

        while start < text_len:
            # 定义当前块的终点
            end = start + self.chunk_size
            
            # 如果还没到全文结尾，尝试找一个优雅的断句点（如句号或换行）
            if end < text_len:
                # 在当前位置向后找 50 个字符，看有没有句号
                search_range = cleaned_text[end - 50:end + 50]
                break_point = re.search(r'[。！？\.!\?]', search_range)
                if break_point:
                    end = (end - 50) + break_point.end()

            chunk = cleaned_text[start:end]
            chunks.append(chunk)
            
            # 关键：下一次开始的位置要往回退一个 overlap 的距离
            start = end - self.chunk_overlap
            
            # 防止死循环（如果 overlap 设得比 chunk_size 还大）
            if start >= end:
                start = end

        return [
            c.strip()
            for c in chunks
            if len(c.strip()) > 10 and not _looks_like_layout_noise(c)
        ] # 过滤掉太短或明显来自图表坐标轴的碎片

# 单元测试逻辑：你可以直接运行这个文件来查看效果
if __name__ == "__main__":
    processor = TextProcessor(chunk_size=100, chunk_overlap=20)
    test_text = """
    这是第一段关于学术研究的描述。我们需要测试这个切片器是否能正常工作。
    如果文本非常长，它应该能够根据我们设定的字符数进行切分，并且保持一定的重叠部分。
    这样在检索的时候，AI 就能看到完整的上下文信息，而不是断章取义。
    """
    result = processor.split_text(test_text)
    for i, block in enumerate(result):
        print(f"--- Chunk {i+1} ---")
        print(block)
