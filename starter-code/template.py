"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from decimal import Decimal
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA

- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ Vingroup (VinFast, Vinpearl)
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa thông tin.

## AVAILABLE TOOLS

{tools}

## CORE RULES (Bắt buộc tuân thủ)

1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). PHẢI gọi tool `search_product_catalog` để lấy dữ liệu thực.
2. KHÔNG BAO GIỜ tự tạo ticket_id. PHẢI gọi tool `submit_support_ticket` để hệ thống tự sinh.
3. Nếu khách hàng hỏi câu FAQ đơn giản (chính sách bảo hành, đổi trả), trả lời trực tiếp KHÔNG cần gọi tool.
4. Nếu câu hỏi cần NHIỀU tool, hãy gọi TUẦN TỰ từng tool rồi tổng hợp kết quả cuối cùng.

## OPERATIONAL BOUNDARIES

- Chỉ trả lời về sản phẩm/dịch vụ thuộc hệ sinh thái Vingroup.
- Từ chối lịch sự nếu câu hỏi ngoài phạm vi (tiền điện tử, chính trị, v.v.).

## OUTPUT CONTRACT

- Luôn trả lời bằng tiếng Việt.
- Khi liệt kê sản phẩm, dùng bảng hoặc danh sách có thứ tự.
- Khi tạo ticket, xác nhận lại ticket_id và mức ưu tiên cho khách hàng.

## RESPONSE FORMAT (cho mỗi bước suy luận)

Thought: <suy nghĩ bước này>
Action: {{"name": "<tên tool>", "args": {{<các tham số>}}}}
Observation: <kết quả trả về từ tool>
... (Lặp lại nếu cần thêm tool)
Final Answer: <câu trả lời hoàn chỉnh cho khách hàng>
"""

_CATALOG_CATEGORIES = {
    "xe_dien": re.compile(r"\b(?:vinfast|xe điện|vf\s*(?:\d+|wild))\b", re.IGNORECASE),
    "du_lich": re.compile(r"\b(?:vinpearl|resort|du lịch|khách sạn)\b", re.IGNORECASE),
}
_FAQ = re.compile(r"\b(?:bảo hành|đổi trả|hoàn tiền)\b", re.IGNORECASE)
_TICKET_REQUEST = re.compile(
    r"\b(?:(?:tạo|mở|gửi)(?:\s+(?:một|giúp tôi))*\s+ticket|khiếu nại|ghi nhận phản hồi|"
    r"báo lỗi|gửi phản ánh|tạo yêu cầu hỗ trợ)\b",
    re.IGNORECASE,
)
_ISSUE = re.compile(
    r"\b(?:bị lỗi|bị hỏng|bị ẩm mốc|bị mốc|không hoạt động|gặp sự cố|báo hết phòng)\b",
    re.IGNORECASE,
)
_NAME = re.compile(
    r"\b(?:tôi tên(?: là)?|tên tôi là|tôi là|khách hàng(?: tên)?|họ tên)\s*[:：]?\s+"
    r"(?P<name>[^,.;!?\n]+?)(?=\s+(?:có|gặp|muốn|cần|và)\b|[,.;!?\n]|$)",
    re.IGNORECASE,
)
_PRICE = re.compile(
    r"\b(?:dưới|tối đa|không quá|đến|tầm|khoảng)\s*"
    r"(?P<amount>\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)"
    r"\s*(?P<unit>tỷ|tỉ|triệu|nghìn|ngàn|vnđ|vnd|đồng|đ)?\b",
    re.IGNORECASE,
)
_PRICE_MULTIPLIERS = {
    "tỷ": 10**9, "tỉ": 10**9, "triệu": 10**6, "nghìn": 10**3, "ngàn": 10**3,
}
_SCOPE = re.compile(r"\b(?:vingroup|vinfast|vinpearl)\b", re.IGNORECASE)
_OUT_OF_SCOPE = re.compile(r"\b(?:tiền điện tử|bitcoin|crypto|chính trị)\b", re.IGNORECASE)


def _catalog_args(text: str, category: str) -> Dict[str, Any]:
    args = {"category": category}
    price = _PRICE.search(text)
    if price:
        amount = price["amount"]
        unit = (price["unit"] or "").casefold()
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", amount):
            amount = amount.replace(".", "").replace(",", "")
        else:
            amount = amount.replace(",", ".")
        args["max_price"] = int(Decimal(amount) * _PRICE_MULTIPLIERS.get(unit, 1))
    return args


def _ticket_args(text: str, clauses: List[str]) -> Dict[str, Any]:
    name = _NAME.search(text)
    issue = next((clause for clause in clauses if _ISSUE.search(clause)), "")
    if issue:
        issue_name = _NAME.search(issue)
        if issue_name:
            issue = issue[issue_name.end():]
        issue = issue.strip(" ,:.!?").split(",", 1)[0]
        issue = re.sub(r"\s+(?:của tôi|tôi)(?=\s+(?:bị|đặt)\b)", "", issue, flags=re.IGNORECASE)
    if not issue:
        request = _TICKET_REQUEST.search(text)
        if request:
            issue = _NAME.sub("", text[request.end():]).strip(" ,:.!?")
    priority_text = re.sub(r"\bkhông gấp\b", "ưu tiên thấp", text, flags=re.IGNORECASE)
    priority = "medium"
    if re.search(
        r"\b(?:khẩn cấp|nghiêm trọng|gấp|ưu tiên cao|mức độ cao|high)\b",
        priority_text,
        re.IGNORECASE,
    ):
        priority = "high"
    elif re.search(
        r"\b(?:ưu tiên thấp|mức độ thấp|low)\b", priority_text, re.IGNORECASE
    ):
        priority = "low"
    return {
        "customer_name": name["name"].strip() if name else "",
        "issue_description": issue[:1].upper() + issue[1:],
        "priority": priority,
    }


def _format_observation(tool_name: str, observation: Any) -> str:
    if tool_name == "search_product_catalog":
        if not observation:
            return "Rất tiếc, không tìm thấy sản phẩm hoặc dịch vụ phù hợp."
        if "error" in observation[0]:
            return observation[0]["error"]
        lines = []
        for index, product in enumerate(observation, 1):
            price = format(product["price_vnd"], ",").replace(",", ".")
            lines.append(f"{index}. {product['name']} — {price} VNĐ")
        return "\n".join(lines)
    if "error" in observation:
        return observation["error"]
    return (
        f"Đã tạo ticket {observation['ticket_id']} cho {observation['customer_name']} "
        f"với mức ưu tiên {observation['priority']}."
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": "[Chatbot Baseline] Tôi đã nhận câu hỏi. Tôi chưa có dữ liệu để xác minh thông tin.",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.system_prompt = SYSTEM_PROMPT.format(
            tools=json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
        )
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        clauses = re.split(r"[.!?;]\s+|\n+|\s+và\s+(?=tôi|cho|hãy)", user_input, flags=re.IGNORECASE)
        in_scope = _SCOPE.search(user_input) or any(
            pattern.search(user_input) for pattern in _CATALOG_CATEGORIES.values()
        )
        needs_ticket = bool(_TICKET_REQUEST.search(user_input) or (in_scope and _ISSUE.search(user_input)))
        is_faq = bool(_FAQ.search(user_input) and not needs_ticket)
        actions = []
        messages = []

        for clause in clauses:
            if _FAQ.search(clause) or _ISSUE.search(clause) or _TICKET_REQUEST.search(clause):
                continue
            category = next(
                (name for name, pattern in _CATALOG_CATEGORIES.items() if pattern.search(clause)),
                None,
            )
            if category:
                actions.append(("search_product_catalog", _catalog_args(clause, category)))
                break

        if needs_ticket:
            ticket_args = _ticket_args(user_input, clauses)
            if not ticket_args["customer_name"]:
                messages.append("Vui lòng cung cấp tên của bạn để tạo ticket hỗ trợ.")
            elif not ticket_args["issue_description"]:
                messages.append("Vui lòng mô tả vấn đề bạn cần hỗ trợ.")
            else:
                actions.append(("submit_support_ticket", ticket_args))
        if is_faq and in_scope:
            messages.append(
                "Chính sách bảo hành và đổi trả phụ thuộc vào từng sản phẩm hoặc dịch vụ. "
                "Bạn vui lòng cho biết mẫu xe hoặc dịch vụ cần hỗ trợ để đối chiếu chính sách áp dụng."
            )
        if _OUT_OF_SCOPE.search(user_input) or not (actions or messages):
            messages.append("Tôi chỉ có thể hỗ trợ về sản phẩm và dịch vụ của Vingroup.")

        iteration = 0
        while iteration < self.max_iterations and iteration < len(actions):
            tool_name, args = actions[iteration]
            observation = TOOL_MAP[tool_name](**args)
            iteration += 1
            self.trace.append({
                "step": iteration,
                "thought": "Tra cứu danh mục." if tool_name == "search_product_catalog" else "Ghi nhận yêu cầu hỗ trợ.",
                "action": {"name": tool_name, "args": args},
                "observation": observation,
            })

        responses = [
            _format_observation(step["action"]["name"], step["observation"])
            for step in self.trace
        ]
        if iteration < len(actions) or self.max_iterations < 1:
            return {
                "answer": "\n\n".join([*responses, "Đã đạt giới hạn số bước xử lý."]),
                "trace": self.trace,
                "iterations": iteration,
                "status": "max_iterations_reached",
            }

        return {
            "answer": "\n\n".join(responses + messages),
            "trace": self.trace,
            "iterations": max(iteration, 1),
            "status": "completed",
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
