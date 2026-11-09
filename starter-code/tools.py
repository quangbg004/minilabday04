import json
import os
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")
_TICKET_LOCK = Lock()
_VIETNAM_TIME = timezone(timedelta(hours=7))

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
    
    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.
    
    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not isinstance(category, str) or type(max_price) is not int or max_price < 0:
        return [{"error": "Danh mục hoặc giá tối đa không hợp lệ."}]

    try:
        with open(catalog_file, encoding="utf-8") as source:
            products = json.load(source)
        if not isinstance(products, list):
            return [{"error": "Danh mục sản phẩm không hợp lệ."}]
        category = category.strip().casefold()
        return [
            product for product in products
            if product["category"].casefold() == category and product["price_vnd"] <= max_price
        ]
    except FileNotFoundError:
        return [{"error": "Product catalog file not found."}]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return [{"error": "Không đọc được dữ liệu danh mục sản phẩm."}]


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.
    
    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.
    
    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    if not all(isinstance(value, str) and value.strip()
               for value in (customer_name, issue_description, priority)):
        return {"error": "Cần cung cấp tên, mô tả vấn đề và mức ưu tiên hợp lệ."}
    priority = priority.strip().casefold()
    if priority not in {"low", "medium", "high"}:
        return {"error": "Mức ưu tiên phải là low, medium hoặc high."}

    with _TICKET_LOCK:
        temporary_path = None
        try:
            try:
                with open(tickets_file, encoding="utf-8") as source:
                    tickets = json.load(source)
            except FileNotFoundError:
                tickets = []

            if not isinstance(tickets, list):
                return {"error": "Dữ liệu ticket không hợp lệ."}
            existing_ids = {ticket["ticket_id"] for ticket in tickets}
            now = datetime.now(_VIETNAM_TIME)
            sequence = len(tickets) + 1
            ticket_id = f"TK-{now:%Y%m%d}-{sequence:03d}"
            while ticket_id in existing_ids:
                sequence += 1
                ticket_id = f"TK-{now:%Y%m%d}-{sequence:03d}"

            ticket = {
                "ticket_id": ticket_id,
                "customer_name": customer_name.strip(),
                "issue_description": issue_description.strip(),
                "priority": priority,
                "status": "open",
                "created_at": now.isoformat(),
                "category": "general",
            }
            tickets.append(ticket)
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=RAW_DATA_DIR,
                                    suffix=".tmp", delete=False) as destination:
                temporary_path = destination.name
                json.dump(tickets, destination, indent=2, ensure_ascii=False)
                destination.write("\n")
            os.replace(temporary_path, tickets_file)
            temporary_path = None
        except (OSError, ValueError, KeyError, TypeError):
            return {"error": "Không thể lưu ticket hỗ trợ."}
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

    return {
        "ticket_id": ticket_id,
        "customer_name": ticket["customer_name"],
        "priority": priority,
        "status": ticket["status"],
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": "Tra cứu sản phẩm hoặc dịch vụ Vingroup theo danh mục và giá tối đa.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["xe_dien", "du_lich"]},
                "max_price": {"type": "integer", "minimum": 0, "description": "Giá tối đa bằng VNĐ."},
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": "Tạo ticket để ghi nhận vấn đề khách hàng cần hỗ trợ.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string", "description": "Tên khách hàng."},
                "issue_description": {"type": "string", "description": "Mô tả vấn đề cần hỗ trợ."},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
