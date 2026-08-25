"""
docs-seeker - 安全护栏
输入侧：提示注入检测
输出侧：敏感信息脱敏
"""

import re

INJECTION_PATTERNS = [
    r"忽略(上述|之前|以上|前面|系统)",
    r"ignore\s+(the\s+)?(above|previous|all)",
    r"忘记.*指令",
    r"forget\s+(the\s+)?instructions",
    r"你是.*不是.*助手",
    r"you\s+are\s+(not|no\s+longer)\s+(an?\s+)?assistant",
    r"扮演.*角色",
    r"act\s+as\s+(a|an)",
    r"输出.*系统.*提示",
    r"(print|output|show|display)\s+(the\s+)?(system\s+)?prompt",
    r"切换.*模式",
    r"switch\s+mode",
    r"忽略.*限制",
    r"ignore\s+restrictions",
]

OFF_TOPIC_PATTERNS = [
    r"(写|生成|编|创作).*(诗|小说|故事|代码|歌词|文章|剧本)",
    r"(write|generate|create|compose).*(poem|story|code|lyrics|article)",
    r"(怎么|如何|怎样).*(攻击|破解|入侵|黑入|越狱)",
    r"(how\s+to|teach\s+me).*(hack|attack|crack|bypass)",
    r"(翻译|translate)\s",
]


def check_injection(question: str) -> tuple[bool, str]:
    question_lower = question.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, question_lower):
            return False, "检测到提示注入模式，请求已拒绝"
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, question_lower):
            return False, "该问题超出了我的职责范围（公司政策查询）"
    return True, ""


SENSITIVE_PATTERNS = [
    (re.compile(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"), "身份证号"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "手机号"),
    (re.compile(r"\b0\d{2,3}-?\d{7,8}\b"), "电话号"),
    (re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b"), "邮箱"),
    (re.compile(r"(?:月薪|年薪|薪资|工资|底薪)\s*\d{3,7}"), "薪资信息"),
    (re.compile(r"\b\d{16,19}\b"), "疑似银行卡号"),
]


def desensitize(text: str) -> tuple[str, list]:
    found = []
    clean_text = text
    for pattern, label in SENSITIVE_PATTERNS:
        matches = pattern.findall(clean_text)
        if matches:
            found.append(f"{label} (共{len(matches)}处)")
            clean_text = pattern.sub("***", clean_text)
    return clean_text, found


def sanitize_output(text: str) -> str:
    clean_text, _ = desensitize(text)
    return clean_text
