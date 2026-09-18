from loguru import logger

from docs_seeker.api.deps import get_llm_client
from docs_seeker.config.settings import settings
from docs_seeker.infra.logger.logging import setup_logging
from docs_seeker.models.llm import LLMRequest

SYSTEM_PROMPT = """
## 角色
你是一个专业的个人助手，请用专业的语言回答用户的问题。
"""
setup_logging(settings.log_level)
logger.info("docs-seeker 启动中...")

trajectory: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

MAX_LOOP = 1


def chat(question: str):
    """
    与助手进行对话
    """
    logger.info(f"用户问题: {question}")
    trajectory.append({"role": "user", "content": question})
    cur_loop = 0
    llm = get_llm_client()
    while cur_loop < MAX_LOOP:
        cur_loop += 1
        logger.info(trajectory)
        message = LLMRequest(
            messages=trajectory,
            max_tokens=settings.llm_decompose_max_tokens,
            temperature=settings.llm_temperature,
            stream=True,
            name="agent",
            model=settings.llm_generate_model or None,
        )
        response = llm.generate(message)
        produced = False

        try:
            # 正文提取走 LLMResponse.iter_text()：兼容 include_usage 末包（choices 为空）
            # 与 reasoning 模型先出 reasoning_content 的情形
            for delta in response.iter_text():
                produced = True
                yield delta
            if not produced:
                logger.warning(
                    f"流式生成正文为空（max_tokens={settings.llm_generate_max_tokens}）"
                    "——疑似 reasoning 吃满预算，检查 LLM_GENERATE_MAX_TOKENS"
                )
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            # langfuse.update_current_span(level="ERROR", status_message=f"答案生成失败: {e}")
            yield {"type": "error", "message": f"答案生成失败: {e}"}
            return

        # trajectory.append({"role": Role.ASSISTANT, "content": answer})


if __name__ == "__main__":
    for delta in chat("请帮我写一段python代码，计算1到100的和"):
        print(delta, end="")
