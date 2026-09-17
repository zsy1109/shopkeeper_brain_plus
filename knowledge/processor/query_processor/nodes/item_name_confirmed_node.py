import logging, re, json
from json import JSONDecodeError

logger = logging.getLogger(__name__)
from typing import Dict, Tuple, List, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.processor.query_processor.base import get_config
from knowledge.utils.mongo_history_util import get_recent_messages


class _ItemNameExtractor:

    def extract_item_name(self, original_query: str, history_context: str) -> Dict[str, Any]:
        """
        提取商品名
        Args:
            original_query: 用户原始查询
            history_context: 历史对话上下文

        Returns:

        """

        # 1. 定义LLM输出默认结果
        llm_result = {"item_names": [], "rewritten_query": original_query}

        # 2. 获取LLM客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except ConnectionError as e:
            logger.error(f"LLM客户端获取失败 原因:{str(e)}")
            return llm_result

        # 3. 获取商品名提取的提示词
        # 3.1 系统提示词
        item_name_system_prompt = "您是一位商品名提取专家，请从用户的问题以及历史对话中提取相关的商品名以及改写原始查询"
        # 3.2 用户提示词
        item_name_user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_context.strip() if history_context else "暂无历史上下文",
            query=original_query)

        # 4. 调用LLM
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=item_name_system_prompt),
                HumanMessage(content=item_name_user_prompt)
            ])
        except Exception as e:
            logger.error(f"LLM调用失败,原因：{str(e)}")
            return llm_result

        # 5. 获取LLM输出内容
        llm_response_content = llm_response.content

        # 6. 判断LLM的输出
        if not llm_response_content:
            return llm_result

        # 7. 清洗(判断输出内容的类型以及空格)和解析(反序列化)
        parsed_result: Dict[str, Any] = self._clean_and_parse(llm_response_content)

        # 8. 组装数据
        llm_result['item_names'] = parsed_result.get('item_names')
        llm_result['rewritten_query'] = parsed_result.get('rewritten_query') if parsed_result.get(
            'rewritten_query') else original_query

        # 9. 返回结果
        return llm_result

    def _clean_and_parse(self, llm_response_content: str) -> Dict[str, Any]:
        """
        清洗以及解析LLM的结果
        Args:
            llm_response_content: llm的输出

        Returns:

        """
        # 1. 去除json代码块围栏标记```{}``` llm模型换了或者模型底层调用的API升级（防御性编程）
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 2. 解析
        try:
            # 2.1 反序列化
            llm_content_obj: Dict[str, Any] = json.loads(content)

            # 2.2 获取item_names
            raw_item_names = llm_content_obj.get('item_names')

            # 2.3 判断类型
            if not isinstance(raw_item_names, list):
                item_names = []
            else:
                item_names = [item_name.strip() for item_name in raw_item_names if
                              isinstance(item_name, str) and item_name.strip()]

            # 2.4 获取rewritten_query
            raw_rewritten_query = llm_content_obj.get('rewritten_query')
            if not isinstance(raw_rewritten_query, str):
                rewritten_query = ""
            else:
                rewritten_query = raw_rewritten_query.strip()

            # 2.5 返回
            return {
                "item_names": item_names,
                "rewritten_query": rewritten_query
            }
        except JSONDecodeError as e:
            logger.error(f"llm输出结果{llm_response_content} 反序列化失败 原因:{str(e)}")
            raise JSONDecodeError(msg=e.msg,
                                  doc=e.doc,
                                  pos=e.pos)


class _ItemNameAligner:

    def __init__(self):
        self._config = get_config()

    def search_and_align(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        检索向量数据库并且和向量数据库中的商品名对齐 最终返回确定的商品名列表或者模糊的商品名列表
        Args:
            item_names: LLM提起到商品名列表

        Returns:

        """

        # 1. 混合检索向量数据库
        search_result: List[Dict[str, Any]] = self._search_vector(item_names)
        if not search_result:
            return [], []
        # 2. 根据混合向量检索到结果做对齐【confirmed/options 】
        confirmed, options = self._align(search_result)

        # 3. 分数差异化过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        # 4. 返回确定的confirmed容器和options容器
        return confirmed, options

    def _search_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """
         对LLM提取到的所有商品名进行向量检索
        Args:
            item_names: LLM提取到商品名列表

        Returns:
          Dict[str, Any]:
          例子：{"extracted_name":"LLM提取的商品名1","matches":[{向量库中查询到的文档1},{向量库中查询到的文档2}]}
          例子：{"extracted_name":"LLM提取的商品名2","matches":[{向量库中查询到的文档1},{向量库中查询到的文档2}]}

        """
        final_search_result = []
        # 1. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return final_search_result
        # 2. 获取BGE-M3嵌入模型
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f"BGE-M3客户端获取失败 原因:{str(e)}")
            return final_search_result

        # 3. 商品名列表向量化(混合向量)
        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vectors(model=bge_m3_client, embedding_documents=item_names)
        except Exception as e:
            logger.error(f"商品列表{item_names}生成混合向量失败 原因:{str(e)} ")
            return final_search_result

        # 4. 混合向量检索
        for index, item_name in enumerate(item_names):
            # 4.1 构建稠密以及混合向量的检索请求
            hybrid_requests = create_hybrid_search_requests(hybrid_vector_result['dense'][index],
                                                            hybrid_vector_result['sparse'][index])

            # 4.2 执行混合检索
            hybrid_search_result = execute_hybrid_search_query(milvus_client=milvus_client,
                                                               collection_name=self._config.item_name_collection,
                                                               search_requests=hybrid_requests,
                                                               ranker_weights=(0.5, 0.5),
                                                               norm_score=True,
                                                               limit=5,
                                                               output_fields=['item_name']
                                                               )

            # 4.3 解析结果

            matches = [{"score": item_search_res['distance'], "item_name": item_search_res['entity']['item_name']} for
                       item_search_res in
                       (hybrid_search_result[0] if hybrid_search_result else [])]

            # 4.4 封装
            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })

        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        主要职责：对齐
        怎么对齐？
        将什么样的商品名item_name放到confirmed
        将什么样的商品名item_name放到options
        将什么样的商品名item_name两个容器都不放
        制定规则：
        两个规则：1.如果从向量数据库中查到的商品名分数比如大于0.7 放到confirmed
        两个规则：2.如果从向量数据库中查到的商品名分数比如小于等于0.7大于0.45 放到options
                 3.如果从向量数据库中查到的商品名分数小于等于0.45 两个容器都不放
        疑问：TODO (后续RAG阈值调参)
         0.7 or 0.45如何给的?不应该拍着脑袋给。压测得到（构建数据集【询问的方式、llm提取到的商品名】 2. 构建阈值集 3.跑完整个流程：得到哪些阈值更适合构建的数据集）

        confirmed容器不应该出现两个一模一样的商品名--->下游检索【根本不需要两个一模一样的商品名】
        options容器中不能出现两个一模一样的商品名--->用户展示[也不应该展示两个一模一样的商品名]
        场景：confirmed中的商品名可能在options出现：如果某个商品名在confirmed中出现，不用在添加到options中。
        记住：
        像options中添加的商品名满足三个条件 条件1：小于high阈值且大于options 条件2：该商品名不在options 条件3：不在confirmed中
        像confirmed中添加商品名的:条件1：阈值大于high(3个小条件) 条件2：该商品名不在confirmed中 条件3：不用考虑options的，哪怕这个商品名已经在options.只要能进入不到
        confirmed中添加



        Args:
            search_result:  向量数据库检索到的结果

        Returns:
          最终两个容器confirmed、options列表中的商品名
        """

        # 1. 定义两个容器
        confirmed_list = []
        option_list = []

        # 2. 遍历检索到的所有商品名的从milvus中的搜索结果
        for item_sea_res in search_result:
            # 2.1 获取extracted_name（LLM提取的商品名）
            llm_extract_item_name = item_sea_res.get('extracted_name')

            # 2.2 获取matches(某一个商品名下的搜索结果)
            item_name_matches = item_sea_res.get('matches')

            # 2.3 (可选) 对某一个商品名下的搜索结果根据分数进行降序排序
            item_name_matches_sorted = sorted(item_name_matches, key=lambda x: x['score'], reverse=True)

            # 2.4 (收集--->分数值大于高置信度阈值的搜索结果)
            high = [h for h in item_name_matches_sorted if h.get('score') > self._config.item_name_high_confidence]

            # 2.5 高置信度的有
            if high:
                # 1) 是否从向量数据库中搜索到的商品名还和LLM提取到的相等，如果相等，最精准
                extract = next((h for h in high if h.get('item_name') == llm_extract_item_name), None)

                if extract:  # 条件很难触发(有可能 概率低)
                    picked = extract.get('item_name')
                    if picked not in confirmed_list:  # 去重
                        confirmed_list.append(picked)
                elif len(high) == 1:
                    picked = high[0]['item_name']
                    if picked not in confirmed_list:
                        confirmed_list.append(picked)
                else:
                    # 有多个分数值比较高的商品名[0.75,0.74,0.71]
                    top_score = high[0]['score']
                    if top_score - high[1]['score'] >= self._config.item_name_score_gap:
                        picked = high[0]['item_name']
                        if picked not in confirmed_list:
                            confirmed_list.append(picked)
                    else:
                        for h in high[:self._config.item_name_max_options]:
                            picked = h.get('item_name')
                            if picked not in option_list and picked not in confirmed_list:
                                option_list.append(picked)
            # 不是高置信度，有可能是中等置信，也有可能不是中等置信
            else:
                mid = [m for m in item_name_matches_sorted if
                       m['score'] >= self._config.item_name_mid_confidence
                       and m.get('item_name') not in option_list
                       and m.get('item_name') not in confirmed_list]
                if mid:
                    for m in mid[:self._config.item_name_max_options]:
                        option_list.append(m.get('item_name'))

        return confirmed_list, option_list[:self._config.item_name_max_options]

    def _item_name_score_filter(self, confirmed: List[str], search_result: List[Dict[str, Any]]):

        """
        # RS12数字万用表和RS13数字万用表
        item_name:RS12数字万用表:0.90
        item_name:万用表:0.71
        item_name:RS13数字万用表:0.88
        Args:
            confirmed:
            search_result:[]

        Returns:

        """

        # 1. 构建 商品名 → 最高分数 的映射
        item_name_score = {}
        for search_result in search_result:
            matches = search_result.get('matches', [])
            for m in matches:
                score = m.get('score', 0)
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        # 2. 防御性检查：如果没有收集到任何分数，直接返回原始 confirmed
        if not item_name_score:
            return confirmed

        # 3. 取出分数值最大的作为基准
        max_score = max(item_name_score.values())
        return [name for name, score in item_name_score.items() if
                max_score - score <= self._config.item_name_score_gap]


class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    # ── 确认回复的匹配模式 ──
    _CONFIRM_PATTERNS = [
        r'^(对|对的?[呀哦啊]?|是的[呢呀啊]?|没错|可以[呀啊]?|是[的]?|就?是?[这那][个些]|嗯+|好+|[Oo][Kk]|[Yy][Ee][Ss]|[Yy])$',
        r'^[1-9]\d*$',
        r'^(第)?[一二三四五六七八九十]+个?$',
    ]

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor()
        self._aligner = _ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        original_query = state.get('original_query')
        session_id = state.get('session_id')

        # 1. 获取历史对话(mongodb)
        history_context = get_recent_messages(session_id=session_id, limit=10)
        formatted_history = []
        for history in history_context:
            role = history.get('role', '')
            text = history.get('text', '')
            formatted_context = f"角色:{role},内容:{text}"
            formatted_history.append(formatted_context)
        formatted_history_str = " ".join(formatted_history)

        # 2. ⭐ 优先检测：用户是否在确认上一轮给出的商品名选项
        confirmed_item_name, original_user_query = self._detect_confirmation(
            original_query, history_context
        )

        if confirmed_item_name:
            # 用户确认了商品名 → 直接写入 state，不设 answer，走检索分支
            state['item_names'] = [confirmed_item_name]
            state['rewritten_query'] = original_user_query or original_query
            logger.info(
                f"用户确认商品名: {confirmed_item_name}，"
                f"rewritten_query: {state['rewritten_query']}"
            )
        else:
            # 3. 正常流程：利用LLM进行商品名提取和查询重写
            llm_result: Dict[str, Any] = self._extractor.extract_item_name(
                original_query, formatted_history_str
            )
            item_names = llm_result.get('item_names')
            rewritten_query = llm_result.get('rewritten_query')

            # 3.5 ⭐ 兜底：LLM没有提取到商品名时，用Milvus的like查询做子串匹配
            #    这样能避免因LLM不识别的商品名（如"AI应用开发实习生"）导致误拦截
            if not item_names:
                fallback_items = self._query_fallback_items(original_query)
                if fallback_items:
                    logger.info(
                        f"LLM未提取到商品名，Milvus子串匹配兜底找到: {fallback_items}"
                    )
                    item_names = fallback_items
                    rewritten_query = original_query

            # 4. 根据item_names做判断
            if item_names:
                confirmed, options = self._aligner.search_and_align(item_names)
            else:
                confirmed, options = [], []

            # 5. ⭐ 兜底：LLM已提取到商品名，但Milvus对齐失败（如集合未加载）
            #    此时直接用LLM提取的商品名作为confirmed，避免误拦截
            if item_names and not confirmed and not options:
                logger.warning(
                    f"Milvus对齐返回空结果，LLM提取商品名={item_names}，"
                    f"使用LLM结果兜底"
                )
                confirmed = list(item_names)

            # 6. 决策
            self._decide(confirmed, options, state, rewritten_query, item_names)

        # 7. 更新state的历史对话(从MongoDB中查询出来)
        state['history'] = history_context
        return state

    # ────────────────────────────────────────────────
    #  Milvus子串匹配兜底：LLM未提取到商品名时的后备方案
    # ────────────────────────────────────────────────
    def _query_fallback_items(self, query: str) -> List[str]:
        """当LLM无法从问题中提取商品名时，通过Milvus的like查询做子串匹配兜底。

        从item_name集合中查询所有已知商品名，检查是否出现在用户query中，
        或者分词后的query词片是否出现在任意商品名中。

        Args:
            query: 用户的原始问题

        Returns:
            匹配到的商品名列表（最多3个）
        """
        from knowledge.utils.client.storage_clients import StorageClients
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            logger.warning(f"Milvus子串匹配兜底：获取Milvus客户端失败: {e}")
            return []

        try:
            results = milvus_client.query(
                self.config.item_name_collection,
                filter='item_name > ""',
                output_fields=['item_name'],
                limit=100
            )
        except Exception as e:
            logger.warning(f"Milvus子串匹配兜底：查询item_name集合失败: {e}")
            return []

        known_names = [r['item_name'] for r in results]
        if not known_names:
            return []

        # 规则1: 已知商品名完整出现在query中
        # 规则2: query完整出现在商品名中
        # 规则3: 任意长度≥2的连续子串互相包含（滑动窗口，中文兼容）
        def _has_common_substring(a: str, b: str, min_len: int = 2) -> bool:
            """检查a和b是否有长度≥min_len的公共连续子串"""
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            for win_len in range(len(shorter), min_len - 1, -1):
                for i in range(len(shorter) - win_len + 1):
                    if shorter[i:i + win_len] in longer:
                        return True
            return False

        matched = []
        for name in known_names:
            if name in query:
                matched.append(name)
                continue
            if query in name:
                matched.append(name)
                continue
            # 滑动窗口：检测query与商品名之间是否有长度≥2的公共子串
            if _has_common_substring(query, name, min_len=2):
                matched.append(name)

        if matched:
            logger.info(f"Milvus子串匹配兜底：query='{query}' → 匹配{matched[:3]}")
        return list(dict.fromkeys(matched))[:3]  # 去重+最多3个

    # ────────────────────────────────────────────────
    #  确认检测：用户说"对的"/"是的"/"1" 时自动提取产品名
    # ────────────────────────────────────────────────
    def _detect_confirmation(
        self, original_query: str, history_context: List[Dict]
    ) -> Tuple[Optional[str], Optional[str]]:
        """检测用户是否在确认上一轮的商品名选项。

        当上一轮助手消息包含 "您是在询问以下产品吗：X、Y？" 且
        当前用户消息为确认类回复时，直接提取对应产品名。

        Returns:
            (确认的商品名, 原始用户问题) 或 (None, None)
        """
        query_stripped = original_query.strip()

        # 1. 匹配确认类回复
        is_confirm = any(
            re.match(pat, query_stripped, re.IGNORECASE)
            for pat in self._CONFIRM_PATTERNS
        )
        if not is_confirm:
            return None, None

        # 2. 从最后一条 assistant 消息中提取选项
        last_assistant_msg = None
        for h in reversed(history_context):
            if h.get('role') == 'assistant':
                last_assistant_msg = h.get('text', '')
                break

        if not last_assistant_msg:
            return None, None

        # 匹配: "您是在询问以下产品吗：商品A、商品B？"
        match = re.search(r'以下产品吗[：:]\s*(.+?)[？?]', last_assistant_msg)
        if not match:
            return None, None

        options_str = match.group(1)
        options = [o.strip() for o in re.split(r'[、,，]', options_str) if o.strip()]
        if not options:
            return None, None

        # 3. 选择商品：数字 → 按索引；其他 → 取第一个
        digit_match = re.match(r'^(\d)$', query_stripped)
        if digit_match:
            idx = int(digit_match.group(1)) - 1
            selected = options[idx] if 0 <= idx < len(options) else options[0]
        else:
            selected = options[0]

        # 4. 获取用户原始问题（最后一条 user 消息之前的那条 user 消息）
        #    历史记录是按 ts DESC 排列的（最新在前），直接遍历找到倒数第二个 user
        user_count = 0
        target_user_msg = None
        for h in history_context:
            if h.get('role') == 'user':
                user_count += 1
                if user_count == 2:
                    target_user_msg = h.get('text', '')
                    break
        # 如果只有一条 user 消息（当前这条），直接用它
        if target_user_msg is None:
            for h in history_context:
                if h.get('role') == 'user':
                    target_user_msg = h.get('text', '')
                    break

        return selected, target_user_msg or original_query

    def _decide(self, confirmed: List[str], options: List[str], state: QueryGraphState,
                rewritten_query: str, item_names: List[str]):
        """
        根据confirmed、options来判断是继续检索还是返回用户提示信息
        Args:
            confirmed:  已经确认的商品名列表
            options: 模糊的商品名列表
            state: 查询状态
            rewritten_query: 重写后的问题
            item_names: LLM提取到的商品名列表

        Returns:

        """

        if confirmed:
            state['item_names'] = confirmed  # 对齐后的商品名
            state['rewritten_query'] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmedNode()
    init_state = {
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        "original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
    }
    llm_result = item_name_confirmed_node.process(init_state)

    print(llm_result)