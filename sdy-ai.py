import streamlit as st
import base64
from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from utils import dataframe_agent

# 页面配置
st.set_page_config(
    page_title="我的AI助手 - 大黄",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是名为大黄的AI助手，友好且乐于助人，能回答各类问题"}]
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.api_key = ""
    st.session_state.df = None  # 存储上传的数据
    st.session_state.current_mode = "normal"  # 模式：normal（自由提问）或data（数据问答）
    st.session_state.theme = "light"  # 默认主题
    st.session_state.selected_model = "gpt-4o-mini"  # 默认模型
    # 设置默认的系统角色描述（支持Markdown）
    st.session_state.messages = [
        {
            "role": "system",
            "content": """
    你是一个友好且专业的AI助手，能提供清晰易懂的回答

    """
        }
    ]
    st.session_state.force_refresh = False  # 强制刷新标记

# 可用模型列表
AVAILABLE_MODELS = {
    "gpt-4o-mini": "GPT-4o Mini (平衡)",
    "gpt-4o": "GPT-4o (高级)",
    "gpt-3.5-turbo": "GPT-3.5 Turbo (经济)",
    "gpt-3.5-turbo-16k": "GPT-3.5 Turbo 16k (长文本)"
}

# 模型配置
MODEL_CONFIG = {
    "gpt-4o-mini": {"temperature": 0.7, "max_tokens": 8000},
    "gpt-4o": {"temperature": 0.7, "max_tokens": 12000},
    "gpt-3.5-turbo": {"temperature": 0.7, "max_tokens": 4000},
    "gpt-3.5-turbo-16k": {"temperature": 0.7, "max_tokens": 16000}
}

# 定义主题配置（使用CSS变量）
theme_config = {
    "light": {
        "--primary-bg": "white",
        "--text-color": "#333",
        "--sidebar-bg": "#f5f5f5",
        "--chat-max-width": "80%",
    },
    "dark": {
        "--primary-bg": "#1f1f1f",
        "--text-color": "white",
        "--sidebar-bg": "#2d2d2d",
        "--chat-max-width": "80%",
    },
    "blue": {
        "--primary-bg": "#e3f2fd",
        "--text-color": "#000",
        "--sidebar-bg": "#d0e8fa",
        "--chat-max-width": "80%",
    }
}


# 应用当前主题的CSS
def apply_theme():
    current_theme = theme_config[st.session_state.theme]
    css = f"""
    <style>
    :root {{
        --primary-bg: {current_theme["--primary-bg"]};
        --text-color: {current_theme["--text-color"]};
        --sidebar-bg: {current_theme["--sidebar-bg"]};
        --chat-max-width: {current_theme["--chat-max-width"]};
    }}

    /* 应用全局样式 */
    .stApp {{
        background-color: var(--primary-bg);
        color: var(--text-color);
    }}

    /* 侧边栏样式 */
    .sidebar-content {{
        background-color: var(--sidebar-bg);
    }}

    /* 聊天消息样式 */
    .stChatMessage {{
        max-width: var(--chat-max-width);
        margin-left: auto;
        margin-right: auto;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# # 应用主题
# st.session_state.theme = 'light'
# apply_theme()


# 图表生成函数
def create_chart(input_data, chart_type):
    """生成统计图表"""
    df_data = pd.DataFrame(
        data={
            "x": input_data["columns"],
            "y": input_data["data"]
        }
    )
    df_data.set_index("x", inplace=True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        plt.figure(figsize=(10, 6))
        plt.plot(df_data.index, df_data["y"], marker="o", linestyle="--")
        plt.ylim(0, df_data["y"].max() * 1.1)
        plt.title("数据可视化")
        plt.xlabel("类别")
        plt.ylabel("数值")
        plt.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(plt.gcf())
        plt.clf()  # 清除图表


# 主题变更回调函数
def update_theme(selected_theme):
    st.session_state.theme = selected_theme
    apply_theme()


# 模型变更回调函数
def update_model():
    # 获取选择框当前值
    new_model = st.session_state.model_selector
    # 检查模型是否真的改变
    if new_model != st.session_state.selected_model:
        st.session_state.selected_model = new_model
        st.session_state.force_refresh = True  # 标记需要刷新

        # 重置对话链，使用新模型
        if 'conversation_chain' in st.session_state:
            del st.session_state['conversation_chain']

        # 显示通知
        st.info(f"已切换到模型: {AVAILABLE_MODELS[new_model]}")


# 侧边栏设计
with st.sidebar:
    # 将本地图片转换为 Base64
    def get_image_base64(path):
        try:
            with open(path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            st.error(f"无法加载背景图片: {str(e)}")
            return None


    # 获取科幻.jpg的Base64编码
    image_base64 = get_image_base64("科幻1.png")


    if image_base64:
        # 添加侧边栏背景图片
        st.markdown(
            f"""
                <style>
                [data-testid="stSidebar"] {{
                    background-image: url("data:image/jpg;base64,{image_base64}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    /* 添加半透明遮罩层，确保文字可读性 */
                    background-blend-mode: overlay;
                    background-color: rgba(255, 255, 255, 0.5);  /* 白色遮罩，透明度70% */
                }}

                /* 调整侧边栏文字颜色，确保在背景上清晰可见 */
                [data-testid="stSidebar"] .sidebar-content {{
                    color: #333;  /* 深灰色文字 */
                }}

                /* 调整按钮和输入框样式 */
                [data-testid="stSidebar"] button,
                [data-testid="stSidebar"] input {{
                    background-color: rgba(255, 255, 255, 0.8);  /* 半透明白色 */
                    border-radius: 4px;
                }}
                </style>
                """,
            unsafe_allow_html=True
        )

    st.image("./jiqi.png", caption="大黄", use_container_width=True)
    st.title("✨ 系统设置")

    # API密钥输入
    st.subheader("🔑 API密钥管理")
    api_key = st.text_input(
        "请输入OpenAI API密钥：",
        type="password",
        placeholder="hk-xxxxxxxxxxxxxxxxxxxx",
        # value=st.session_state.api_key
    )
    if api_key:
        st.session_state.api_key = api_key

    # 系统角色设置
    st.subheader("🎭 角色设定")
    system_role = st.text_area(
        "设置AI角色（支持Markdown）：",
        value=st.session_state.messages[0]["content"],  # 使用当前系统消息作为默认值
        height=150,
        key="sidebar_system_role_text_area"
    )
    if st.button("更新角色设定", key="update_role_button"):
        st.session_state.messages[0]["content"] = system_role

        # 重置对话链，强制使用新的系统消息
        if 'conversation_chain' in st.session_state:
            del st.session_state['conversation_chain']

        st.success("✅ 角色设定已更新！")  # 显示成功提示

    # 模型选择
    st.subheader("🤖 模型选择")
    st.selectbox(
        "选择AI模型：",
        list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS[x],
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model),
        on_change=update_model,
        key="model_selector"
    )

    # 显示当前模型信息
    st.write(f"当前使用: {AVAILABLE_MODELS[st.session_state.selected_model]}")
    st.write(f"上下文长度: {MODEL_CONFIG[st.session_state.selected_model]['max_tokens']} tokens")

    # 模式切换
    st.subheader("🔄 模式切换")
    mode = st.radio("选择模式：", ["自由提问", "数据问答"],
                    index=0 if st.session_state.current_mode == "normal" else 1)
    if mode == "自由提问" and st.session_state.current_mode != "normal":
        st.session_state.current_mode = "normal"
    elif mode == "数据问答" and st.session_state.current_mode != "data":
        st.session_state.current_mode = "data"

    # 数据上传功能（仅在数据问答模式显示）
    if st.session_state.current_mode == "data":
        st.subheader("📊 数据上传")
        option = st.radio("请选择数据文件类型:", ("Excel", "CSV"))
        file_type = "xlsx" if option == "Excel" else "csv"
        data = st.file_uploader(f"上传你的{option}数据文件", type=file_type)
        if data:
            try:
                if file_type == "xlsx":
                    st.session_state["df"] = pd.read_excel(data, sheet_name='data')
                else:
                    st.session_state["df"] = pd.read_csv(data)
                with st.expander("原始数据"):
                    st.dataframe(st.session_state["df"])
            except Exception as e:
                st.error(f"无法读取文件: {str(e)}")


    # 侧边栏主题切换部分
    st.subheader("🎨 界面主题")
    new_theme = st.radio(
        "选择主题：",
        ["light", "dark", "blue"],
        index=1
        # key="theme_selector"
    )

    # 在主循环中检查变化
    # if 'theme_selector' in st.session_state and st.session_state.theme_selector != st.session_state.theme:
    #     st.session_state.theme = st.session_state.theme_selector
    #     apply_theme()

    if 'theme' in st.session_state and new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        apply_theme()

    # 功能按钮
    add_vertical_space(2)
    if st.button("🗑️ 清空对话历史", key="clear_conversation_button"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.memory.clear()
    if st.button("🚀 重新初始化", key="reinitialize_button"):
        st.session_state.clear()
        # 重置主题和模型状态
        st.session_state.theme = "light"
        st.session_state.selected_model = "gpt-4o-mini"
        st.session_state.force_refresh = False
        apply_theme()


# 强制刷新逻辑 - 使用隐藏按钮
if st.session_state.force_refresh:
    # 创建一个隐藏的按钮并立即点击它来触发刷新
    st.session_state.force_refresh = False
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("刷新", key="hidden_refresh_button"):
            pass

# 主界面布局
st.title("🤖 我的AI助手 - 大黄")
if st.session_state.current_mode == "normal":
    colored_header(label="", description="自由提问模式：直接输入任何问题，我会尽力解答！", color_name="blue-70")
else:
    colored_header(label="", description="数据问答模式：上传数据后，可询问与数据相关的问题", color_name="blue-70")

# 显示当前使用的模型
st.write(f"💡 当前使用模型: **{AVAILABLE_MODELS[st.session_state.selected_model]}**")

# 显示数据上传状态（仅在数据问答模式显示）
if st.session_state.current_mode == "data" and st.session_state.df is not None:
    st.info(f"当前已加载数据：{st.session_state.df.shape[0]}行，{st.session_state.df.shape[1]}列")

# 显示对话历史
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue  # 不显示系统角色
    st.chat_message(msg["role"]).write(msg["content"])

# 根据模式显示不同的输入界面
if st.session_state.current_mode == "normal":
    # 自由提问模式输入框
    user_input = st.chat_input("请输入问题（支持Markdown格式）：")

    if user_input:
        # 检查API密钥
        if not st.session_state.api_key:
            st.warning("⚠️ 请先在侧边栏输入API密钥", icon="⚠️")
            st.stop()

        # 显示用户消息
        st.chat_message("human").write(user_input)
        st.session_state.messages.append({"role": "human", "content": user_input})

        # 在自由提问模式生成AI回复的部分
        with st.spinner("大黄烧脑中..."):
            placeholder = st.empty()  # 创建占位符用于显示 GIF
            try:
                # 获取当前模型配置
                model_config = MODEL_CONFIG[st.session_state.selected_model]

                # 创建或获取对话链
                if 'conversation_chain' not in st.session_state:
                    model = ChatOpenAI(
                        model=st.session_state.selected_model,
                        api_key=st.session_state.api_key,
                        base_url="https://twapi.openai-hk.com/v1",
                        temperature=model_config["temperature"],
                        max_tokens=model_config["max_tokens"]
                    )

                    # 使用 ChatPromptTemplate 明确设置系统消息
                    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", st.session_state.messages[0]["content"]),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{input}")
                    ])

                    st.session_state.conversation_chain = ConversationChain(
                        llm=model,
                        prompt=prompt_template,
                        memory=st.session_state.memory,
                        verbose=True
                    )

                # 显示 GIF 动图
                gif_url = "宕机.jpg"
                placeholder.image(gif_url, width=80)

                # 使用对话链生成回复
                response = st.session_state.conversation_chain.predict(input=user_input)

                # 显示AI回复
                st.chat_message("ai").write(response)
                st.session_state.messages.append({"role": "ai", "content": response})

            except Exception as e:
                st.error(f"生成回复时出错: {str(e)}")
            finally:
                placeholder.empty()
else:
    # 数据问答模式输入框
    query = st.text_area(
        "请输入你关于以上数据集的问题或数据可视化需求：",
        disabled="df" not in st.session_state or st.session_state.df is None,
        key="main_query_text_area"
    )
    button = st.button("生成回答", key="main_generate_answer_button")

    if button and ("df" not in st.session_state or st.session_state.df is None):
        st.info("请先上传数据文件")
        st.stop()

    if button and query:
        with st.spinner("大黄烧脑中..."):
            placeholder = st.empty()  # 创建占位符用于显示 GIF
            try:

                # 显示 GIF 动图（示例地址，可替换为本地文件或自定义 URL）
                gif_url = "宕机.jpg"  # 示例：思考中的机器人
                placeholder.image(gif_url, width=80)  # 调整宽度
                result = dataframe_agent(st.session_state["df"], query)

                if "answer" in result:
                    st.chat_message("ai").write(result["answer"])
                    st.session_state.messages.append({"role": "ai", "content": result["answer"]})
                if "table" in result:
                    st.chat_message("ai").table(pd.DataFrame(result["table"]["data"],
                                                             columns=result["table"]["columns"]))

                if "bar" in result:
                    st.chat_message("ai").markdown("### 柱状图分析")
                    create_chart(result["bar"], "bar")
                if "line" in result:
                    st.chat_message("ai").markdown("### 折线图分析")
                    create_chart(result["line"], "line")


            except Exception as e:
                st.error(f"处理请求时出错: {str(e)}")
            finally:
                placeholder.empty()  #

# 底部提示
st.markdown("""
    <div style="text-align: right; color: #666; margin-top: 2rem;">
        提示：输入 <code>/help</code> 查看功能列表 | 使用侧边栏切换模式
    </div>
    """, unsafe_allow_html=True)