# -*- coding: utf-8 -*-
import gradio as gr
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 导入日志模块
from daily_logger import daily_logger, log_exceptions, log_function_call, log_info, log_warning, log_error, log_exception, log_debug

# 导入其他模块
import back_configuration as bc
from skin_analysis import Sample
import gemma3n_models as gm
import deepseek_R1_reasoning as dp

# 交互模块
import os
import uuid
import shutil
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append('.')

# 启动日志
log_info("LittleSkin智能皮肤检测平台启动")

# 全局任务管理
import threading
current_task_id = None
task_lock = threading.Lock()

def generate_task_id():
    """生成唯一的任务ID"""
    import time
    return f"task_{int(time.time() * 1000)}"

def is_task_current(task_id):
    """检查任务是否仍然是当前任务"""
    global current_task_id
    with task_lock:
        return current_task_id == task_id

def set_current_task(task_id):
    """设置当前任务ID"""
    global current_task_id
    with task_lock:
        current_task_id = task_id
        log_info(f"设置当前任务ID: {task_id}")

# 模拟数据生成函数
def generate_mock_skin_data():
    """生成模拟皮肤分析数据，用于阿里云API失败时的降级处理"""
    mock_data = {
        "data": {
            "elements": [
                {
                    "type": "acne",
                    "confidence": 0.85,
                    "location": {"x": 120, "y": 80, "width": 30, "height": 25},
                    "severity": "轻度"
                },
                {
                    "type": "wrinkle",
                    "confidence": 0.72,
                    "location": {"x": 200, "y": 150, "width": 40, "height": 15},
                    "severity": "中度"
                },
                {
                    "type": "spot",
                    "confidence": 0.68,
                    "location": {"x": 180, "y": 120, "width": 20, "height": 20},
                    "severity": "轻度"
                }
            ],
            "overall_score": 75,
            "skin_type": "混合性",
            "main_concerns": ["痘痘", "细纹", "色斑"],
            "recommendations": [
                "建议使用温和的洁面产品",
                "注意防晒，使用SPF30+的防晒霜",
                "保持充足睡眠，减少熬夜",
                "多喝水，保持皮肤水分"
            ]
        },
        "status": "mock_data",
        "message": "使用模拟数据进行演示分析"
    }
    log_info("生成模拟皮肤分析数据")
    return mock_data

def css_to_js(example_images, static_urls, intro_content, benefit_content):
  
    # —— Intro 区块 ——
    gr.HTML(intro_content, elem_id="intro-section")
    # —— Benefit 区块：极客风 & 创造飞跃版 ——
    gr.HTML(benefit_content, elem_id="benefit-section")


    # —— 上传与滑动图像区块 ——
    gr.Markdown("---", elem_id="infer-divider")
    row1 = ''.join(f'<img class="sliding-image" src="{url}" />' for url in example_images[:8])
    row2 = ''.join(f'<img class="sliding-image" src="{url}" />' for url in example_images[8:])
    gr.HTML(f"""
    <div class="sliding-images-wrapper">
    <div class="sliding-row first-row">{row1}{row1}</div>
    <div class="sliding-row second-row">{row2}{row2}</div>
    </div>
    """, elem_id="slider-container")

    gr.Markdown("---", elem_id="infer-divider")
    gr.Markdown("### ?? 用例图片【使用方法：剪切用例图片，然后将图片粘贴至图片上传区域】", elem_id="infer-title")
    
    # 在 Blocks 中插入
    static_imgs = "".join(
        f'<img class="horizontal-static-img" src="{url}" />'
        for url in static_urls
    )
    gr.HTML(f"""
    <div class="horizontal-static-section">
      {static_imgs}
    </div>
    """, elem_id="horizontal-static")

# 流式分段输出推理过程和真实输出
def stream_print(response):
    reasoning_content = ""
    content = ""
    reasoning_printed = False
    content_printed = False
    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            if not reasoning_printed:
                print("[推理过程]\n", end='', flush=True)
                reasoning_printed = True
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content

        if hasattr(delta, 'content') and delta.content:
            if not content_printed:
                print("\n[真实输出]\n", end='', flush=True)
                content_printed = True
            print(delta.content, end='', flush=True)
            content += delta.content

# —— 上传与滑动图像区块 ——
@log_exceptions
def save_uploaded_image(image_path):  # 将图片保存到本地
    """保存上传的图片到images文件夹，返回新的文件路径"""
    log_debug(f"接收到图片路径: {image_path}")

    if not image_path or not os.path.exists(image_path):
        log_error(f"图片路径不存在: {image_path}")
        return None

    # 确保images目录存在
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        log_info(f"创建images目录: {images_dir}")

    # 生成唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_extension = os.path.splitext(image_path)[1] or '.png'
    new_filename = f"uploaded_{timestamp}_{unique_id}{file_extension}"
    new_filepath = os.path.join(images_dir, new_filename)

    # 复制文件到新位置
    shutil.copy2(image_path, new_filepath)
    log_info(f"图片已保存到: {new_filepath}")

    return new_filepath

# 导入时间模块用于重试延迟
import time

# 皮肤数据分析函数 - 独立于可视化，支持重试
@log_exceptions
def get_skin_analysis_data(image):
    """获取皮肤分析数据，独立于可视化过程，支持重试机制"""
    log_info("开始皮肤数据分析")

    if image is None:
        log_warning("未检测到图片")
        return None, "? 未检测到图片"

    try:
        # 第一步是保存图片
        saved_path = save_uploaded_image(image)
        if not saved_path:
            log_error("图片保存失败")
            return None, "? 图片保存失败"

        log_info(f"开始调用阿里云皮肤分析API，图片路径: {saved_path}")

        # 重试机制：最多重试3次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                log_info(f"第{attempt + 1}次尝试调用阿里云API")

                # 获取皮肤分析数据
                skin_analysis, oss_img_url = bc.skin_analysis_instantiation(saved_path)
                log_debug(f"OSS图片URL: {oss_img_url}")

                skins_data = Sample.main(sys.argv[1:], skin_analysis, oss_img_url)

                if skins_data:
                    log_info(f"皮肤数据分析成功完成（第{attempt + 1}次尝试）")
                    return skins_data, "? 皮肤数据分析完成"
                else:
                    log_warning(f"第{attempt + 1}次尝试：皮肤数据分析返回空结果")
                    if attempt < max_retries - 1:
                        log_info(f"等待2秒后进行第{attempt + 2}次重试...")
                        import time
                        time.sleep(2)
                        continue
                    else:
                        log_warning("阿里云API失败，使用模拟数据继续流程")
                        # 使用模拟数据继续流程，确保用户体验
                        mock_data = generate_mock_skin_data()
                        return mock_data, "?? 阿里云服务暂时不可用，使用模拟数据进行演示分析"

            except Exception as retry_error:
                log_error(f"第{attempt + 1}次尝试失败: {str(retry_error)}")
                if attempt < max_retries - 1:
                    log_info(f"等待3秒后进行第{attempt + 2}次重试...")
                    import time
                    time.sleep(3)
                    continue
                else:
                    log_exception(f"所有重试均失败，最终异常: {str(retry_error)}")
                    log_warning("阿里云API异常，使用模拟数据继续流程")
                    # 使用模拟数据继续流程
                    mock_data = generate_mock_skin_data()
                    return mock_data, f"?? 阿里云服务异常，使用模拟数据进行演示分析（已重试{max_retries}次）"

    except Exception as e:
        log_exception(f"皮肤数据分析异常: {str(e)}")
        return None, f"? 皮肤数据分析失败: {str(e)}"

# 数据可视化函数 - 独立处理，不影响主流程
@log_exceptions
def generate_visualization_chart(skins_data):
    """生成可视化图表，独立处理，失败不影响主流程"""
    log_info("开始生成数据可视化图表")

    if not skins_data:
        log_warning("皮肤数据为空，无法生成可视化图表")
        return """
        <div style="text-align: center; padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
            <h3 style="color: #856404; margin-bottom: 15px;">?? 可视化图表生成失败</h3>
            <p style="color: #856404;">皮肤数据为空，无法生成可视化图表</p>
        </div>
        """

    try:
        log_info("调用Gemma3n模型生成可视化图表")
        api_key, invoke_url, model_name, max_tokens = bc.skin_data_visualization()
        chart_result = gm.gemma3n_skin_quickchartURL(skins_data, api_key, invoke_url, model_name, max_tokens)
        # 处理返回值，可能是URL字符串或(URL, config)元组
        if isinstance(chart_result, tuple):
            chart_url, config = chart_result
        else:
            chart_url = chart_result

        log_info(f"可视化图表生成成功，URL: {chart_url}")

        # 创建成功的可视化HTML
        html_content = f"""
        <div style="
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 25px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="color: #333; margin-bottom: 20px; font-weight: 600;">?? 皮肤表征可视化结果</h3>
            <img src="{chart_url}"
             alt="皮肤分析结果"
             style="max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    cursor: pointer;"
             onclick="window.open(this.src, '_blank')"
             onerror="this.parentElement.innerHTML='<div style=\\'color: #dc3545; padding: 20px; background: #ffffff;\\'>? 图表加载失败，可能是网络问题</div>'">
            <p style="margin-top: 15px; color: #666; font-size: 14px;">点击图片可查看大图</p>
        </div>
        """
        return html_content

    except Exception as e:
        log_error(f"可视化生成失败: {str(e)}", exc_info=True)

        # 返回错误提示，但不阻断主流程
        return f"""
        <div style="
            background: #ffffff;
            border: 1px solid #f5c6cb;
            border-radius: 12px;
            padding: 25px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="color: #721c24; margin-bottom: 15px; font-weight: 600;">? 可视化图表生成失败</h3>
            <p style="color: #721c24; margin-bottom: 10px;">可能原因：网络拥堵、API限流或服务暂时不可用</p>
            <p style="color: #6c757d; font-size: 14px;">但这不影响皮肤分析结果的生成</p>
            <details style="margin-top: 15px; text-align: left;">
                <summary style="color: #6c757d; cursor: pointer;">查看详细错误信息</summary>
                <pre style="background: #f8f9fa; padding: 10px; margin-top: 5px; border-radius: 4px; font-size: 12px; color: #495057;">{str(e)}</pre>
            </details>
        </div>
        """

# 导入markdown渲染库
import markdown

# HTML内容格式化函数 - 支持Markdown渲染
def format_reasoning_html(content):
    """将推理内容格式化为HTML，支持Markdown渲染"""
    if not content:
        return """
        <div id="reasoning-container" style="
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            min-height: 400px;
            max-height: 600px;
            overflow-y: scroll;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            line-height: 1.6;
            position: relative;
        ">
        <div style="color: #6c757d; text-align: center; padding: 50px;">
            ?? 等待开始推理分析...
        </div>
        </div>
        <script>
        setTimeout(function() {
            var container = document.getElementById('reasoning-container');
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        }, 100);
        </script>
        """

    # 使用markdown渲染内容
    try:
        md = markdown.Markdown(extensions=['fenced_code', 'tables'])
        rendered_content = md.convert(content)
    except:
        # 如果markdown渲染失败，使用HTML转义
        import html
        rendered_content = html.escape(content).replace('\n', '<br>')

    return f"""
    <style>
    #reasoning-container {{
        scrollbar-width: thin;
        scrollbar-color: #6c757d #f8f9fa;
    }}
    #reasoning-container::-webkit-scrollbar {{
        width: 8px;
    }}
    #reasoning-container::-webkit-scrollbar-track {{
        background: #f8f9fa;
    }}
    #reasoning-container::-webkit-scrollbar-thumb {{
        background: #6c757d;
        border-radius: 4px;
    }}
    #reasoning-container::-webkit-scrollbar-thumb:hover {{
        background: #495057;
    }}
    #reasoning-container h1, #reasoning-container h2, #reasoning-container h3 {{
        color: #343a40;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }}
    #reasoning-container code {{
        background: #e9ecef;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Consolas', 'Monaco', monospace;
    }}
    #reasoning-container pre {{
        background: #e9ecef;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }}
    #reasoning-container blockquote {{
        border-left: 4px solid #007bff;
        padding-left: 15px;
        margin: 15px 0;
        color: #6c757d;
    }}
    </style>
    <div id="reasoning-container" style="
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        min-height: 400px;
        max-height: 600px;
        overflow-y: scroll;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 14px;
        line-height: 1.6;
        position: relative;
    ">
    <div style="color: #495057;">
        {rendered_content}
    </div>
    </div>
    <script>
    (function() {{
        var container = document.getElementById('reasoning-container');
        if (!container) return;

        // 检查是否已经设置过滚动监听器
        if (!container.hasAttribute('data-scroll-handler-set')) {{
            var userInteracting = false;
            var lastUserInteraction = 0;
            var lastScrollTop = 0;
            var autoScrollEnabled = true;

            // 监听多种用户交互事件
            var interactionEvents = ['mousedown', 'wheel', 'touchstart', 'keydown'];
            interactionEvents.forEach(function(eventType) {{
                container.addEventListener(eventType, function(e) {{
                    userInteracting = true;
                    lastUserInteraction = Date.now();

                    // 如果是滚轮事件，立即禁用自动滚动
                    if (eventType === 'wheel') {{
                        autoScrollEnabled = false;
                        container.setAttribute('data-auto-scroll', 'false');

                        // 3秒后重新启用自动滚动（如果用户在底部）
                        setTimeout(function() {{
                            var currentScrollTop = container.scrollTop;
                            var maxScrollTop = container.scrollHeight - container.clientHeight;
                            if (currentScrollTop >= maxScrollTop - 20) {{
                                autoScrollEnabled = true;
                                container.setAttribute('data-auto-scroll', 'true');
                            }}
                        }}, 3000);
                    }}
                }}, {{ passive: true }});
            }});

            // 监听滚动事件
            container.addEventListener('scroll', function(e) {{
                var currentScrollTop = container.scrollTop;
                var maxScrollTop = container.scrollHeight - container.clientHeight;
                var now = Date.now();

                // 如果最近有用户交互，认为是用户主动滚动
                if (now - lastUserInteraction < 1000) {{
                    userInteracting = true;
                    autoScrollEnabled = false;
                    container.setAttribute('data-auto-scroll', 'false');
                }}

                // 检查是否滚动到底部
                if (currentScrollTop >= maxScrollTop - 20) {{
                    // 延迟恢复自动滚动，避免误判
                    setTimeout(function() {{
                        if (container.scrollTop >= container.scrollHeight - container.clientHeight - 20) {{
                            autoScrollEnabled = true;
                            container.setAttribute('data-auto-scroll', 'true');
                            userInteracting = false;
                        }}
                    }}, 500);
                }}

                lastScrollTop = currentScrollTop;
            }}, {{ passive: true }});

            // 监听鼠标离开事件
            container.addEventListener('mouseleave', function() {{
                setTimeout(function() {{
                    userInteracting = false;
                }}, 1000);
            }});

            // 标记已设置监听器
            container.setAttribute('data-scroll-handler-set', 'true');
            container.setAttribute('data-auto-scroll', 'true');
        }}

        // 智能滚动逻辑 - 更保守的自动滚动
        setTimeout(function() {{
            var autoScroll = container.getAttribute('data-auto-scroll') === 'true';
            var now = Date.now();
            var timeSinceLastInteraction = now - (parseInt(container.getAttribute('data-last-interaction')) || 0);

            // 只有在明确允许自动滚动且用户最近没有交互时才滚动
            if (autoScroll && timeSinceLastInteraction > 500) {{
                var currentScrollTop = container.scrollTop;
                var maxScrollTop = container.scrollHeight - container.clientHeight;

                // 只有当前不在底部时才滚动
                if (currentScrollTop < maxScrollTop - 10) {{
                    container.scrollTop = container.scrollHeight;
                }}
            }}
        }}, 100);

        // 记录当前时间戳
        container.setAttribute('data-last-interaction', Date.now().toString());
    }})();
    </script>
    """

def format_real_output_html(content):
    """将真实输出内容格式化为HTML，支持Markdown渲染"""
    if not content:
        return """
        <div id="real-output-container" style="
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            min-height: 400px;
            max-height: 600px;
            overflow-y: scroll;
            font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
            font-size: 15px;
            line-height: 1.8;
            position: relative;
        ">
        <div style="color: #6c757d; text-align: center; padding: 50px;">
            ? 等待推理完成后显示结果...
        </div>
        </div>
        <script>
        setTimeout(function() {
            var container = document.getElementById('real-output-container');
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        }, 100);
        </script>
        """

    # 使用markdown渲染内容
    try:
        md = markdown.Markdown(extensions=['fenced_code', 'tables'])
        rendered_content = md.convert(content)
    except:
        # 如果markdown渲染失败，使用HTML转义
        import html
        rendered_content = html.escape(content).replace('\n', '<br>')

    return f"""
    <style>
    #real-output-container {{
        scrollbar-width: thin;
        scrollbar-color: #007bff #ffffff;
    }}
    #real-output-container::-webkit-scrollbar {{
        width: 8px;
    }}
    #real-output-container::-webkit-scrollbar-track {{
        background: #ffffff;
    }}
    #real-output-container::-webkit-scrollbar-thumb {{
        background: #007bff;
        border-radius: 4px;
    }}
    #real-output-container::-webkit-scrollbar-thumb:hover {{
        background: #0056b3;
    }}
    #real-output-container h1, #real-output-container h2, #real-output-container h3 {{
        color: #007bff;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.3em;
    }}
    #real-output-container h4, #real-output-container h5, #real-output-container h6 {{
        color: #495057;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
    }}
    #real-output-container code {{
        background: #f8f9fa;
        color: #e83e8c;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Consolas', 'Monaco', monospace;
    }}
    #real-output-container pre {{
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
        margin: 15px 0;
    }}
    #real-output-container blockquote {{
        border-left: 4px solid #28a745;
        background: #f8fff9;
        padding: 10px 15px;
        margin: 15px 0;
        color: #155724;
    }}
    #real-output-container table {{
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
    }}
    #real-output-container th, #real-output-container td {{
        border: 1px solid #dee2e6;
        padding: 8px 12px;
        text-align: left;
    }}
    #real-output-container th {{
        background: #f8f9fa;
        font-weight: bold;
    }}
    #real-output-container ul, #real-output-container ol {{
        padding-left: 20px;
        margin: 10px 0;
    }}
    #real-output-container li {{
        margin: 5px 0;
    }}
    </style>
    <div id="real-output-container" style="
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        min-height: 400px;
        max-height: 600px;
        overflow-y: scroll;
        font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
        font-size: 15px;
        line-height: 1.8;
        position: relative;
    ">
    <div style="color: #212529;">
        {rendered_content}
    </div>
    </div>
    <script>
    (function() {{
        var container = document.getElementById('real-output-container');
        if (!container) return;

        // 检查是否已经设置过滚动监听器
        if (!container.hasAttribute('data-scroll-handler-set')) {{
            var userInteracting = false;
            var lastUserInteraction = 0;
            var lastScrollTop = 0;
            var autoScrollEnabled = true;

            // 监听多种用户交互事件
            var interactionEvents = ['mousedown', 'wheel', 'touchstart', 'keydown'];
            interactionEvents.forEach(function(eventType) {{
                container.addEventListener(eventType, function(e) {{
                    userInteracting = true;
                    lastUserInteraction = Date.now();

                    // 如果是滚轮事件，立即禁用自动滚动
                    if (eventType === 'wheel') {{
                        autoScrollEnabled = false;
                        container.setAttribute('data-auto-scroll', 'false');

                        // 3秒后重新启用自动滚动（如果用户在底部）
                        setTimeout(function() {{
                            var currentScrollTop = container.scrollTop;
                            var maxScrollTop = container.scrollHeight - container.clientHeight;
                            if (currentScrollTop >= maxScrollTop - 20) {{
                                autoScrollEnabled = true;
                                container.setAttribute('data-auto-scroll', 'true');
                            }}
                        }}, 3000);
                    }}
                }}, {{ passive: true }});
            }});

            // 监听滚动事件
            container.addEventListener('scroll', function(e) {{
                var currentScrollTop = container.scrollTop;
                var maxScrollTop = container.scrollHeight - container.clientHeight;
                var now = Date.now();

                // 如果最近有用户交互，认为是用户主动滚动
                if (now - lastUserInteraction < 1000) {{
                    userInteracting = true;
                    autoScrollEnabled = false;
                    container.setAttribute('data-auto-scroll', 'false');
                }}

                // 检查是否滚动到底部
                if (currentScrollTop >= maxScrollTop - 20) {{
                    // 延迟恢复自动滚动，避免误判
                    setTimeout(function() {{
                        if (container.scrollTop >= container.scrollHeight - container.clientHeight - 20) {{
                            autoScrollEnabled = true;
                            container.setAttribute('data-auto-scroll', 'true');
                            userInteracting = false;
                        }}
                    }}, 500);
                }}

                lastScrollTop = currentScrollTop;
            }}, {{ passive: true }});

            // 监听鼠标离开事件
            container.addEventListener('mouseleave', function() {{
                setTimeout(function() {{
                    userInteracting = false;
                }}, 1000);
            }});

            // 标记已设置监听器
            container.setAttribute('data-scroll-handler-set', 'true');
            container.setAttribute('data-auto-scroll', 'true');
        }}

        // 智能滚动逻辑 - 更保守的自动滚动
        setTimeout(function() {{
            var autoScroll = container.getAttribute('data-auto-scroll') === 'true';
            var now = Date.now();
            var timeSinceLastInteraction = now - (parseInt(container.getAttribute('data-last-interaction')) || 0);

            // 只有在明确允许自动滚动且用户最近没有交互时才滚动
            if (autoScroll && timeSinceLastInteraction > 500) {{
                var currentScrollTop = container.scrollTop;
                var maxScrollTop = container.scrollHeight - container.clientHeight;

                // 只有当前不在底部时才滚动
                if (currentScrollTop < maxScrollTop - 10) {{
                    container.scrollTop = container.scrollHeight;
                }}
            }}
        }}, 100);

        // 记录当前时间戳
        container.setAttribute('data-last-interaction', Date.now().toString());
    }})();
    </script>
    """

# 流式推理函数 - 真正的流式输出
def stream_deepseek_analysis(skin_data, user_prompt):
    """流式输出DeepSeek推理过程和真实输出"""
    log_info("开始DeepSeek推理分析")

    # 获取当前任务ID
    global current_task_id
    with task_lock:
        task_id = current_task_id

    try:
        # 如果没有皮肤数据，直接返回错误
        if not skin_data:
            log_error("皮肤数据为空，无法进行推理分析")
            error_reasoning = format_reasoning_html("? 皮肤数据为空，无法进行推理分析")
            error_real = format_real_output_html("? 无法进行分析")
            yield error_reasoning, error_real
            return

        log_info("实例化DeepSeek模型")
        # 实例化deepseek模型
        dp_api_key, dp_base_url, dp_model_name = bc.deepseek_R1_instantiation()

        user_question = user_prompt or "请分析我的皮肤状况"
        log_info(f"用户问题: {user_question}")

        log_info("调用DeepSeek API")

        try:
            response = dp.dp_analysis_result(skin_data, dp_api_key, dp_base_url, dp_model_name, user_question)
            log_info("DeepSeek API调用成功，开始流式输出")
        except Exception as api_error:
            log_exception(f"DeepSeek API调用失败: {str(api_error)}")
            error_reasoning = format_reasoning_html(f"? API调用失败: {str(api_error)}")
            error_real = format_real_output_html("? 无法连接到DeepSeek服务")
            yield error_reasoning, error_real
            return

        reasoning_content = ""
        real_content = ""

        # 初始状态
        initial_reasoning = format_reasoning_html("?? 正在连接DeepSeek模型...")
        initial_real = format_real_output_html("? 等待推理完成...")
        yield initial_reasoning, initial_real

        log_info("开始流式接收DeepSeek响应")

        for chunk in response:
            try:
                # 检查任务是否仍然是当前任务
                if not is_task_current(task_id):
                    log_info(f"推理任务 {task_id} 已被中断，停止流式输出")
                    interrupted_reasoning = format_reasoning_html("?? 推理已被新的图片分析中断")
                    interrupted_real = format_real_output_html("?? 分析已中断，请查看新的分析结果")
                    yield interrupted_reasoning, interrupted_real
                    return

                delta = chunk.choices[0].delta

                # 处理推理过程 - 实时流式输出
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                    reasoning_html = format_reasoning_html(reasoning_content)
                    real_html = format_real_output_html(real_content) if real_content else format_real_output_html("? 推理中，请稍候...")
                    yield reasoning_html, real_html

                # 处理真实输出 - 实时流式输出
                if hasattr(delta, 'content') and delta.content:
                    real_content += delta.content
                    reasoning_html = format_reasoning_html(reasoning_content) if reasoning_content else format_reasoning_html("? 推理完成")
                    real_html = format_real_output_html(real_content)
                    yield reasoning_html, real_html

            except Exception as chunk_error:
                log_error(f"处理响应块时出错: {str(chunk_error)}")
                continue

        # 简化日志记录
        if reasoning_content:
            log_info("推理过程输出完成")
        if real_content:
            log_info("真实输出完成")

        log_info("DeepSeek推理分析完成")

        # 确保最终状态
        final_reasoning = format_reasoning_html(reasoning_content if reasoning_content else "?? 未收到推理内容")
        final_real = format_real_output_html(real_content if real_content else "?? 未收到分析结果")
        yield final_reasoning, final_real

    except Exception as e:
        log_exception(f"推理过程出错: {str(e)}")
        error_reasoning = format_reasoning_html(f"? 推理过程出错: {str(e)}")
        error_real = format_real_output_html("? 真实输出获取失败")
        yield error_reasoning, error_real



# 主提交函数 - 独立处理皮肤分析和可视化
@log_exceptions
def main_submit_fn(image, user_prompt):
    """主提交处理函数 - 皮肤分析和可视化独立处理"""
    log_info("=== 用户提交分析请求 ===")
    log_debug(f"用户输入: {user_prompt}")

    # 生成新的任务ID并设置为当前任务
    task_id = generate_task_id()
    set_current_task(task_id)

    if image is None:
        log_warning("用户未上传图片")
        return "? 请先上传图片", "", "", "", ""

    # 第一步：获取皮肤分析数据（核心数据，必须成功）
    skin_data, analysis_status = get_skin_analysis_data(image)

    # 检查任务是否仍然是当前任务
    if not is_task_current(task_id):
        log_info(f"任务 {task_id} 已被新任务中断，停止处理")
        return "?? 任务已被新的图片分析中断", "", "", "", ""

    # 如果皮肤数据分析失败，整个流程无法继续
    if skin_data is None:
        log_error(f"皮肤数据分析失败: {analysis_status}")
        return analysis_status, "", "", "", ""

    log_info("皮肤数据分析成功，准备启动可视化和推理")

    # 第二步：返回加载中的可视化占位符，不阻塞推理
    loading_visualization = """
    <div style="
        background: #ffffff;
        border: 1px solid #bbdefb;
        border-radius: 12px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    ">
        <h3 style="color: #1976d2; margin-bottom: 15px; font-weight: 600;">?? 正在生成可视化图表...</h3>
        <p style="color: #1976d2;">请稍候，图表生成中...</p>
    </div>
    """

    # 返回初始状态，推理过程将通过生成器函数流式更新
    log_info("返回初始状态，准备启动流式推理")
    return analysis_status, loading_visualization, "?? 正在启动推理分析...", "", skin_data

# 可视化更新函数 - 在后台异步更新可视化结果
@log_exceptions
def update_visualization(skin_data):
    """后台更新可视化图表"""
    log_info("开始后台更新可视化图表")

    # 获取当前任务ID
    global current_task_id
    with task_lock:
        task_id = current_task_id

    if not skin_data:
        log_warning("皮肤数据为空，返回占位符")
        return """
        <div style="
            background: #ffffff;
            border: 1px solid #ffeaa7;
            border-radius: 12px;
            padding: 25px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="color: #856404; font-weight: 600;">?? 可视化图表生成失败</h3>
            <p style="color: #856404;">皮肤数据为空，无法生成可视化图表</p>
        </div>
        """

    # 检查任务是否仍然是当前任务
    if not is_task_current(task_id):
        log_info(f"可视化任务 {task_id} 已被中断，停止生成")
        return """
        <div style="
            background: #ffffff;
            border: 1px solid #ffc107;
            border-radius: 12px;
            padding: 25px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="color: #856404; font-weight: 600;">?? 可视化已被中断</h3>
            <p style="color: #856404;">新的图片分析已开始，请查看最新结果</p>
        </div>
        """

    # 在后台生成真正的可视化图表
    try:
        visualization_html = generate_visualization_chart(skin_data)

        # 再次检查任务是否仍然是当前任务
        if not is_task_current(task_id):
            log_info(f"可视化任务 {task_id} 在生成完成后被中断")
            return """
            <div style="
                background: #ffffff;
                border: 1px solid #ffc107;
                border-radius: 12px;
                padding: 25px;
                margin: 10px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            ">
                <h3 style="color: #856404; font-weight: 600;">?? 可视化已被中断</h3>
                <p style="color: #856404;">新的图片分析已开始，请查看最新结果</p>
            </div>
            """

        log_info("后台可视化图表更新成功")
        return visualization_html
    except Exception as e:
        log_error(f"后台可视化更新失败: {str(e)}", exc_info=True)
        return f"""
        <div style="
            background: #ffffff;
            border: 1px solid #f5c6cb;
            border-radius: 12px;
            padding: 25px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="color: #721c24; margin-bottom: 15px; font-weight: 600;">? 可视化图表更新失败</h3>
            <p style="color: #721c24;">后台更新过程中出现错误，但不影响分析结果</p>
        </div>
        """



# 图像上传和文本输入模块（交互机制）
def img_and_text_module():
    gr.Markdown("---", elem_id="infer-divider")
    gr.Markdown("### ?? 使用操作", elem_id="infer-title")

    with gr.Row():
        with gr.Column():
            # 图片上传 - 固定显示尺寸
            image_input = gr.Image(
                label="上传图片 (支持拖拽、粘贴Ctrl+V)",
                sources=["upload", "clipboard"],
                type="filepath",
                height=300,  # 固定高度
                width=700,   # 固定宽度
                container=True,
                show_label=True,
                show_download_button=False,
                interactive=True
            )

            # 用户输入
            text_input = gr.Textbox(
                label="用户输入 (可选)",
                placeholder="输入任何备注信息...",
                lines=3
            )

            # 提交按钮
            submit_btn = gr.Button("?? 开始分析", variant="primary")

        with gr.Column():
            # 结果显示
            status_output = gr.Textbox(
                label="图片加载状态",
                lines=6,
                interactive=False
            )

            # 修改为HTML组件
            result_output = gr.HTML(
                label="?? 皮肤表征可视化"
            )

    # —— 推理过程区块 ——
    gr.Markdown("---", elem_id="infer-divider")
    gr.Markdown("### ?? 模型推理过程", elem_id="infer-title")
    model_proc = gr.HTML(
        value="""
        <div style="
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            min-height: 400px;
            max-height: 600px;
            overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        ">
        <div style="color: #6c757d; text-align: center; padding: 50px;">
            ?? 等待开始推理分析...
        </div>
        </div>
        """,
        elem_id="reasoning-output"
    )

    # 这里采用分页器
    with gr.Tabs(elem_id="infer-tabs"):
        with gr.Tab(label="模型真实输出"):
            out_real = gr.HTML(
                value="""
                <div style="
                    background: #ffffff;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 20px;
                    min-height: 400px;
                    max-height: 600px;
                    overflow-y: auto;
                    font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
                    font-size: 15px;
                    line-height: 1.8;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                ">
                <div style="color: #6c757d; text-align: center; padding: 50px;">
                    ? 等待推理完成后显示结果...
                </div>
                </div>
                """,
                elem_id="real-output"
            )


    # 创建隐藏的状态组件来存储皮肤数据
    skin_data_state = gr.State()

    # 绑定事件 - 分步骤处理
    # 第一步：处理图片和初始化
    submit_event = submit_btn.click(
        fn=main_submit_fn,
        inputs=[image_input, text_input],
        outputs=[status_output, result_output, model_proc, out_real, skin_data_state]
    )

    # 第二步：启动DeepSeek流式推理分析
    submit_event.then(
        fn=stream_deepseek_analysis,
        inputs=[skin_data_state, text_input],
        outputs=[model_proc, out_real]
    )

    # 第三步：可视化更新（独立运行，不阻塞推理）
    submit_event.then(
        fn=update_visualization,
        inputs=[skin_data_state],
        outputs=[result_output]
    )
 

if __name__ == "__main__":
    try:
        log_info("开始初始化LittleSkin智能皮肤检测平台")

        # 扩展到16张示例图片，左右缓慢移动
        example_images = [
            "https://img.caimei365.com/group1/M00/00/90/rB-lF2R39-OAHQQkAAB9wI3HLH4661.jpg",  "https://tse2.mm.bing.net/th/id/OIP.DQlOipLSs0-N3_96hUfokwAAAA?rs=1&pid=ImgDetMain&o=7&rm=3",
            "https://pic4.zhimg.com/v2-29ad4557685061400b9043c0d31210ef_r.jpg",  "https://th.bing.com/th/id/R.b1d993a7b36e04deaae55749e6d308ea?rik=ElWZbshKU4UvEA&riu=http%3a%2f%2fwww.poolingmed.com%2fuploads%2fallimg%2f230828%2f3-230RQ0125V40.png&ehk=QNYl27LAt55%2bXmzqReKD7uASD5X0rJFtaoBLpt5kfnA%3d&risl=&pid=ImgRaw&r=0",
            "https://tse2.mm.bing.net/th/id/OIP.-uVPJPQGEIRnRPlnqIDrYwHaE7?rs=1&pid=ImgDetMain&o=7&rm=3",  "https://img95.699pic.com/photo/40243/6814.jpg_wh300.jpg!/fh/300/quality/90",
            "https://tse1.mm.bing.net/th/id/OIP.em_bhwpX_QdIxgy-OqoWsQHaEr?rs=1&pid=ImgDetMain&o=7&rm=3",  "https://tse1.mm.bing.net/th/id/OIP.3L9VXDb6tcx3D_-NSzf4PgHaFG?rs=1&pid=ImgDetMain&o=7&rm=3",
            "https://tse2.mm.bing.net/th/id/OIP.L6oM_fN6Y-_aL-wTt60X5QHaE8?rs=1&pid=ImgDetMain&o=7&rm=3",  "https://tse2.mm.bing.net/th/id/OIP.smIa2wnuvdNfw4so9yeXKgHaEK?rs=1&pid=ImgDetMain&o=7&rm=3",
            "https://tse2.mm.bing.net/th/id/OIP.itAUXhoncNybr1gibv9tpAHaHa?rs=1&pid=ImgDetMain&o=7&rm=3", "https://www.tsinghua.edu.cn/__local/0/51/D0/1DABE2068E43934F40D71BBABFA_3EA7F33A_C2E9.jpeg",
            "https://tse2.mm.bing.net/th/id/OIP.yOCsSiqWfIOBk_GCTvEm3QHaH4?rs=1&pid=ImgDetMain&o=7&rm=3", "https://omo-oss-image.thefastimg.com/portal-saas/new2022093017322217252/cms/image/404a4453-a53f-4165-81f6-c436bc9d542e.jpg",
            "https://tse4.mm.bing.net/th/id/OIP.6vtiRdZeOsycCXY-5ZQRvAHaE7?rs=1&pid=ImgDetMain&o=7&rm=3", "https://tse4.mm.bing.net/th/id/OIP.0pWcWQHPPLhRZmBQSNGEiAHaE8?rs=1&pid=ImgDetMain&o=7&rm=3",
        ]

        # —— 静态示例图片区块 ——
        static_urls = [
            "https://file.youlai.cn/cnkfile1/M00/15/DD/o4YBAFmDDHOAdqt8AAPwJoMLhxA75.jpeg",
            "https://file.youlai.cn/cnkfile1/M00/13/EA/o4YBAFlE-IaAc222AAHhT-U_cMQ95.jpeg",
            "https://ts1.tc.mm.bing.net/th/id/R-C.9b31695bebedc66344e3b340b80f8f3f?rik=UO%2bLLJFpYv%2f6KQ&riu=http%3a%2f%2fiiyi4.120askimages.com%2fbingli%2f130129%2f5942757019.jpg&ehk=uod4PmVdafrEzRHtos1tMmSocaMG0H7WGFtDXTpLdRM%3d&risl=&pid=ImgRaw&r=0",
            "https://pic4.zhimg.com/v2-04d7472b9dcb6b461a14a059f0c5dadf_r.jpg",
            "https://file.fh21static.com/fhfile1/M00/05/FE/o4YBAF_Z9iOACqoKAADT5ouIa2A53.jpeg",
            "https://kano-sns.guahao.cn/EGI464084158",
            "https://ts4.tc.mm.bing.net/th/id/OIP-C.ypO-xoXiUGYPhiPUGAxoeQHaE8?rs=1&pid=ImgDetMain&o=7&rm=3",
            "https://file.youlai.cn/cnkfile1/M00/15/BF/ooYBAFmDC0aATqf3AAXIc9deRBI82.jpeg",
        ]

        log_info("加载前端配置")
        # 自定义 JavaScript 代码实现拖拽功能
        custom_css, intro_content, benefit_content = bc.front_end_instantiation()

        log_info("创建Gradio界面")
        with gr.Blocks(css=custom_css, title="LittleSkin - 智能皮肤检测平台") as demo:

            css_to_js(example_images, static_urls, intro_content, benefit_content)

            img_and_text_module()

        log_info("启动Web服务器，自动选择端口")
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=False)

    except KeyboardInterrupt:
        log_info("用户中断程序运行")
    except Exception as e:
        log_exception(f"程序启动失败: {str(e)}")
        raise
    finally:
        log_info("=== LittleSkin智能皮肤检测平台关闭 ===")
