import streamlit as st
import asyncio
import os
import re
import json
import asyncio
from pathlib import Path

from chemdx_agent.main_agent import run_main_agent


agent_color = {
    'MainAgent': '#f0f2f6',
    'SampleAgent': '#e6f3ff',
    'Message2Agent': '#e6ffe6',
    'ToolAgent': '#fff2e6',
    'Error': '#ffe6e6',
    'Success': '#e6ffe6',
    'Warning': '#fff2e6',
}

agent_emoji = {
    "MainAgent": "💻",
    "SampleAgent": "🔍",
}

def split_line_to_agent_and_message(line: str):
    if line.startswith("[Question]"):
        return "MainAgent", "Question", line.replace("[Question]", "").strip()
    elif line.startswith("[Final Answer]"):
        return "MainAgent", "Final Answer", line.replace("[Final Answer]", "").strip() 
    elif line.startswith("[Evaluation]"):
        return "MainAgent", "Evaluation", line.replace("[Evaluation]", "").strip()
    else:
        agent, message_type, message = re.match(r"^\[(.+?)\](.+?):(.+)", line).groups()

        return agent.strip(), message_type.strip(), message.strip()

def message_box(agent_name, message_type, content, color="#f0f2f6", emoji="🤖", is_tool=False):
    margin_left = "margin-left: 20px;" if is_tool else ""
    message_type_color = "#333333" if is_tool else "gray"  # Tool일 때 더 진한 색상
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:10px; border-radius:10px; margin-bottom:10px; {margin_left}">
            {emoji} <strong>{agent_name}</strong> <span style="color:{message_type_color}; font-size:12px;">[{message_type}]</span><br>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )

async def monitor_log_file(log_file_path, placeholder):
    """Monitor log.txt file in real-time and stream to Streamlit (asyncio-based)"""
    if not os.path.exists(log_file_path):
        return
    
    last_position = 0
    accumulated_lines = []
    displayed_lines = set()  # Track already displayed lines
    
    while True:
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                f.seek(last_position)
                new_content = f.read()
                if new_content:
                    # Add new lines
                    new_lines = new_content.split('\n')
                    for line in new_lines:
                        if line.strip() and line not in accumulated_lines:
                            accumulated_lines.append(line)
                    
                    # Display only new lines (skip already displayed ones)
                    for line in accumulated_lines:
                        if line not in displayed_lines:
                            try:
                                agent, message_type, message = split_line_to_agent_and_message(line)

                                try:
                                    message_json = json.loads(message.strip())
                                    new_message = ""
                                    for key, value in message_json.items():
                                        new_message += f"• {key}: {value}\n\n"
                                    message = new_message.strip()
                                except:
                                    pass
                                
                                is_tool = False
                                if agent.startswith("Tool-"):
                                    color = "#9A9DEB"
                                    emoji = "🔧"
                                    agent = agent.replace("Tool-", "")
                                    is_tool = True
                                else:
                                    color = agent_color.get(agent, "#f0f2f6")
                                    emoji = agent_emoji.get(agent, "🤖")          

                                message_box(agent, message_type, message, color, emoji, is_tool)
                                
                                # Track completed lines
                                displayed_lines.add(line)
                            except Exception as e:
                                # Display as default if parsing fails
                                message_box("System", "Log", line, "#f0f2f6")
                                displayed_lines.add(line)
                    
                    last_position = f.tell()
                await asyncio.sleep(0.1)  # Check every 100ms
        except Exception as e:
            break

async def run_main_agent_with_logging(question):
    """Run main_agent (logging file handler already outputs to log.txt)"""
    # Run main_agent
    result = await run_main_agent(question)
    await asyncio.sleep(2)
    return result

async def run_demo_async(question):
    """Run main_agent and log monitoring simultaneously using asyncio"""
    # Create placeholder for log output
    log_placeholder = st.empty()
    
    # log.txt file path
    log_file_path = Path("log.txt")
    
    # Start log monitoring and main_agent execution simultaneously
    log_task = asyncio.create_task(monitor_log_file(log_file_path, log_placeholder))
    agent_task = asyncio.create_task(run_main_agent_with_logging(question))
    
    # Wait for both tasks to complete
    result = await agent_task
    log_task.cancel()  # Stop log monitoring
    
    return result


verbose = True
search_internet = False
default_openai_key = None

title = "ChemDX Agentic AI!"
description = """- We are building a agentic AI system for the ChemDX database.

### Start!
"""

def run_demo():
    st.title(title)
    st.write(description)

    col1, col2 = st.columns((8, 1))
    with col1:
        input_question = st.text_input('Enter your question 👇')

    with col2:
        st.write("\n")
        start_button = st.button('Start')

    if start_button:
        if input_question:
            st.subheader('Running...')
            
            # Run using asyncio
            text = asyncio.run(run_demo_async(input_question))
            
            st.subheader('Final Answer')
            
            # Display each field of FinalAnswer object separately
            try:
                # Check if text is FinalAnswer object and extract each field
                if hasattr(text, 'task') and hasattr(text, 'final_answer') and hasattr(text, 'evaluation'):
                    # If it's a FinalAnswer object
                    st.markdown("**Task:**")
                    st.info(text.task)
                    
                    st.markdown("**Final Answer:**")
                    st.success(text.final_answer)
                    
                    st.markdown("**Evaluation:**")
                    st.warning(text.evaluation)
                else:
                    # If it's regular text, display using existing method
                    st.text_area('', value=text, height=100,
                                 max_chars=None, key=None)
            except Exception as e:
                # Display using existing method if error occurs
                st.text_area('', value=text, height=100,
                             max_chars=None, key=None)

        else:
            st.warning('Please enter a question.')

if __name__ == '__main__':
    run_demo()