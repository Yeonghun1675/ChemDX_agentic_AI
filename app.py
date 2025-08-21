import streamlit as st
import asyncio
import os
import json
import asyncio
from pathlib import Path

from chemdx_agent.schema import AgentState
from chemdx_agent.main_agent import run_main_agent
from chemdx_agent.utils import split_line_to_agent_and_message


agent_color = {
    'MainAgent': '#f0f2f6',
    'SampleAgent': '#e6f3ff',
    'ColorTrendAgent': '#e6ffe6',
    'PhosphorDataResearchAgent': '#fff2e6',
    'MatDXTrendAgent': '#e6e6ff',
    'ESTMTrendAgent': '#e6f9ff',
    'PhosphorTrendAgent': '#ffe6f2',
    'PhosphorLookupAgent': '#f9e6ff',
    'RecommendAgent': '#e6fff9',
    'TrendAgent': '#fff0b3',
}

agent_emoji = {
    "MainAgent": "💻",
    "SampleAgent": "🔍",
    'ColorTrendAgent': '🎨',
    'PhosphorDataResearchAgent': '📊',
    'MatDXTrendAgent': '🧪',
    'ESTMTrendAgent': '🌍',
    'PhosphorTrendAgent': '💡',
    'PhosphorLookupAgent': '🔎',
    'RecommendAgent': '🤖',
    'TrendAgent': '📈',
}



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

def _process_message_content(message):
    """Process message content and convert JSON to bullet points if possible"""
    try:
        message_json = json.loads(message.strip())
        new_message = ""
        for key, value in message_json.items():
            new_message += f"• {key}: {value}\n\n"
        return new_message.strip()
    except:
        return message

def _get_agent_styling(agent):
    """Get color, emoji, and tool status for an agent"""
    if agent.startswith("Tool-"):
        return "#9A9DEB", "🔧", agent.replace("Tool-", ""), True
    else:
        color = agent_color.get(agent, "#f0f2f6")
        emoji = agent_emoji.get(agent, "🤖")
        return color, emoji, agent, False

def _display_line(line, displayed_lines):
    """Display a single line if it hasn't been displayed yet"""
    if line in displayed_lines:
        return False
    
    try:
        agent, message_type, message = split_line_to_agent_and_message(line)
        message = _process_message_content(message)
        color, emoji, agent_name, is_tool = _get_agent_styling(agent)
        
        message_box(agent_name, message_type, message, color, emoji, is_tool)
        displayed_lines.add(line)
        return True
    except Exception as e:
        # Display as default if parsing fails
        message_box("System", "Log", line, "#f0f2f6")
        displayed_lines.add(line)
        return True

def generate_mermaid_flowchart(agent_sequence):
    """
    Generate mermaid flowchart code from agent sequence
    
    Args:
        agent_sequence: List of agent names like ['MainAgent', 'SampleAgent', ...]
    
    Returns:
        str: Mermaid flowchart code
    """
    if not agent_sequence:
        return ""
    
    # Remove consecutive duplicates
    cleaned_sequence = []
    for i, agent in enumerate(agent_sequence):
        if i == 0 or agent != agent_sequence[i-1]:
            cleaned_sequence.append(agent)
    
    # Add start and end
    if cleaned_sequence:
        cleaned_sequence = ['start'] + cleaned_sequence + ['end']
    
    # Generate mermaid code
    mermaid_code = "```mermaid\nflowchart TD\n"
    
    # Add nodes
    for i, agent in enumerate(cleaned_sequence):
        if agent == 'start':
            mermaid_code += f"    {agent}[Start]\n"
        elif agent == 'end':
            mermaid_code += f"    {agent}[End]\n"
        else:
            mermaid_code += f"    {agent}[{agent}]\n"
    
    # Add connections
    for i in range(len(cleaned_sequence) - 1):
        current = cleaned_sequence[i]
        next_agent = cleaned_sequence[i + 1]
        mermaid_code += f"    {current} --> {next_agent}\n"
    
    mermaid_code += "```"
    
    return mermaid_code

def extract_agent_sequence_from_log():
    """Extract agent sequence from log.txt file"""
    agent_sequence = []
    log_file_path = Path("log.txt")
    
    if not log_file_path.exists():
        return agent_sequence
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        agent, _, _ = split_line_to_agent_and_message(line)
                        agent_sequence.append(agent)
                    except:
                        continue
    except Exception as e:
        st.error(f"Error reading log file: {e}")
    
    return agent_sequence

def display_agent_flowchart(agent_sequence):
    """Display mermaid flowchart in Streamlit with expandable section"""
    mermaid_code = generate_mermaid_flowchart(agent_sequence)
    if mermaid_code:
        with st.expander("📊 Agent Flow Chart", expanded=False):
            st.markdown("### Agent Execution Flow")
            st.code(mermaid_code, language="mermaid")
            st.markdown(mermaid_code)

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
                
                if not new_content:
                    await asyncio.sleep(0.1)
                    continue
                
                # Add new lines
                new_lines = new_content.split('\n')
                for line in new_lines:
                    if line.strip() and line not in accumulated_lines:
                        accumulated_lines.append(line)
                
                # Display only new lines
                for line in accumulated_lines:
                    _display_line(line, displayed_lines)
                
                last_position = f.tell()
                await asyncio.sleep(0.1)  # Check every 100ms
                
        except Exception as e:
            break

async def run_main_agent_with_logging(question):
    """Run main_agent (logging file handler already outputs to log.txt)"""
    # Run main_agent
    state = AgentState()
    result = await run_main_agent(question, state)
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
            
            # Display agent flow chart at the end
            st.markdown("---")
            agent_sequence = extract_agent_sequence_from_log()
            if agent_sequence:
                display_agent_flowchart(agent_sequence)
            else:
                st.info("No agent execution log found.")

        else:
            st.warning('Please enter a question.')

if __name__ == '__main__':
    run_demo()