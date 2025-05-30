import streamlit as st
import autogen
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
from io import BytesIO
import PyPDF2
from PIL import Image
import pytesseract
import requests
import logging
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import sqlite3
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Database setup
DB_PATH = "kyc_data.db"

def init_database():
    """Initialize SQLite database for storing KYC records"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kyc_validations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            validation_id TEXT UNIQUE,
            timestamp TEXT,
            decision TEXT,
            risk_level TEXT,
            supporting_filename TEXT,
            bank_filename TEXT,
            processing_time REAL,
            created_date TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_validation_record(result: Dict[str, Any], processing_time: float):
    """Save validation record to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO kyc_validations 
            (validation_id, timestamp, decision, risk_level, supporting_filename, 
             bank_filename, processing_time, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get("validation_id"),
            result.get("timestamp"),
            result.get("decision"),
            result.get("risk_level"),
            result.get("documents", {}).get("supporting_document", {}).get("filename"),
            result.get("documents", {}).get("bank_document", {}).get("filename"),
            processing_time,
            datetime.now().strftime("%Y-%m-%d")
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to save validation record: {e}")

def get_dashboard_data():
    """Get dashboard data from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get today's validations
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM kyc_validations WHERE created_date = ?",
            conn, params=[today]
        ).iloc[0]['count']
        
        # Get total validations for comparison (yesterday)
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        yesterday_count = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM kyc_validations WHERE created_date = ?",
            conn, params=[yesterday]
        ).iloc[0]['count']
        
        # Calculate percentage change
        today_delta = 0
        if yesterday_count > 0:
            today_delta = ((today_count - yesterday_count) / yesterday_count) * 100
        
        # Get approval rate
        total_validations = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM kyc_validations WHERE created_date >= ?",
            conn, params=[(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")]
        ).iloc[0]['count']
        
        approved_validations = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM kyc_validations WHERE decision = 'APPROVED' AND created_date >= ?",
            conn, params=[(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")]
        ).iloc[0]['count']
        
        approval_rate = (approved_validations / total_validations * 100) if total_validations > 0 else 0
        
        # Get pending reviews (REQUIRES_REVIEW)
        pending_count = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM kyc_validations WHERE decision = 'REQUIRES_REVIEW'",
            conn
        ).iloc[0]['count']
        
        # Get average processing time
        avg_time = pd.read_sql_query(
            "SELECT AVG(processing_time) as avg_time FROM kyc_validations WHERE created_date >= ?",
            conn, params=[(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")]
        ).iloc[0]['avg_time']
        
        avg_time = avg_time if avg_time else 0
        
        # Get weekly trend data
        weekly_data = pd.read_sql_query('''
            SELECT created_date, COUNT(*) as validations
            FROM kyc_validations 
            WHERE created_date >= ?
            GROUP BY created_date
            ORDER BY created_date
        ''', conn, params=[(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")])
        
        # Get decision distribution
        decision_data = pd.read_sql_query('''
            SELECT decision, COUNT(*) as count
            FROM kyc_validations 
            WHERE created_date >= ?
            GROUP BY decision
        ''', conn, params=[(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")])
        
        conn.close()
        
        return {
            "today_count": today_count,
            "today_delta": today_delta,
            "approval_rate": approval_rate,
            "pending_count": pending_count,
            "avg_time": avg_time,
            "weekly_data": weekly_data,
            "decision_data": decision_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        return {
            "today_count": 0,
            "today_delta": 0,
            "approval_rate": 0,
            "pending_count": 0,
            "avg_time": 0,
            "weekly_data": pd.DataFrame(),
            "decision_data": pd.DataFrame()
        }

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def login_page():
    """Display simplified login page"""
    st.set_page_config(
        page_title="KYC Login",
        page_icon="🔐",
        layout="wide"
    )
    
    # Custom CSS for login page
    st.markdown("""
        <style>
        .login-title {
            font-size: 3rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 1rem;
            text-align: center;
        }
        .login-subtitle {
            font-size: 1.5rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .main-content {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 80vh;
            flex-direction: column;
        }
        .stTextInput > div > div > input {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for login form
    with st.sidebar:
        st.markdown("### 🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if username == "aiplanet" and password == "aiplanet000":
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
     
    # Main content area - Application name and subtitle
    st.markdown("""
        <div class="main-content">
            <h1 class='login-title'>KYC Validation System</h1>
            <p class='login-subtitle'>Please login to continue</p>
        </div>
    """, unsafe_allow_html=True)

class SimpleDocumentProcessor:
    """Simple document processor focused on text extraction"""
    
    def __init__(self, azure_endpoint: str = None, azure_key: str = None):
        self.azure_endpoint = azure_endpoint
        self.azure_key = azure_key
    
    def extract_text_from_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Extract text from document using Azure Form Recognizer or fallback"""
        
        if self.azure_endpoint and self.azure_key:
            # Try Azure Form Recognizer first
            text = self._azure_extract_text(file_bytes)
            if text:
                return {
                    "text": text,
                    "method": "Azure Form Recognizer",
                    "filename": filename
                }
        
        # Fallback to basic extraction
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            text = self._extract_from_pdf(file_bytes)
        elif file_extension in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
            text = self._extract_from_image(file_bytes)
        else:
            text = file_bytes.decode('utf-8', errors='ignore')
        
        return {
            "text": text,
            "method": "Basic extraction",
            "filename": filename
        }
    
    def _azure_extract_text(self, file_bytes: bytes) -> str:
        """Extract text using Azure Form Recognizer API"""
        try:
            # Use the Read API for general text extraction
            analyze_url = f"{self.azure_endpoint}/formrecognizer/documentModels/prebuilt-read:analyze?api-version=2023-07-31"
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.azure_key,
                'Content-Type': 'application/octet-stream'
            }
            
            # Start analysis
            response = requests.post(analyze_url, headers=headers, data=file_bytes)
            if response.status_code != 202:
                logger.warning(f"Azure API returned status {response.status_code}")
                return ""
            
            # Get operation location
            operation_location = response.headers.get("Operation-Location")
            if not operation_location:
                return ""
            
            # Poll for results
            headers_get = {'Ocp-Apim-Subscription-Key': self.azure_key}
            
            import time
            max_attempts = 30
            for _ in range(max_attempts):
                result = requests.get(operation_location, headers=headers_get)
                result_json = result.json()
                
                if result_json.get("status") == "succeeded":
                    # Extract all text content
                    content = result_json.get("analyzeResult", {}).get("content", "")
                    return content
                elif result_json.get("status") == "failed":
                    logger.error("Azure analysis failed")
                    return ""
                
                time.sleep(2)
            
            logger.warning("Azure analysis timeout")
            return ""
            
        except Exception as e:
            logger.error(f"Azure extraction failed: {e}")
            return ""
    
    def _extract_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            text_parts = []
            
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def _extract_from_image(self, image_bytes: bytes) -> str:
        """Extract text from image using Tesseract"""
        try:
            image = Image.open(BytesIO(image_bytes))
            return pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            return ""

class SimpleKYCSystem:
    """Simple KYC system with two agents: Reviewer and Validator"""
    
    def __init__(self, azure_openai_endpoint: str, api_key: str, model_name: str, 
                azure_form_endpoint: str = None, azure_form_key: str = None):
        
        self.document_processor = SimpleDocumentProcessor(azure_form_endpoint, azure_form_key)
        self.agent_interactions = []  # Store agent interactions for visualization
        
        # Setup AutoGen configuration
        self.llm_config = {
            "config_list": [
                {
                    "model": model_name,
                    "api_key": api_key,
                    "base_url": azure_openai_endpoint,
                    "api_type": "azure",
                    "api_version": "2024-02-15-preview"
                }
            ],
            "temperature": 0.1,
            "timeout": 60,
        }
        
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup the two agents"""
        
        # Reviewer Agent - extracts and compares information
        self.reviewer_agent = autogen.ConversableAgent(
            name="KYC_Reviewer",
            system_message="""You are a document reviewer. Your job is to:

1. Extract key information from both documents:
   - Given Name
   - Date of birth
   - Address
   - ID numbers
   - Account numbers (from bank documents)
   - Any other personal details

2. Compare the information between documents and identify:
   - Matching information
   - Discrepancies
   - Missing information

Present your findings in a clear, structured format listing what matches, what doesn't match, and what's missing.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # Validator Agent - makes final decision
        self.validator_agent = autogen.ConversableAgent(
            name="KYC_Validator", 
            system_message="""You are a KYC validator. Based on the reviewer's findings, you must:

1. Generate a validation report with:
   - APPROVED/REJECTED/REQUIRES_REVIEW status
   - Risk level (LOW/MEDIUM/HIGH)
   - Specific reasons for your decision
   - Required actions if any

2. Focus on:
   - Identity verification
   - Information consistency
   - Document completeness

Keep your report concise and actionable.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
    
    def validate_documents(self, supporting_doc_bytes: bytes, supporting_filename: str,
                          bank_doc_bytes: bytes, bank_filename: str) -> Dict[str, Any]:
        """Main validation process"""
        
        try:
            # Reset agent interactions
            self.agent_interactions = []
            
            # Extract text from both documents
            supporting_data = self.document_processor.extract_text_from_document(
                supporting_doc_bytes, supporting_filename
            )
            
            bank_data = self.document_processor.extract_text_from_document(
                bank_doc_bytes, bank_filename
            )
            
            # Prepare context for agents
            context = f"""
Please analyze these KYC documents:

SUPPORTING DOCUMENT ({supporting_data['filename']}):
{supporting_data['text'][:3000]}

BANK DOCUMENT ({bank_data['filename']}):
{bank_data['text'][:3000]}

Please extract key information from both documents and compare them for KYC validation.
"""
            
            # Start the conversation
            chat_result = self.reviewer_agent.initiate_chat(
                self.validator_agent,
                message=context,
                max_turns=4
            )
            
            # Process the results
            return self._process_chat_result(chat_result, supporting_data, bank_data)
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _process_chat_result(self, chat_result, supporting_data: Dict, bank_data: Dict) -> Dict[str, Any]:
        """Process the chat result into a structured report"""
        
        # Extract messages from chat and store interactions
        messages = []
        if hasattr(chat_result, 'chat_history'):
            messages = chat_result.chat_history
        else:
            messages = [{"content": "Analysis completed", "name": "system"}]
        
        # Process agent interactions for visualization
        for i, msg in enumerate(messages):
            agent_name = msg.get('name', 'System')
            content = msg.get('content', '')
            
            self.agent_interactions.append({
                "step": i + 1,
                "agent": agent_name,
                "message": content[:500] + "..." if len(content) > 500 else content,
                "timestamp": datetime.now().isoformat(),
                "message_type": "request" if i % 2 == 0 else "response"
            })
        
        # Find reviewer and validator responses
        reviewer_analysis = ""
        validator_decision = ""
        
        for msg in messages:
            content = msg.get('content', '')
            name = msg.get('name', '')
            
            if 'reviewer' in name.lower():
                reviewer_analysis = content
            elif 'validator' in name.lower():
                validator_decision = content
        
        # Extract decision from validator response
        decision = "REQUIRES_REVIEW"  # default
        if "APPROVED" in validator_decision.upper():
            decision = "APPROVED"
        elif "REJECTED" in validator_decision.upper():
            decision = "REJECTED"
        
        # Extract risk level
        risk_level = "MEDIUM"  # default
        if "LOW" in validator_decision.upper():
            risk_level = "LOW"
        elif "HIGH" in validator_decision.upper():
            risk_level = "HIGH"
        
        return {
            "validation_id": f"KYC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "status": "COMPLETED",
            "decision": decision,
            "risk_level": risk_level,
            "documents": {
                "supporting_document": {
                    "filename": supporting_data['filename'],
                    "extraction_method": supporting_data['method'],
                    "text_length": len(supporting_data['text'])
                },
                "bank_document": {
                    "filename": bank_data['filename'],
                    "extraction_method": bank_data['method'],
                    "text_length": len(bank_data['text'])
                }
            },
            "reviewer_analysis": reviewer_analysis,
            "validator_decision": validator_decision,
            "agent_interactions": self.agent_interactions,
            "extracted_text": {
                "supporting_document": supporting_data['text'][:1000] + "..." if len(supporting_data['text']) > 1000 else supporting_data['text'],
                "bank_document": bank_data['text'][:1000] + "..." if len(bank_data['text']) > 1000 else bank_data['text']
            }
        }

def create_dashboard():
    """Create dynamic KPI dashboard"""
    st.markdown("### 📊 KYC Dashboard")
    
    # Get dynamic data
    dashboard_data = get_dashboard_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_str = f"↑ {dashboard_data['today_delta']:.1f}%" if dashboard_data['today_delta'] > 0 else f"↓ {abs(dashboard_data['today_delta']):.1f}%" if dashboard_data['today_delta'] < 0 else "No change"
        st.metric(
            label="📈 Today's Validations",
            value=str(dashboard_data['today_count']),
            delta=delta_str
        )
    
    with col2:
        st.metric(
            label="✅ Approval Rate",
            value=f"{dashboard_data['approval_rate']:.1f}%",
            delta="Last 7 days"
        )
    
    with col3:
        st.metric(
            label="⚠️ Pending Reviews",
            value=str(dashboard_data['pending_count']),
            delta="Total pending"
        )
    
    with col4:
        st.metric(
            label="⏱️ Avg Process Time",
            value=f"{dashboard_data['avg_time']:.1f} sec",
            delta="Last 7 days"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekly validation trend
        if not dashboard_data['weekly_data'].empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dashboard_data['weekly_data']['created_date'],
                y=dashboard_data['weekly_data']['validations'],
                mode='lines+markers',
                name='Validations',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title="📈 Weekly Validation Trend",
                height=200,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No validation data available for trend chart")
    
    with col2:
        # Decision distribution
        if not dashboard_data['decision_data'].empty:
            labels = dashboard_data['decision_data']['decision'].tolist()
            values = dashboard_data['decision_data']['count'].tolist()
            colors = ['#2E8B57', '#DC143C', '#FF8C00']
            
            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values,
                hole=0.5,
                marker_colors=colors[:len(labels)]
            )])
            fig.update_layout(
                title="🎯 Decision Distribution (Last 30 days)",
                height=200,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No decision data available for pie chart")

def display_agent_communication(agent_interactions: List[Dict]):
    """Display agent communication with better UI"""
    st.subheader("🤖 Agent Communication Flow")
    
    if not agent_interactions:
        st.info("No agent interactions to display")
        return
    
    # Create a timeline visualization
    for i, interaction in enumerate(agent_interactions):
        agent_name = interaction.get('agent', 'Unknown')
        message = interaction.get('message', '')
        step = interaction.get('step', i + 1)
        msg_type = interaction.get('message_type', 'info')
        
        # Determine agent color and icon
        if 'reviewer' in agent_name.lower():
            agent_color = "#1f77b4"
            agent_icon = "🔍"
            agent_display = "KYC Reviewer"
        elif 'validator' in agent_name.lower():
            agent_color = "#2ca02c"
            agent_icon = "✅"
            agent_display = "KYC Validator"
        else:
            agent_color = "#ff7f0e"
            agent_icon = "🤖"
            agent_display = agent_name
        
        # Create expandable message container
        with st.container():
            col1, col2 = st.columns([1, 10])
            
            with col1:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {agent_color}; 
                        color: white; 
                        padding: 10px; 
                        border-radius: 50%; 
                        text-align: center; 
                        width: 50px; 
                        height: 50px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center;
                        font-size: 20px;
                        margin: 10px 0;
                    ">
                        {agent_icon}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f0f2f6; 
                        padding: 15px; 
                        border-radius: 10px; 
                        margin: 10px 0;
                        border-left: 4px solid {agent_color};
                    ">
                        <strong style="color: {agent_color};">{agent_display} - Step {step}</strong>
                        <div style="margin-top: 8px; color: #333;">
                            {message}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Add connection line (except for last item)
        if i < len(agent_interactions) - 1:
            st.markdown(
                """
                <div style="
                    margin-left: 25px; 
                    width: 2px; 
                    height: 20px; 
                    background-color: #ddd;
                "></div>
                """, 
                unsafe_allow_html=True
            )

def create_main_ui():
    """Create main Streamlit UI"""
    
    st.set_page_config(
        page_title="KYC Validation System",
        page_icon="🏦",
        layout="wide"
    )
    
    # Top navigation with logout
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🏦 KYC Document Validation System")
        st.markdown(f"Welcome, **{st.session_state.get('username', 'User')}**!")
    
    with col2:
        if st.button("🚪 Logout", type="secondary"):
            st.session_state["authenticated"] = False
            st.session_state.pop("username", None)
            st.rerun()
    
    st.markdown("Upload two documents to compare and validate KYC information")
    
    # Add dashboard
    create_dashboard()
    st.divider()
    
    # Load configuration from environment variables
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    model_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-4")
    azure_form_endpoint = os.getenv("AZURE_FORM_ENDPOINT", "")
    azure_form_key = os.getenv("AZURE_FORM_KEY", "")
    
    # Check configuration
    if not azure_openai_endpoint or not openai_api_key:
        st.error("❌ Azure OpenAI configuration missing. Please check your .env file.")
        st.info("Required environment variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY")
        return
    
    # Sidebar with recent activity only
    with st.sidebar:
        st.header("📊 Recent Activity")
        try:
            conn = sqlite3.connect(DB_PATH)
            recent_validations = pd.read_sql_query('''
                SELECT validation_id, decision, timestamp 
                FROM kyc_validations 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''', conn)
            conn.close()
            
            if not recent_validations.empty:
                for _, row in recent_validations.iterrows():
                    decision_icon = {"APPROVED": "✅", "REJECTED": "❌", "REQUIRES_REVIEW": "⚠️"}.get(row['decision'], "❓")
                    validation_short_id = row['validation_id'][-8:] if row['validation_id'] else "Unknown"
                    timestamp = datetime.fromisoformat(row['timestamp']).strftime("%m/%d %H:%M") if row['timestamp'] else "N/A"
                    st.markdown(f"{decision_icon} **{validation_short_id}**  \n*{timestamp}*")
            else:
                st.write("No recent validations")
        except Exception as e:
            st.write("No data available")
        
        # Configuration status
        st.divider()
        st.subheader("⚙️ Configuration Status")
        st.success("✅ Azure OpenAI configured")
        if azure_form_endpoint and azure_form_key:
            st.success("✅ Azure Form Recognizer enabled")
        else:
            st.info("ℹ️ Using basic text extraction")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📄 Supporting Document")
        st.caption("ID, Passport, Driver's License, etc.")
        
        supporting_file = st.file_uploader(
            "Upload supporting document",
            type=['pdf', 'jpg', 'jpeg', 'png'],
            key="supporting"
        )
        
        if supporting_file:
            st.success(f"✅ {supporting_file.name}")
            if supporting_file.type.startswith('image'):
                st.image(supporting_file, caption="Preview", use_column_width=True)
    
    with col2:
        st.header("🏛️ Bank Document")
        st.caption("Bank statement, account form, etc.")
        
        bank_file = st.file_uploader(
            "Upload bank document",
            type=['pdf', 'jpg', 'jpeg', 'png'],
            key="bank"
        )
        
        if bank_file:
            st.success(f"✅ {bank_file.name}")
            if bank_file.type.startswith('image'):
                st.image(bank_file, caption="Preview", use_column_width=True)
    
    # Validation button
    st.divider()
    
    if st.button("🚀 Validate Documents", type="primary", use_container_width=True):
        
        # Check inputs
        if not supporting_file or not bank_file:
            st.error("Please upload both documents")
            return
        
        # Process documents
        start_time = datetime.now()
        with st.spinner("Processing documents..."):
            try:
                # Initialize KYC system
                kyc_system = SimpleKYCSystem(
                    azure_openai_endpoint=azure_openai_endpoint,
                    api_key=openai_api_key,
                    model_name=model_name,
                    azure_form_endpoint=azure_form_endpoint,
                    azure_form_key=azure_form_key
                )
                
                # Validate documents
                result = kyc_system.validate_documents(
                    supporting_file.read(), supporting_file.name,
                    bank_file.read(), bank_file.name
                )
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Save to database
                if result.get("status") != "ERROR":
                    save_validation_record(result, processing_time)
                
                # Display results
                display_results(result)
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                
                with st.expander("Error Details"):
                    st.code(str(e))

def display_results(result: Dict[str, Any]):
    """Display validation results"""
    
    if result.get("status") == "ERROR":
        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        return
    
    st.success("✅ Validation Completed!")
    
    # Main results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        decision = result.get("decision", "UNKNOWN")
        decision_color = {
            "APPROVED": "🟢",
            "REJECTED": "🔴", 
            "REQUIRES_REVIEW": "🟡"
        }.get(decision, "⚪")
        st.metric("Decision", f"{decision_color} {decision}")
    
    with col2:
        risk_level = result.get("risk_level", "UNKNOWN")
        risk_color = {
            "LOW": "🟢",
            "MEDIUM": "🟡",
            "HIGH": "🔴"
        }.get(risk_level, "⚪")
        st.metric("Risk Level", f"{risk_color} {risk_level}")
    
    with col3:
        st.metric("Validation ID", result.get("validation_id", "N/A")[-8:])
    
    st.divider()
    
    # Agent Communication Flow
    agent_interactions = result.get("agent_interactions", [])
    if agent_interactions:
        display_agent_communication(agent_interactions)
        st.divider()
    
    # Document processing info
    st.subheader("📋 Document Processing")
    
    docs_info = []
    for doc_type, doc_data in result.get("documents", {}).items():
        docs_info.append({
            "Document": doc_type.replace("_", " ").title(),
            "Filename": doc_data.get("filename", "N/A"),
            "Extraction Method": doc_data.get("extraction_method", "N/A"),
            "Text Length": f"{doc_data.get('text_length', 0):,} characters"
        })
    
    if docs_info:
        st.dataframe(pd.DataFrame(docs_info), use_container_width=True)
    
    # Analysis results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Reviewer Analysis")
        reviewer_analysis = result.get("reviewer_analysis", "No analysis available")
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background-color: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 4px solid #1f77b4;
                    min-height: 200px;
                    font-size: 14px;
                    line-height: 1.6;
                ">
                    {reviewer_analysis}
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    with col2:
        st.subheader("✅ Validator Decision")
        validator_decision = result.get("validator_decision", "No decision available")
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background-color: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 4px solid #2ca02c;
                    min-height: 200px;
                    font-size: 14px;
                    line-height: 1.6;
                ">
                    {validator_decision}
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    # Extracted text preview
    with st.expander("📄 Extracted Text Preview"):
        extracted_text = result.get("extracted_text", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Supporting Document:**")
            st.text_area(
                "Supporting Text",
                extracted_text.get("supporting_document", "No text extracted"),
                height=200,
                key="supporting_text"
            )
        
        with col2:
            st.write("**Bank Document:**")
            st.text_area(
                "Bank Text", 
                extracted_text.get("bank_document", "No text extracted"),
                height=200,
                key="bank_text"
            )
    
    # Export report
    st.subheader("📥 Export Report")
    
    report_json = json.dumps(result, indent=2, default=str)
    st.download_button(
        label="📄 Download Full Report (JSON)",
        data=report_json,
        file_name=f"kyc_report_{result.get('validation_id', 'unknown')}.json",
        mime="application/json"
    )

def main():
    """Main application entry point"""
    # Initialize database
    init_database()
    
    # Check authentication
    if not check_authentication():
        login_page()
    else:
        create_main_ui()

if __name__ == "__main__":
    main()