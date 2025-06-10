
import streamlit as st
from typing import Optional, List, Type, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

class LLMConfig:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        max_tokens: int = 2500,
        temperature: float = 0.7,
        custom_model: Optional[Any] = None
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.custom_model = custom_model

class PlacementQuestion(BaseModel):
    question: str = Field(..., description="The placement interview question")
    difficulty: str = Field(..., description="Difficulty level: Easy/Medium/Hard")
    topic_area: str = Field(..., description="Topic area like DSA, System Design, etc.")
    answer: str = Field(..., description="Detailed answer or approach")
    coding_solution: Optional[str] = Field(None, description="Code solution if applicable")
    explanation: str = Field(..., description="Explanation of the approach")
    follow_up_questions: List[str] = Field(..., description="Potential follow-up questions")

class QuizQuestion(BaseModel):
    question: str = Field(..., description="The quiz question")
    options: List[str] = Field(..., description="Multiple choice options (4 options)")
    correct_answer: int = Field(..., description="Index of correct answer (0-3)")
    explanation: str = Field(..., description="Explanation of correct answer")
    difficulty: str = Field(..., description="Difficulty level: Easy/Medium/Hard")
    topic_area: str = Field(..., description="Topic area")
    time_limit: int = Field(default=60, description="Time limit in seconds")

class PlacementQuestionSet(BaseModel):
    company_tier: str = Field(..., description="Tier 1 or Tier 2")
    company_type: str = Field(..., description="Type of company")
    topic: str = Field(..., description="Main topic area")
    questions: List[PlacementQuestion] = Field(..., description="List of placement questions")

class QuizSet(BaseModel):
    topic: str = Field(..., description="Quiz topic")
    difficulty: str = Field(..., description="Quiz difficulty level")
    questions: List[QuizQuestion] = Field(..., description="List of quiz questions")

class StudyPlan(BaseModel):
    title: str = Field(..., description="Title of the study plan")
    company_tier: str = Field(..., description="Target company tier")
    duration: str = Field(..., description="Preparation timeline")
    daily_schedule: List[str] = Field(..., description="Daily study schedule")
    week_wise_plan: List[dict] = Field(..., description="Week-wise preparation plan")
    important_topics: List[str] = Field(..., description="Priority topics to cover")
    practice_resources: List[str] = Field(..., description="Recommended practice platforms")
    mock_interview_schedule: List[str] = Field(..., description="Mock interview timeline")

class PlacementPreparationEngine:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            llm_config = LLMConfig()
        self.llm = self._initialize_llm(llm_config)

    def _initialize_llm(self, llm_config: LLMConfig):
        if llm_config.custom_model:
            return llm_config.custom_model
        else:
            return ChatGoogleGenerativeAI(
                model=llm_config.model_name,
                google_api_key=llm_config.api_key,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature
            )
    
    def generate_placement_questions(self, topic: str, company_tier: str, company_type: str, num: int = 8) -> PlacementQuestionSet:
        parser = PydanticOutputParser(pydantic_object=PlacementQuestionSet)
        format_instructions = parser.get_format_instructions()

        prompt_template = """
        Generate {num} placement interview questions for {company_tier} companies in the {company_type} sector, focusing on {topic}.

        For each question, provide:
        1. A realistic interview question that would be asked
        2. Difficulty level (Easy/Medium/Hard)
        3. Topic area classification
        4. Detailed answer or approach
        5. Code solution if it's a coding question (use Python)
        6. Clear explanation of the approach
        7. 2-3 potential follow-up questions

        Company Tier Context:
        - Tier 1: FAANG, Microsoft, Google, Amazon, Meta, Netflix, Apple, Top unicorns
        - Tier 2: Well-established tech companies, good startups, consulting firms

        Company Type Context:
        - Product: Focus on system design, scalability, user experience
        - Service: Focus on problem-solving, client requirements, optimization
        - Fintech: Focus on security, algorithms, data handling
        - E-commerce: Focus on scale, performance, recommendation systems
        - Consulting: Focus on problem-solving, communication, business logic

        Make questions realistic and relevant to actual placement interviews.

        {format_instructions}
        """

        prompt = PromptTemplate(
            input_variables=["num", "topic", "company_tier", "company_type"],
            template=prompt_template,
            partial_variables={"format_instructions": format_instructions}
        )

        chain = prompt | self.llm
        results = chain.invoke({
            "num": num, 
            "topic": topic, 
            "company_tier": company_tier,
            "company_type": company_type
        })

        try:
            return parser.parse(results.content)
        except Exception as e:
            st.error(f"Error generating placement questions: {e}")
            return PlacementQuestionSet(
                company_tier=company_tier,
                company_type=company_type,
                topic=topic,
                questions=[]
            )

    def generate_quiz(self, topic: str, difficulty: str, num_questions: int = 10) -> QuizSet:
        parser = PydanticOutputParser(pydantic_object=QuizSet)
        format_instructions = parser.get_format_instructions()

        time_mapping = {"Easy": 45, "Medium": 60, "Hard": 90}
        time_limit = time_mapping.get(difficulty, 60)

        prompt_template = """
        Generate {num_questions} multiple choice quiz questions on {topic} with {difficulty} difficulty level.

        For each question, provide:
        1. A clear, specific question
        2. Exactly 4 multiple choice options
        3. The index (0-3) of the correct answer
        4. A detailed explanation of why the answer is correct
        5. Difficulty level: {difficulty}
        6. Topic area: {topic}
        7. Time limit: {time_limit} seconds

        Guidelines:
        - Questions should test practical knowledge and understanding
        - Options should be plausible but only one clearly correct
        - Explanations should be educational and help learning
        - Focus on concepts commonly asked in placement interviews
        - Mix theoretical and practical questions

        {format_instructions}
        """

        prompt = PromptTemplate(
            input_variables=["num_questions", "topic", "difficulty", "time_limit"],
            template=prompt_template,
            partial_variables={"format_instructions": format_instructions}
        )

        chain = prompt | self.llm
        results = chain.invoke({
            "num_questions": num_questions,
            "topic": topic,
            "difficulty": difficulty,
            "time_limit": time_limit
        })

        try:
            return parser.parse(results.content)
        except Exception as e:
            st.error(f"Error generating quiz: {e}")
            return QuizSet(
                topic=topic,
                difficulty=difficulty,
                questions=[]
            )

    def generate_study_plan(self, company_tier: str, preparation_duration: str, focus_areas: List[str]) -> StudyPlan:
        parser = PydanticOutputParser(pydantic_object=StudyPlan)
        format_instructions = parser.get_format_instructions()

        focus_areas_str = ", ".join(focus_areas)
        
        prompt_template = """
        Create a comprehensive study plan for placement preparation targeting {company_tier} companies.
        
        Preparation Duration: {preparation_duration}
        Focus Areas: {focus_areas}

        The study plan should include:
        1. A clear title
        2. Target company tier
        3. Total preparation timeline
        4. Daily study schedule (6-8 hours breakdown)
        5. Week-wise preparation plan with specific goals
        6. Priority topics to cover based on company tier
        7. Recommended practice platforms and resources
        8. Mock interview schedule

        Week-wise plan format:
        {{"week": "Week 1", "focus": "Topic Focus", "goals": ["Goal 1", "Goal 2"], "practice": "Practice recommendation"}}

        Tier 1 companies expect: Advanced DSA, System Design, Leadership Principles, Complex Problem Solving
        Tier 2 companies expect: Good DSA, Basic System Design, Problem Solving, Communication Skills

        Make it practical and achievable within the given timeframe.

        {format_instructions}
        """

        prompt = PromptTemplate(
            input_variables=["company_tier", "preparation_duration", "focus_areas"],
            template=prompt_template,
            partial_variables={"format_instructions": format_instructions}
        )

        chain = prompt | self.llm
        results = chain.invoke({
            "company_tier": company_tier,
            "preparation_duration": preparation_duration,
            "focus_areas": focus_areas_str
        })

        try:
            return parser.parse(results.content)
        except Exception as e:
            st.error(f"Error generating study plan: {e}")
            return StudyPlan(
                title=f"Placement Preparation Plan - {company_tier}",
                company_tier=company_tier,
                duration=preparation_duration,
                daily_schedule=[],
                week_wise_plan=[],
                important_topics=[],
                practice_resources=[],
                mock_interview_schedule=[]
            )

# Initialize the engine

llm_config = LLMConfig(api_key=os.getenv("GOOGLE_API_KEY"))

placement_engine = PlacementPreparationEngine(llm_config)

# Initialize session state for quiz
if 'quiz_state' not in st.session_state:
    st.session_state.quiz_state = {
        'active': False,
        'questions': [],
        'current_q': 0,
        'answers': [],
        'start_time': None,
        'question_start_time': None,
        'time_taken': [],
        'score': 0
    }

# Streamlit App
st.set_page_config(page_title="Placement Prep", page_icon="🎯", layout="wide")

st.title("🎯 Placement Prep Pro")
st.markdown("""
Complete placement preparation system for top-tier companies. Get company-specific questions, 
study plans, and time-based quizzes powered by AI!
""")

# Constants
TIER_1_COMPANIES = [
    "Google", "Microsoft", "Amazon", "Meta (Facebook)", "Apple", "Netflix", "Adobe", 
    "Salesforce", "Oracle", "IBM", "Goldman Sachs", "Morgan Stanley", "Uber", "Airbnb",
    "LinkedIn", "Twitter", "Spotify", "Dropbox", "Atlassian", "Palantir"
]

TIER_2_COMPANIES = [
    "Accenture", "TCS", "Infosys", "Wipro", "HCL", "Cognizant", "Capgemini", "Deloitte",
    "PwC", "EY", "KPMG", "Flipkart", "Paytm", "Zomato", "Swiggy", "Ola", "PhonePe",
    "Razorpay", "Freshworks", "Zoho", "Byju's", "Unacademy"
]

COMPANY_TYPES = [
    "Product Companies", "Service Companies", "Fintech", "E-commerce", 
    "Consulting", "Startups", "Banking & Finance", "Healthcare Tech"
]

TECHNICAL_TOPICS = [
    "Data Structures & Algorithms", "System Design", "Object-Oriented Programming",
    "Database Management", "Operating Systems", "Computer Networks", "Web Development",
    "Machine Learning", "Cloud Computing", "Cybersecurity", "DevOps", "Mobile Development"
]

# Sidebar for navigation
st.sidebar.title("📋 Preparation Mode")
prep_mode = st.sidebar.radio(
    "Choose preparation type:",
    ["🎯 Practice Questions", "📚 Study Plan", "🧠 Time-Based Quiz"]
)

if prep_mode == "🎯 Practice Questions":
    st.header("🎯 Practice Interview Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        company_tier = st.selectbox("🏆 Company Tier:", ["Tier 1(PBC)", "Tier 2(SBC)"])
        
    with col2:
        company_type = st.selectbox("🏢 Company Type:", COMPANY_TYPES)
        
    with col3:
        num_questions = st.number_input("❓ Number of Questions:", min_value=3, max_value=15, value=8)
    
    topic = st.selectbox("💻 Technical Topic:", TECHNICAL_TOPICS)
    
    if st.button("🚀 Generate Practice Questions", type="primary"):
        with st.spinner("🧠 Generating placement questions..."):
            question_set = placement_engine.generate_placement_questions(
                topic, company_tier, company_type, num_questions
            )
        
        st.success(f"✅ Generated {len(question_set.questions)} questions for {company_tier} {company_type} companies")
        
        for i, q in enumerate(question_set.questions, 1):
            with st.expander(f"❓ Question {i}: {q.question} [{q.difficulty}]"):
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.markdown(f"**📋 Answer/Approach:**")
                    st.write(q.answer)
                    
                    if q.coding_solution:
                        st.markdown("**💻 Code Solution:**")
                        st.code(q.coding_solution, language="python")
                    
                    st.markdown(f"**💡 Explanation:**")
                    st.write(q.explanation)
                
                with col_right:
                    st.markdown(f"**🔍 Topic:** {q.topic_area}")
                    st.markdown(f"**⚡ Difficulty:** {q.difficulty}")
                    
                    if q.follow_up_questions:
                        st.markdown("**🤔 Follow-up Questions:**")
                        for fq in q.follow_up_questions:
                            st.write(f"• {fq}")

elif prep_mode == "📚 Study Plan":
    st.header("📚 Personalized Study Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_tier = st.selectbox("🎯 Target Company Tier:", ["Tier 1", "Tier 2"])
        prep_duration = st.selectbox("⏱️ Preparation Duration:", 
                                   ["1 Month", "2 Months", "3 Months", "6 Months"])
    
    with col2:
        focus_areas = st.multiselect("🔍 Focus Areas:", TECHNICAL_TOPICS, 
                                   default=["Data Structures & Algorithms", "System Design"])
    
    if st.button("📋 Generate Study Plan", type="primary"):
        if focus_areas:
            with st.spinner("📚 Creating personalized study plan..."):
                study_plan = placement_engine.generate_study_plan(target_tier, prep_duration, focus_areas)
            
            st.header(f"📚 {study_plan.title}")
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"**🎯 Target:** {study_plan.company_tier} Companies")
            with col_info2:
                st.markdown(f"**⏱️ Duration:** {study_plan.duration}")
            
            # Daily Schedule
            st.subheader("📅 Daily Study Schedule")
            for item in study_plan.daily_schedule:
                st.write(f"• {item}")
            
            # Week-wise Plan
            st.subheader("🗓️ Week-wise Preparation Plan")
            for week_plan in study_plan.week_wise_plan:
                with st.expander(f"📆 {week_plan.get('week', 'Week')} - {week_plan.get('focus', 'Focus Area')}"):
                    st.markdown("**🎯 Goals:**")
                    for goal in week_plan.get('goals', []):
                        st.write(f"• {goal}")
                    st.markdown(f"**📝 Practice:** {week_plan.get('practice', 'Practice recommendation')}")
            
            # Important Topics & Resources
            col_topics, col_resources = st.columns(2)
            
            with col_topics:
                st.subheader("⭐ Priority Topics")
                for topic in study_plan.important_topics:
                    st.write(f"• {topic}")
            
            with col_resources:
                st.subheader("🔗 Practice Resources")
                for resource in study_plan.practice_resources:
                    st.write(f"• {resource}")
            
            # Mock Interview Schedule
            st.subheader("🎤 Mock Interview Timeline")
            for mock in study_plan.mock_interview_schedule:
                st.write(f"• {mock}")
        else:
            st.warning("⚠️ Please select at least one focus area.")

elif prep_mode == "🧠 Time-Based Quiz":
    st.header("🧠 Time-Based Quiz Challenge")
    
    if not st.session_state.quiz_state['active']:
        # Quiz Setup
        st.markdown("### 🎯 Quiz Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quiz_topic = st.selectbox("📚 Quiz Topic:", TECHNICAL_TOPICS)
        
        with col2:
            quiz_difficulty = st.selectbox("⚡ Difficulty Level:", ["Easy", "Medium", "Hard"])
        
        with col3:
            num_quiz_questions = st.number_input("🔢 Number of Questions:", min_value=5, max_value=20, value=10)
        
        st.markdown("### ⏱️ Quiz Rules:")
        st.markdown(f"""
        - **Time Limit per Question:** {45 if quiz_difficulty == 'Easy' else 60 if quiz_difficulty == 'Medium' else 90} seconds
        - **Total Questions:** {num_quiz_questions}
        - **Scoring:** Correct answer = 10 points, Time bonus up to 5 points
        - **No going back:** Once answered, you move to the next question
        """)
        
        if st.button("🚀 Start Quiz", type="primary"):
            with st.spinner("🧠 Generating quiz questions..."):
                quiz_set = placement_engine.generate_quiz(quiz_topic, quiz_difficulty, num_quiz_questions)
            
            if quiz_set.questions:
                st.session_state.quiz_state = {
                    'active': True,
                    'questions': quiz_set.questions,
                    'current_q': 0,
                    'answers': [],
                    'start_time': time.time(),
                    'question_start_time': time.time(),
                    'time_taken': [],
                    'score': 0,
                    'topic': quiz_topic,
                    'difficulty': quiz_difficulty
                }
                st.rerun()
            else:
                st.error("Failed to generate quiz questions. Please try again.")
    
    else:
        # Active Quiz
        quiz_state = st.session_state.quiz_state
        current_q_idx = quiz_state['current_q']
        
        if current_q_idx < len(quiz_state['questions']):
            current_q = quiz_state['questions'][current_q_idx]
            
            # Progress bar
            progress = (current_q_idx + 1) / len(quiz_state['questions'])
            st.progress(progress)
            
            # Question header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"### Question {current_q_idx + 1} of {len(quiz_state['questions'])}")
            with col2:
                st.markdown(f"**Topic:** {current_q.topic_area}")
            with col3:
                st.markdown(f"**Difficulty:** {current_q.difficulty}")
            
            # Timer
            time_elapsed = time.time() - quiz_state['question_start_time']
            time_remaining = max(0, current_q.time_limit - time_elapsed)
            
            # Timer display
            timer_container = st.container()
            with timer_container:
                if time_remaining > 10:
                    st.markdown(f"⏱️ **Time Remaining: {int(time_remaining)}s**")
                else:
                    st.markdown(f"🔥 **Time Remaining: {int(time_remaining)}s**")
            
            # Question
            st.markdown(f"### {current_q.question}")
            
            # Options
            answer_key = f"answer_{current_q_idx}"
            selected_answer = st.radio(
                "Choose your answer:",
                options=range(len(current_q.options)),
                format_func=lambda x: f"{chr(65+x)}. {current_q.options[x]}",
                key=answer_key
            )
            
            # Auto-submit when time runs out
            if time_remaining <= 0:
                st.warning("⏰ Time's up!")
                # Record the answer (or -1 for no answer)
                quiz_state['answers'].append(selected_answer if answer_key in st.session_state else -1)
                quiz_state['time_taken'].append(current_q.time_limit)
                quiz_state['current_q'] += 1
                quiz_state['question_start_time'] = time.time()
                st.rerun()
            
            # Submit button
            if st.button("✅ Submit Answer", type="primary"):
                # Record answer and time
                quiz_state['answers'].append(selected_answer)
                quiz_state['time_taken'].append(time_elapsed)
                
                # Calculate score
                if selected_answer == current_q.correct_answer:
                    base_score = 10
                    time_bonus = max(0, int((current_q.time_limit - time_elapsed) / current_q.time_limit * 5))
                    quiz_state['score'] += base_score + time_bonus
                
                quiz_state['current_q'] += 1
                quiz_state['question_start_time'] = time.time()
                st.rerun()
            
        else:
            # Quiz Complete - Show Results
            st.balloons()
            st.header("🎉 Quiz Complete!")
            
            total_time = time.time() - quiz_state['start_time']
            total_questions = len(quiz_state['questions'])
            correct_answers = sum(1 for i, ans in enumerate(quiz_state['answers']) 
                                if ans == quiz_state['questions'][i].correct_answer)
            
            # Overall Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Score", f"{quiz_state['score']}/{total_questions * 15}")
            with col2:
                st.metric("Correct Answers", f"{correct_answers}/{total_questions}")
            with col3:
                st.metric("Accuracy", f"{(correct_answers/total_questions)*100:.1f}%")
            with col4:
                st.metric("Total Time", f"{int(total_time//60)}m {int(total_time%60)}s")
            
            # Performance Analysis
            st.subheader("📊 Performance Analysis")
            
            avg_time = sum(quiz_state['time_taken']) / len(quiz_state['time_taken'])
            fast_answers = sum(1 for t in quiz_state['time_taken'] if t < avg_time * 0.7)
            
            performance_text = ""
            if quiz_state['score'] >= total_questions * 12:
                performance_text = "🌟 Excellent! You're well prepared for placements!"
            elif quiz_state['score'] >= total_questions * 9:
                performance_text = "👍 Good performance! Keep practicing to improve."
            elif quiz_state['score'] >= total_questions * 6:
                performance_text = "📚 Need more practice. Focus on weak areas."
            else:
                performance_text = "💪 Keep studying! You've got room for improvement."
            
            st.info(performance_text)
            
            # Detailed Results
            st.subheader("📝 Detailed Results")
            
            for i, (question, user_answer, time_taken) in enumerate(zip(
                quiz_state['questions'], quiz_state['answers'], quiz_state['time_taken']
            )):
                correct = user_answer == question.correct_answer
                
                with st.expander(f"Question {i+1}: {'✅' if correct else '❌'} ({time_taken:.1f}s)"):
                    st.write(f"**Q:** {question.question}")
                    
                    # Show options with highlighting
                    for j, option in enumerate(question.options):
                        if j == question.correct_answer:
                            st.success(f"✅ {chr(65+j)}. {option} (Correct)")
                        elif j == user_answer and not correct:
                            st.error(f"❌ {chr(65+j)}. {option} (Your answer)")
                        else:
                            st.write(f"{chr(65+j)}. {option}")
                    
                    st.info(f"**Explanation:** {question.explanation}")
            
            # Reset Quiz
            if st.button("🔄 Take Another Quiz", type="primary"):
                st.session_state.quiz_state = {
                    'active': False,
                    'questions': [],
                    'current_q': 0,
                    'answers': [],
                    'start_time': None,
                    'question_start_time': None,
                    'time_taken': [],
                    'score': 0
                }
                st.rerun()

# Footer with tips
st.markdown("---")
st.markdown(
    """
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h4>🎯 Quick Preparation Tips:</h4>
        <ul>
            <li><strong>Tier 1 Companies:</strong> Focus on advanced DSA, system design, and leadership principles</li>
            <li><strong>Tier 2 Companies:</strong> Master fundamentals, practice communication, and show problem-solving skills</li>
            <li><strong>Time Management:</strong> Practice answering questions quickly and accurately</li>
            <li><strong>Regular Practice:</strong> Take quizzes regularly to build confidence and speed</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center; color: #888; margin-top: 20px;">
        Made by Zoya for aspiring software engineers 
    </div>
    """,
    unsafe_allow_html=True
)