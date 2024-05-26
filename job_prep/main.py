# Warning control
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew
import os

gh_token = os.getenv("GITHUB_TOKEN")
base_dir = ".\\job_prep\\"
fake_resume_path = base_dir + "cv.docx"
github_url = "https://github.com/pedrocavalero"
from crewai_tools import (
    FileReadTool,
    ScrapeWebsiteTool,
    DOCXSearchTool,
    SerperDevTool,
    GithubSearchTool,
)

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
github_tool = GithubSearchTool(
    github_repo=github_url,
    content_types=["code"],  # Options: code, repo, pr, issue
    gh_token=gh_token,
)
read_resume = FileReadTool(file_path=fake_resume_path)
semantic_search_resume = DOCXSearchTool(docx=fake_resume_path)

# Agent 1: Researcher
researcher = Agent(
    role="Tech Job Researcher",
    goal="Make sure to do amazing analysis on " "job posting to help job applicants",
    tools=[scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Job Researcher, your prowess in "
        "navigating and extracting critical "
        "information from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."
    ),
)

# Agent 2: Profiler
profiler = Agent(
    role="Personal Profiler for Engineers",
    goal="Do increditble research on job applicants "
    "to help them stand out in the job market",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume, github_tool],
    verbose=True,
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    ),
)

# Agent 3: Resume Strategist
resume_strategist = Agent(
    role="Resume Strategist for Engineers",
    goal="Find all the best ways to make a " "resume stand out in the job market.",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume, github_tool],
    verbose=True,
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."
    ),
)

# Agent 4: Interview Preparer
interview_preparer = Agent(
    role="Engineering Interview Preparer",
    goal="Create interview questions and talking points "
    "based on the resume and job requirements",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    ),
)

# Task for Researcher Agent: Extract Job Requirements
research_task = Task(
    description=(
        "Analyze the job posting URL provided ({job_posting_url}) "
        "to extract key skills, experiences, and qualifications "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of job requirements, including necessary "
        "skills, qualifications, and experiences."
    ),
    agent=researcher,
    # async_execution=True,
)

# Task for Profiler Agent: Compile Comprehensive Profile
profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "
        "using the GitHub ({github_url}) URLs, and personal write-up "
        "({personal_writeup}). Utilize tools to extract and "
        "synthesize information from these sources."
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "
        "project experiences, contributions, interests, and "
        "communication style."
    ),
    agent=profiler,
    # async_execution=True,
)

# Task for Resume Strategist Agent: Align Resume with Job Requirements
resume_strategy_task = Task(
    description=(
        "Using the profile and job requirements obtained from "
        "previous tasks, tailor the resume to highlight the most "
        "relevant areas. Employ tools to adjust and enhance the "
        "resume content. Make sure this is the best resume even but "
        "don't make up any information. Update every section, "
        "inlcuding the initial summary, work experience, skills, "
        "and education. All to better reflrect the candidates "
        "abilities and how it matches the job posting."
    ),
    expected_output=(
        "An updated resume that effectively highlights the candidate's "
        "qualifications and experiences relevant to the job."
    ),
    output_file=base_dir + "tailored_resume.md",
    context=[research_task, profile_task],
    agent=resume_strategist,
)

# Task for Interview Preparer Agent: Develop Interview Materials
interview_preparation_task = Task(
    description=(
        "Create a set of potential interview questions and talking "
        "points based on the tailored resume and job requirements. "
        "Utilize tools to generate relevant questions and discussion "
        "points. Make sure to use these question and talking points to "
        "help the candiadte highlight the main points of the resume "
        "and how it matches the job posting."
    ),
    expected_output=(
        "A document containing key questions and talking points "
        "that the candidate should prepare for the initial interview."
    ),
    output_file=base_dir + "interview_materials.md",
    context=[research_task, profile_task, resume_strategy_task],
    agent=interview_preparer,
)

job_application_crew = Crew(
    agents=[researcher, profiler, resume_strategist, interview_preparer],
    tasks=[
        research_task,
        profile_task,
        resume_strategy_task,
        interview_preparation_task,
    ],
    verbose=True,
)

job_application_inputs = {
    "job_posting_url": "https://www.linkedin.com/jobs/view/3892018380/",
    "github_url": github_url,
    "personal_writeup": """Highly skilled Staff Software Engineer with a proven 
    track record in architecting and developing B2C online products. Proficient 
    in Java and Python, with expertise in frameworks such as Spring, Flask, and 
    Hibernate. Experienced in working with various databases, including 
    Postgres, MySQL, and Redis. Well-versed in modern development practices 
    such as Git, Jenkins, and Docker, and familiar with Agile methodologies and 
    project management. Strong knowledge of system and software architecture, 
    design patterns, and testing. Demonstrated leadership abilities as a tech 
    lead and manager. Seeking a challenging role where I can leverage my 
    technical expertise, strategic mindset, and passion for clean code and 
    architecture to deliver innovative solutions and drive business growth.""",
}

### this execution will take a few minutes to run
result = job_application_crew.kickoff(inputs=job_application_inputs)
