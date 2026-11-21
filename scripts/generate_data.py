"""Generate a synthetic India job feed + a labelled retrieval eval set.

Synthetic, so every posting has a known role family — that is what makes relevance labels possible.
Swap in a real feed with the same JSONL schema (see aeromatch.ingest.JobIn) for production."""
import json
import random
from pathlib import Path

random.seed(14)
OUT = Path(__file__).resolve().parent.parent / "data"

# family: (titles, core skills, optional skills, responsibilities, plain-language query paraphrases)
FAMILIES = {
    "backend_python": (
        ["Backend Engineer (Python)", "Python Developer", "Software Engineer - Backend", "API Developer"],
        ["python", "fastapi", "postgresql", "rest api"], ["django", "redis", "docker", "celery", "aws"],
        ["design and ship REST APIs consumed by our mobile apps", "own services end to end in production",
         "model data in PostgreSQL and tune slow queries"],
        ["build the server side of web products and expose endpoints for apps",
         "write services that power an app behind the scenes"]),
    "backend_java": (
        ["Java Developer", "Backend Engineer (Java)", "Spring Boot Developer"],
        ["java", "spring boot", "microservices", "mysql"], ["kafka", "hibernate", "kubernetes", "aws"],
        ["build high-throughput microservices on the JVM", "integrate payment and ledger systems"],
        ["work on JVM services for a fintech platform", "enterprise server applications on the JVM"]),
    "backend_go": (
        ["Go Developer", "Golang Engineer", "Backend Engineer (Go)"],
        ["go", "grpc", "postgresql", "microservices"], ["kubernetes", "redis", "kafka"],
        ["write low-latency services in Go", "design gRPC contracts between services"],
        ["low latency systems programming for distributed services", "concurrent network services"]),
    "frontend": (
        ["Frontend Developer", "React Developer", "UI Engineer", "Frontend Engineer (React)"],
        ["react", "typescript", "javascript", "css"], ["next.js", "redux", "tailwind", "jest"],
        ["build pixel-perfect, accessible user interfaces", "own the web app's component library"],
        ["make the screens users click on in the browser", "turn Figma designs into a working website"]),
    "fullstack": (
        ["Full Stack Developer", "Full-Stack Engineer (MERN)", "Software Engineer - Full Stack"],
        ["react", "node.js", "mongodb", "javascript"], ["express", "typescript", "aws", "docker"],
        ["ship features across the React frontend and Node backend", "own features from database to UI"],
        ["work on both the website and the server behind it", "build complete product features alone"]),
    "android": (
        ["Android Developer", "Mobile Engineer (Android)", "Kotlin Developer"],
        ["kotlin", "android", "jetpack compose"], ["java", "firebase", "retrofit", "mvvm"],
        ["build features for our Android app with millions of installs", "improve app startup time"],
        ["build apps for phones that run Google's OS", "make our Play Store app better"]),
    "ios": (
        ["iOS Developer", "Mobile Engineer (iOS)", "Swift Developer"],
        ["swift", "ios", "swiftui"], ["objective-c", "xcode", "core data", "combine"],
        ["build delightful iPhone experiences", "own the iOS release pipeline"],
        ["build iPhone apps", "work on our App Store app for Apple devices"]),
    "flutter": (
        ["Flutter Developer", "Cross-Platform Mobile Developer"],
        ["flutter", "dart"], ["firebase", "android", "ios", "bloc"],
        ["ship one codebase to Android and iOS", "build reusable widgets"],
        ["one codebase apps for both phone platforms", "cross platform app development"]),
    "data_analyst": (
        ["Data Analyst", "Business Analyst", "Product Analyst"],
        ["sql", "excel", "power bi"], ["python", "tableau", "statistics", "pandas"],
        ["build dashboards for business teams", "analyse funnels and run A/B test readouts"],
        ["make dashboards and reports for managers", "dig into numbers to answer business questions"]),
    "data_scientist": (
        ["Data Scientist", "Applied Scientist", "Data Scientist - ML"],
        ["python", "machine learning", "statistics", "pandas"], ["scikit-learn", "xgboost", "sql", "pytorch"],
        ["build predictive models for churn and pricing", "design experiments and causal analyses"],
        ["predict customer behaviour with models", "use statistics and modelling to drive decisions"]),
    "ml_engineer": (
        ["Machine Learning Engineer", "ML Engineer", "MLOps Engineer"],
        ["python", "pytorch", "mlops", "docker"], ["kubernetes", "mlflow", "aws", "tensorflow"],
        ["take models from notebook to production", "build training and serving pipelines"],
        ["deploy trained models to production and monitor them", "productionise deep learning models"]),
    "genai_engineer": (
        ["GenAI Engineer", "LLM Engineer", "AI Engineer (LLMs)", "Applied AI Engineer"],
        ["python", "llm", "rag", "embeddings"], ["langchain", "vector database", "prompt engineering", "fastapi"],
        ["build agents that use tools and retrieval", "evaluate and improve LLM output quality"],
        ["build chatbots on top of large language models", "make AI assistants that answer from documents"]),
    "data_engineer": (
        ["Data Engineer", "Big Data Engineer", "Analytics Engineer"],
        ["python", "sql", "spark", "airflow"], ["kafka", "dbt", "snowflake", "aws", "bigquery"],
        ["build batch and streaming data pipelines", "own the data warehouse and its models"],
        ["move and transform large datasets into a warehouse", "build ETL pipelines"]),
    "devops": (
        ["DevOps Engineer", "Site Reliability Engineer", "Platform Engineer", "Cloud Engineer"],
        ["kubernetes", "docker", "aws", "terraform"], ["linux", "ci/cd", "prometheus", "ansible"],
        ["automate infrastructure as code", "keep production reliable and observable"],
        ["manage cloud servers and deployments", "keep our systems up and automate releases"]),
    "security": (
        ["Security Engineer", "Application Security Engineer", "Cybersecurity Analyst"],
        ["security", "penetration testing", "owasp"], ["siem", "network security", "python", "cloud security"],
        ["run threat modelling and pen tests", "triage vulnerabilities and harden systems"],
        ["find vulnerabilities before attackers do", "ethical hacking and defending our systems"]),
    "qa": (
        ["QA Engineer", "SDET", "Test Automation Engineer"],
        ["selenium", "test automation", "java"], ["cypress", "api testing", "postman", "jmeter"],
        ["build automated regression suites", "own release quality gates"],
        ["write automated tests for web apps", "make sure releases don't break things"]),
    "embedded": (
        ["Embedded Software Engineer", "Firmware Engineer", "Embedded C Developer"],
        ["c", "embedded c", "rtos", "microcontrollers"], ["c++", "arm", "linux", "iot"],
        ["write firmware for ARM microcontrollers", "bring up new boards and drivers"],
        ["program low level hardware devices", "firmware for IoT devices"]),
    "cpp_systems": (
        ["C++ Developer", "Systems Engineer (C++)", "Low Latency Developer"],
        ["c++", "linux", "multithreading", "data structures"], ["networking", "stl", "python", "algorithms"],
        ["build low-latency trading systems", "optimise performance-critical code paths"],
        ["high performance trading infrastructure", "performance critical native code"]),
    "ui_ux": (
        ["UI/UX Designer", "Product Designer", "UX Designer"],
        ["figma", "user research", "prototyping"], ["design systems", "wireframing", "usability testing"],
        ["design end-to-end product flows", "run user interviews and usability tests"],
        ["design how an app looks and feels", "create mockups and talk to users"]),
    "product_manager": (
        ["Product Manager", "Associate Product Manager", "Technical Product Manager"],
        ["product management", "roadmapping", "stakeholder management"], ["sql", "analytics", "jira", "a/b testing"],
        ["own the roadmap for a core product area", "write PRDs and prioritise with engineering"],
        ["decide what features to build next", "own a product's direction and priorities"]),
    "sales": (
        ["Sales Executive", "Business Development Executive", "Account Executive"],
        ["sales", "lead generation", "negotiation"], ["crm", "b2b sales", "communication", "salesforce"],
        ["close deals with mid-market customers", "build a pipeline of qualified leads"],
        ["sell software to businesses", "find new clients and close deals"]),
    "digital_marketing": (
        ["Digital Marketing Executive", "Performance Marketer", "SEO Specialist", "Growth Marketer"],
        ["seo", "google ads", "social media marketing"], ["content marketing", "google analytics", "meta ads"],
        ["run paid campaigns and optimise CAC", "grow organic traffic"],
        ["run online ad campaigns and grow website traffic", "get more users through ads and search"]),
    "content_writer": (
        ["Content Writer", "Technical Writer", "Copywriter"],
        ["content writing", "copywriting", "editing"], ["seo", "documentation", "research"],
        ["write long-form blogs and product docs", "own the brand voice"],
        ["write articles and product documentation", "writing blogs and website copy"]),
    "hr": (
        ["HR Executive", "Talent Acquisition Specialist", "Technical Recruiter"],
        ["recruitment", "sourcing", "onboarding"], ["hrms", "employee engagement", "linkedin recruiter"],
        ["hire engineers across levels", "run onboarding and employee engagement"],
        ["hire people for tech teams", "manage hiring and onboarding of new employees"]),
}

CITIES = ["Bengaluru", "Gurugram", "Hyderabad", "Pune", "Mumbai", "Chennai", "Noida", "Delhi", "Kolkata"]
CITY_W = [26, 14, 14, 11, 11, 9, 7, 5, 3]
MODES, MODE_W = ["onsite", "hybrid", "remote"], [45, 35, 20]
EXP_BANDS = [(0, 1), (0, 2), (1, 3), (2, 5), (3, 6), (5, 8), (7, 12)]
PREFIX = ["Nova", "Zen", "Blue", "Kite", "Arc", "Pixel", "Saffron", "Indigo", "Orbit", "Lumen", "Tidal",
          "Quark", "Banyan", "Monsoon", "Vertex", "Cobalt", "Peak", "Rapid", "Swift", "Teal"]
SUFFIX = ["Labs", "Tech", "Systems", "AI", "Pay", "Health", "Works", "Commerce", "Cloud", "Logistics", "Finserv"]
COMPANIES = [f"{p}{s}" for p in PREFIX for s in SUFFIX]
OPENERS = ["We are a fast-growing startup", "Join a profitable SaaS company", "We are a Series B fintech",
           "Join our product engineering team", "We are hiring for a high-ownership role",
           "Be part of a small, senior team"]


def salary(exp_min: int) -> tuple[float, float]:
    lo = round(3 + exp_min * 2.6 + random.uniform(0, 4), 1)
    return lo, round(lo * random.uniform(1.3, 1.8), 1)


def make_job(i: int, family: str) -> tuple[dict, str]:
    titles, core, optional, duties, _ = FAMILIES[family]
    exp_min, exp_max = random.choice(EXP_BANDS)
    skills = random.sample(core, k=max(2, len(core) - random.randint(0, 1))) + \
        random.sample(optional, k=random.randint(1, min(3, len(optional))))
    s_lo, s_hi = salary(exp_min)
    title = random.choice(titles)
    if exp_min >= 5 and random.random() < 0.6:
        title = "Senior " + title
    elif exp_max <= 2 and random.random() < 0.4:
        title = random.choice(["Junior ", "Associate ", ""]) + title
    company = random.choice(COMPANIES)
    desc = (f"{random.choice(OPENERS)}. As a {title} at {company} you will "
            f"{' and '.join(random.sample(duties, k=min(2, len(duties))))}. "
            f"You have {exp_min}-{exp_max} years of experience and are comfortable with "
            f"{', '.join(skills[:3])}.")
    job = dict(external_id=f"J{i:05d}", title=title, company=company, description=desc, skills=skills,
               city=random.choices(CITIES, CITY_W)[0], work_mode=random.choices(MODES, MODE_W)[0],
               exp_min=exp_min, exp_max=exp_max, salary_min_lpa=s_lo, salary_max_lpa=s_hi)
    return job, family


def make_query(qid: int, family: str) -> dict:
    titles, core, _, _, paraphrases = FAMILIES[family]
    kind = ["title", "skills", "paraphrase"][qid % 3]
    if kind == "title":
        text = random.choice(["looking for", "any openings for", "show me", "I want"]) + " " + \
            random.choice(titles).lower() + " roles"
    elif kind == "skills":
        picked = random.sample(core, k=min(3, len(core)))
        text = f"I know {', '.join(picked[:-1])} and {picked[-1]} - what jobs fit me?"
    else:
        text = "I want to " + random.choice(paraphrases)
    filters = {}
    if random.random() < 0.5:
        filters["city"] = random.choices(CITIES[:6], CITY_W[:6])[0]
    if random.random() < 0.3:
        filters["work_mode"] = random.choice(MODES)
    if random.random() < 0.4:
        filters["years_experience"] = random.choice([0, 1, 2, 4])
    return dict(qid=qid, kind=kind, text=text, family=family, filters=filters)


def main(n_jobs: int = 3000, n_queries: int = 144) -> None:
    OUT.mkdir(exist_ok=True)
    fams = list(FAMILIES)
    with open(OUT / "jobs.jsonl", "w") as fj, open(OUT / "job_families.json", "w") as ff:
        labels = {}
        for i in range(n_jobs):
            job, fam = make_job(i, fams[i % len(fams)] if i < len(fams) * 20 else random.choice(fams))
            labels[job["external_id"]] = fam
            fj.write(json.dumps(job) + "\n")
        json.dump(labels, ff)
    with open(OUT / "eval_queries.jsonl", "w") as fq:
        for q in range(n_queries):
            fq.write(json.dumps(make_query(q, fams[q % len(fams)])) + "\n")
    print(f"wrote {n_jobs} jobs, {n_queries} queries, {len(fams)} role families -> {OUT}")


if __name__ == "__main__":
    main()
