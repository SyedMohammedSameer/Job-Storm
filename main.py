"""
JobStorm - Automated Job Application Document Generator
A FastAPI service that tailors resume, cover letter, and messages to job descriptions.
Uses OpenRouter API for LLM-based minimal document customization.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default model - can be changed to any OpenRouter supported model
DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"

# Paths to your gold standard templates
TEMPLATES_DIR = Path("templates")
OUTPUT_DIR = Path("output")

# Ensure directories exist
TEMPLATES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class JobDetails(BaseModel):
    """Job information from scraper"""
    job_id: str = Field(..., description="Unique identifier for the job")
    company_name: str
    job_title: str
    job_description: str
    location: Optional[str] = None
    salary_range: Optional[str] = None
    job_url: Optional[str] = None
    requirements: Optional[list[str]] = None
    responsibilities: Optional[list[str]] = None
    company_info: Optional[str] = None
    application_deadline: Optional[str] = None
    extra_fields: Optional[dict] = None


class JobBatchRequest(BaseModel):
    """Batch of jobs to process"""
    jobs: list[JobDetails]
    generate_pdf: bool = False


class CustomizationRequest(BaseModel):
    """Single job customization request"""
    job: JobDetails
    generate_pdf: bool = False


class CustomizationResponse(BaseModel):
    """Response with customized documents"""
    job_id: str
    company_name: str
    job_title: str
    resume_latex: str
    cover_letter_latex: str
    message: str
    output_dir: str
    pdf_generated: bool = False


# ============================================================================
# LLM CLIENT
# ============================================================================

class OpenRouterClient:
    """Client for OpenRouter API"""
    
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "JobStorm"
        }
    
    async def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """Send completion request to OpenRouter"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 4096
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OPENROUTER_BASE_URL,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenRouter API error: {response.text}"
                )
            
            result = response.json()
            return result["choices"][0]["message"]["content"]


# ============================================================================
# DOCUMENT CUSTOMIZER
# ============================================================================

class DocumentCustomizer:
    """Handles document customization logic"""
    
    RESUME_SYSTEM_PROMPT = """You are an expert resume tailoring assistant. Your task is to make MINIMAL, TARGETED modifications to a LaTeX resume to better align with a specific job description.

CRITICAL RULES:
1. PRESERVE the exact LaTeX structure, formatting, and commands
2. Only modify content that directly relates to the job requirements
3. Do NOT add fake experience, skills, or qualifications
4. Do NOT hallucinate or invent any information
5. Only reorder, emphasize, or slightly rephrase EXISTING content
6. Keep all personal information unchanged
7. Return ONLY the complete LaTeX code, no explanations

Allowed modifications:
- Reorder bullet points to prioritize relevant experience
- Slightly rephrase descriptions to use keywords from JD (if truthful)
- Adjust the professional summary/objective if present
- Emphasize relevant skills that already exist in the resume"""

    COVER_LETTER_SYSTEM_PROMPT = """You are an expert cover letter customization assistant. Your task is to tailor a LaTeX cover letter template to a specific job and company.

CRITICAL RULES:
1. PRESERVE the exact LaTeX structure and formatting
2. Replace placeholder text with job-specific content
3. Do NOT hallucinate or invent qualifications
4. Keep the tone professional and authentic
5. Reference specific aspects of the job description
6. Keep it concise - no more than one page worth of content
7. Return ONLY the complete LaTeX code, no explanations

Focus on:
- Why you're interested in THIS specific role at THIS company
- How your existing experience aligns with their needs
- Specific skills/achievements relevant to the job requirements"""

    MESSAGE_SYSTEM_PROMPT = """You are an expert at crafting professional outreach messages for job applications. Your task is to customize a template message for a specific job opportunity.

CRITICAL RULES:
1. Keep the message brief (under 150 words)
2. Do NOT hallucinate or make up qualifications
3. Be specific to the role and company
4. Maintain a professional but personable tone
5. Return ONLY the message text, no explanations or formatting

The message should:
- Mention the specific role and company
- Briefly highlight 1-2 relevant qualifications
- Express genuine interest
- Include a clear call to action"""

    def __init__(self, llm_client: OpenRouterClient):
        self.llm = llm_client
    
    async def customize_resume(self, template: str, job: JobDetails) -> str:
        """Customize resume for specific job"""
        user_prompt = f"""Here is the job I'm applying to:

COMPANY: {job.company_name}
POSITION: {job.job_title}
LOCATION: {job.location or 'Not specified'}

JOB DESCRIPTION:
{job.job_description}

{f"KEY REQUIREMENTS: {chr(10).join('- ' + r for r in job.requirements)}" if job.requirements else ""}

{f"KEY RESPONSIBILITIES: {chr(10).join('- ' + r for r in job.responsibilities)}" if job.responsibilities else ""}

---

Here is my current LaTeX resume to customize:

{template}

---

Return the customized LaTeX resume with minimal, truthful modifications to better align with this job."""

        return await self.llm.complete(self.RESUME_SYSTEM_PROMPT, user_prompt)
    
    async def customize_cover_letter(self, template: str, job: JobDetails) -> str:
        """Customize cover letter for specific job"""
        user_prompt = f"""Here is the job I'm applying to:

COMPANY: {job.company_name}
POSITION: {job.job_title}
LOCATION: {job.location or 'Not specified'}
{f"COMPANY INFO: {job.company_info}" if job.company_info else ""}

JOB DESCRIPTION:
{job.job_description}

{f"KEY REQUIREMENTS: {chr(10).join('- ' + r for r in job.requirements)}" if job.requirements else ""}

---

Here is my LaTeX cover letter template to customize:

{template}

---

Return the customized LaTeX cover letter tailored to this specific job and company."""

        return await self.llm.complete(self.COVER_LETTER_SYSTEM_PROMPT, user_prompt)
    
    async def customize_message(self, template: str, job: JobDetails) -> str:
        """Customize outreach message for specific job"""
        user_prompt = f"""Here is the job I'm applying to:

COMPANY: {job.company_name}
POSITION: {job.job_title}
{f"JOB URL: {job.job_url}" if job.job_url else ""}

JOB DESCRIPTION (brief):
{job.job_description[:1000]}...

---

Here is my message template to customize:

{template}

---

Return a customized, brief message for this specific opportunity."""

        return await self.llm.complete(self.MESSAGE_SYSTEM_PROMPT, user_prompt, temperature=0.4)


# ============================================================================
# PDF GENERATOR
# ============================================================================

def compile_latex_to_pdf(latex_content: str, output_path: Path, filename: str) -> Optional[Path]:
    """Compile LaTeX to PDF using pdflatex"""
    tex_file = output_path / f"{filename}.tex"
    pdf_file = output_path / f"{filename}.pdf"
    
    # Write LaTeX content
    tex_file.write_text(latex_content)
    
    try:
        # Run pdflatex twice for proper references
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_path), str(tex_file)],
                capture_output=True,
                text=True,
                timeout=60
            )
        
        if pdf_file.exists():
            # Clean up auxiliary files
            for ext in [".aux", ".log", ".out"]:
                aux_file = output_path / f"{filename}{ext}"
                if aux_file.exists():
                    aux_file.unlink()
            return pdf_file
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"PDF compilation failed: {e}")
    
    return None


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="JobStorm",
    description="Automated Job Application Document Generator",
    version="1.0.0"
)

# Initialize clients
llm_client = OpenRouterClient(OPENROUTER_API_KEY)
customizer = DocumentCustomizer(llm_client)


def load_template(template_name: str) -> str:
    """Load a LaTeX template from the templates directory"""
    template_path = TEMPLATES_DIR / template_name
    if not template_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Template not found: {template_name}. Please add it to the '{TEMPLATES_DIR}' directory."
        )
    return template_path.read_text()


def sanitize_filename(name: str) -> str:
    """Create safe filename from string"""
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()


@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "service": "JobStorm",
        "status": "running",
        "endpoints": {
            "/customize": "POST - Customize documents for a single job",
            "/batch": "POST - Process multiple jobs",
            "/templates": "GET - List available templates",
            "/output/{job_id}": "GET - Retrieve output files"
        },
        "templates_dir": str(TEMPLATES_DIR.absolute()),
        "output_dir": str(OUTPUT_DIR.absolute())
    }


@app.get("/templates")
async def list_templates():
    """List available templates"""
    templates = list(TEMPLATES_DIR.glob("*.tex"))
    return {
        "templates": [t.name for t in templates],
        "required": ["resume.tex", "cover_letter.tex", "message.txt"],
        "templates_dir": str(TEMPLATES_DIR.absolute())
    }


@app.post("/customize", response_model=CustomizationResponse)
async def customize_documents(request: CustomizationRequest):
    """Customize all documents for a single job"""
    job = request.job
    
    # Load templates
    try:
        resume_template = load_template("resume.tex")
        cover_letter_template = load_template("cover_letter.tex")
        message_template = load_template("message.txt")
    except HTTPException as e:
        raise e
    
    # Customize documents
    resume_latex = await customizer.customize_resume(resume_template, job)
    cover_letter_latex = await customizer.customize_cover_letter(cover_letter_template, job)
    message = await customizer.customize_message(message_template, job)
    
    # Clean up LLM output (remove markdown code blocks if present)
    for content in [resume_latex, cover_letter_latex]:
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    
    resume_latex = resume_latex.strip()
    if resume_latex.startswith("```"):
        resume_latex = "\n".join(resume_latex.split("\n")[1:])
    if resume_latex.endswith("```"):
        resume_latex = "\n".join(resume_latex.split("\n")[:-1])
        
    cover_letter_latex = cover_letter_latex.strip()
    if cover_letter_latex.startswith("```"):
        cover_letter_latex = "\n".join(cover_letter_latex.split("\n")[1:])
    if cover_letter_latex.endswith("```"):
        cover_letter_latex = "\n".join(cover_letter_latex.split("\n")[:-1])
    
    # Create output directory for this job
    safe_name = sanitize_filename(f"{job.job_id}_{job.company_name}_{job.job_title}")
    job_output_dir = OUTPUT_DIR / safe_name
    job_output_dir.mkdir(exist_ok=True)
    
    # Save customized documents
    (job_output_dir / "resume.tex").write_text(resume_latex)
    (job_output_dir / "cover_letter.tex").write_text(cover_letter_latex)
    (job_output_dir / "message.txt").write_text(message)
    
    # Save job details for reference
    (job_output_dir / "job_details.json").write_text(job.model_dump_json(indent=2))
    
    # Generate PDFs if requested
    pdf_generated = False
    if request.generate_pdf:
        resume_pdf = compile_latex_to_pdf(resume_latex, job_output_dir, "resume")
        cover_pdf = compile_latex_to_pdf(cover_letter_latex, job_output_dir, "cover_letter")
        pdf_generated = resume_pdf is not None and cover_pdf is not None
    
    return CustomizationResponse(
        job_id=job.job_id,
        company_name=job.company_name,
        job_title=job.job_title,
        resume_latex=resume_latex,
        cover_letter_latex=cover_letter_latex,
        message=message,
        output_dir=str(job_output_dir.absolute()),
        pdf_generated=pdf_generated
    )


@app.post("/batch")
async def batch_customize(request: JobBatchRequest):
    """Process multiple jobs"""
    results = []
    errors = []
    
    for job in request.jobs:
        try:
            result = await customize_documents(
                CustomizationRequest(job=job, generate_pdf=request.generate_pdf)
            )
            results.append({
                "job_id": result.job_id,
                "company": result.company_name,
                "position": result.job_title,
                "output_dir": result.output_dir,
                "status": "success"
            })
        except Exception as e:
            errors.append({
                "job_id": job.job_id,
                "company": job.company_name,
                "position": job.job_title,
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "processed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }


@app.get("/output/{job_id}")
async def get_output(job_id: str):
    """List output files for a job"""
    # Find matching directory
    matching_dirs = list(OUTPUT_DIR.glob(f"{job_id}*"))
    
    if not matching_dirs:
        raise HTTPException(status_code=404, detail=f"No output found for job_id: {job_id}")
    
    job_dir = matching_dirs[0]
    files = list(job_dir.glob("*"))
    
    return {
        "job_id": job_id,
        "output_dir": str(job_dir.absolute()),
        "files": [f.name for f in files]
    }


@app.get("/output/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a specific output file"""
    matching_dirs = list(OUTPUT_DIR.glob(f"{job_id}*"))
    
    if not matching_dirs:
        raise HTTPException(status_code=404, detail=f"No output found for job_id: {job_id}")
    
    file_path = matching_dirs[0] / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(file_path, filename=filename)


# ============================================================================
# CLI / MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                        JobStorm v1.0                          ║
    ║           Automated Job Application Document Generator         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Setup:                                                        ║
    ║  1. Set OPENROUTER_API_KEY environment variable               ║
    ║  2. Add templates to ./templates/:                            ║
    ║     - resume.tex (your gold standard resume)                  ║
    ║     - cover_letter.tex (your gold standard cover letter)      ║
    ║     - message.txt (your gold standard message)                ║
    ║  3. Send job JSON to /customize or /batch endpoints           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
