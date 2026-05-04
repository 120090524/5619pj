PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

demo:
	$(PYTHON) -m sentry_jury.cli --config configs/course_project_mock.yaml

proposal:
	cd proposal && pdflatex proposal.tex && bibtex proposal && pdflatex proposal.tex && pdflatex proposal.tex

clean:
	rm -rf runs
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find proposal -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.blg" -o -name "*.fdb_latexmk" -o -name "*.fls" -o -name "*.log" -o -name "*.out" \) -delete
