JOB = main
SECDIR = sections
MDDIR  = ..

.PHONY: all clean sections

all: $(JOB).pdf

# Cheap markdown -> LaTeX preview. The submission is hand-cleaned LaTeX,
# but this gets you 90% of the way and lets you keep editing the .md files.
# Requires pandoc.
sections:
	mkdir -p $(SECDIR)
	for f in 00_abstract 01_introduction 02_background 03_design \
	         04_implementation 05_evaluation 06_related 07_conclusion; do \
	  pandoc $(MDDIR)/$$f.md \
	         --from gfm \
	         --to latex \
	         --no-highlight \
	         --top-level-division=section \
	         -o $(SECDIR)/$$f.tex; \
	done

$(JOB).pdf: $(JOB).tex sections
	pdflatex -interaction=nonstopmode $(JOB)
	-bibtex $(JOB)
	pdflatex -interaction=nonstopmode $(JOB)
	pdflatex -interaction=nonstopmode $(JOB)

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc $(JOB).pdf
	rm -rf $(SECDIR)
