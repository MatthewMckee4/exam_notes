clean:
	find . -name '*.aux' -exec rm -rf {} +
	find . -name '*.log' -exec rm -rf {} +
	find . -name '*.out' -exec rm -rf {} +
	find . -name '*.toc' -exec rm -rf {} +
	find . -name '*.lof' -exec rm -rf {} +
	find . -name '*.lot' -exec rm -rf {} +
	find . -name '*.fls' -exec rm -rf {} +
	find . -name '*.fdb_latexmk' -exec rm -rf {} +
	find . -name '*.synctex.gz' -exec rm -rf {} +
	find . -name '*.bbl' -exec rm -rf {} +
	find . -name '*.blg' -exec rm -rf {} +
	find . -name '*.brf' -exec rm -rf {} +
	find . -name '*.nav' -exec rm -rf {} +
	find . -name '*.snm' -exec rm -rf {} +
	find . -name '*.vrb' -exec rm -rf {} +
	find . -name '*.pdf' -not -path './notes/2024-2025/computing_science/data_fundamentals/fact_sheets/*' -exec rm -rf {} +

compile:
	find . -name '*.tex' -execdir pdflatex {} \;

rename:
	find . -type f -name "notes.pdf" -not -path "./.git/*" | while read -r file; do \
		dir=$$(dirname "$$file"); \
		if [ "$$dir" != "." ]; then \
			parentdir=$$(basename "$$dir"); \
			mv "$$file" "$$dir/$$parentdir-notes.pdf"; \
		fi \
	done