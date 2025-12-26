#!/bin/bash
# Auto-updates CITATION.bib with the latest DOI and version
VERSION=$(grep "version:" CITATION.cff | awk '{print $2}' | tr -d '"')
DATE=$(date +%Y-%m-%d)
DOI="10.5281/zenodo.18051097" # Update after Zenodo deposit

cat << EOB > CITATION.bib
@software{DAT_E6_Resilience_2025,
  author = {Bryan Lab},
  title = {DAT-E6-Resilience: A Computational Framework for DAT 2.0},
  version = {${VERSION}},
  doi = {${DOI}},
  url = {https://github.com/SolomonB14D3/DAT-E6-Resilience},
  year = {2025},
  month = {12}
}
EOB
echo "✅ CITATION.bib updated to version ${VERSION}"
