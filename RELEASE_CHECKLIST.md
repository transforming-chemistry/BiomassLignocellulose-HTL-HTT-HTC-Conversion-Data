# Release Checklist

## Pre-release tasks

- [ ] Update contact email in README.md (section 9)
- [ ] Update contact email in CITATION.cff
- [ ] Update contact email in metadata.json
- [ ] Add ORCID to CITATION.cff (if available)
- [ ] Generate schema_description.pdf from column_metadata_cleaned.csv
- [ x] Test load_and_inspect.ipynb runs without errors
- [ x] Verify all visualizations display correctly
- [ x] Run sync_dataset.sh to ensure latest data
- [ x] Check master_dataset.csv row count matches documentation (3,386)
- [x ] Verify no syntax errors in Python modules

## Documentation review

- [x ] README.md: Check for typos and clarity
- [ ] README.md: Verify all section numbers and references
- [ ] metadata.json: Validate JSON syntax
- [ ] CITATION.cff: Validate CFF v1.2.0 syntax
- [ ] LICENSE: Verify CC BY 4.0 text is complete
- [ ] Module README: Check code examples

## Quality assurance

- [ ] Verify catalyst='none' with ratio=0.0 for blank runs
- [ ] Check no duplicate yields_basis column (should be removed)
- [ ] Confirm tracking columns (_method, _imputed, _Note) at end
- [ ] Verify LRI formula in metadata
- [ ] Check Tier and Lignin_Rich columns removed from master_dataset

## Package preparation

- [ ] Create DB_release.zip containing all files
- [ ] Test ZIP extraction and file integrity
- [ ] Verify folder structure preserved
- [ ] Check file sizes reasonable (<5 MB total)

## DB upload

- [ ] Create Zenodo dataset entry
- [ ] Upload DB_release.zip
- [ ] Set license to CC BY 4.0
- [ ] Add keywords from metadata.json
- [ ] Add description from README
- [ ] Add authors/contributors
- [ ] Publish and obtain DOI

## Post-publication

- [ ] Update README.md with Zenodo (or other) DOI
- [ ] Update CITATION.cff with DOI
- [ ] Update metadata.json with DOI
- [ ] Create git tag: v1.0.0
- [ ] Push updated files to GitHub
- [ ] Create GitHub release with link to Zenodo

## Validation (post-release)

- [ ] Test Zenodo download works
- [ ] Verify DOI resolves correctly
- [ ] Test citation formats generated correctly
- [ ] Confirm README displays properly on GitHub
- [ ] Check CITATION.cff recognized by GitHub

## Optional enhancements

- [ ] Add schema_description.pdf visualization
- [ ] Create supplementary plots for paper
- [ ] Generate dataset statistics summary table
- [ ] Create video tutorial for dataset usage
- [ ] Write blog post announcement
- [ ] Share on social media / research networks
