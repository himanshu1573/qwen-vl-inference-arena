# Report PDF

`InferTutor_Arena_Himanshu_Prajapati.pdf` — 9-page illustrated report of the RunPod runs.

Rebuild (charts read the result JSONs, then LaTeX via tectonic):

```bash
cd docs/report
uv run --python 3.12 --with matplotlib python charts.py ../../starter_code/results_infertutor fig
tectonic -X compile report.tex && mv report.pdf InferTutor_Arena_Himanshu_Prajapati.pdf
```
