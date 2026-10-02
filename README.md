# ML Learning

My machine learning practice repo. Each project lives in its own folder with its own notebook, data, README and requirements file, the same way the Dataquest project walkthroughs repo is laid out.

I mostly start from a guided walkthrough, type the code myself, and then write up what I found in my own words.

## Projects

| Folder | What it does | Main tools |
|---|---|---|
| [olympic-medal-prediction](olympic-medal-prediction) | Predicts how many medals a team will win at the Olympics using linear regression | pandas, scikit-learn, seaborn |

More folders will be added as I work through new projects.

## Running a project

Every project has its own `requirements.txt`. From the repo root:

```bash
git clone https://github.com/ig-matrix/ml-learning.git
cd ml-learning

python -m venv ml-env
source ml-env/Scripts/activate      # Git Bash on Windows
# source ml-env/bin/activate        # Mac/Linux

pip install -r olympic-medal-prediction/requirements.txt
```

Then open the notebook inside the project folder in VS Code and select the `ml-env` kernel.

## Credits

Project structure and datasets come from the [Dataquest project walkthroughs](https://github.com/dataquestio/project-walkthroughs).

## Author

Ighawosa Omoma
[GitHub](https://github.com/ig-matrix) | [LinkedIn](https://linkedin.com/in/ighawosa-omoma)
