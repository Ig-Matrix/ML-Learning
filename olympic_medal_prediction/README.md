# Olympic Medal Prediction

A beginner machine learning project that predicts how many medals a country will win at the Olympics, using a linear regression model trained on historical team data.

I built this by following the Dataquest [beginner ML project walkthrough](https://github.com/dataquestio/project-walkthroughs/tree/master/beginner_ml), typing the code myself in a Jupyter notebook inside VS Code and writing my own notes as I went.

## The question

Can we predict how many medals a team will win at a given Games using only past data?

## Dataset

`teams.csv` has one row per team per Games. The columns I used:

| Column | Meaning |
|---|---|
| `team` | Team code, e.g. USA, NGR |
| `country` | Country name |
| `year` | Year of the Games |
| `athletes` | Number of athletes the team sent |
| `age` | Average age of the team's athletes |
| `prev_medals` | Medals the team won at the previous Games |
| `medals` | Medals won (the target) |

The file is the cleaned version of the Olympic athlete events data, taken from the Dataquest walkthrough repo.

## Approach

1. **Explore the data.** I checked which columns move together with `medals` using a correlation matrix.
2. **Clean it.** I dropped rows with missing values.
3. **Split by time.** Training data is every Games before 2012 (1,609 rows). Test data is 2012 onward (405 rows). Splitting by year means the model never sees the future.
4. **Train.** A scikit-learn `LinearRegression` with two predictors: `athletes` and `prev_medals`.
5. **Predict and clean up.** Negative predictions were set to 0 and the rest rounded to whole medals.
6. **Evaluate.** I used mean absolute error, then looked at the error per team and compared it to each team's average medals.

## Findings

- `prev_medals` (0.92) and `athletes` (0.84) are strongly correlated with `medals`. `age` and `year` are close to 0, so I left them out.
- `athletes` and `prev_medals` are also correlated with each other (0.81), so they carry some of the same information.
- Medal counts are heavily right-skewed. Most team entries won only a few medals and a small number of teams won a lot.
- Learned equation: `medals = 0.07 * athletes + 0.75 * prev_medals - 2.142`
- The model's mean absolute error on the test set is about **3.3 medals**, meaning predictions are off by roughly 3 medals on average.
- The error ratio (error divided by the team's average medals) is low for teams that win plenty of medals, such as France, Canada and the USA. It is much higher for teams that win very few, where a small miss looks big in proportion.

## Ideas to improve it

- Add more predictors
- Try a different model

## How to run it

Set up the environment from the repo root (see the [main README](../README.md)), then:

```bash
pip install -r olympic-medal-prediction/requirements.txt
```

Open `medal_prediction.ipynb` in VS Code and select the `ml-env` kernel. Keep `teams.csv` in the same folder as the notebook.

## Tools

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib, Jupyter

## Credits

Project structure and dataset from the [Dataquest project walkthroughs](https://github.com/dataquestio/project-walkthroughs).
